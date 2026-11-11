"""SVGD over bounded acceleration sequences, with hard-graph CP acceptance."""
from dataclasses import replace
from time import perf_counter
import logging
import jax
import jax.numpy as jnp
import numpy as np
from planning.dynamics import project_accelerations, rollout
from planning.acceptance import separation_acceptance
from planning.svgd_cp_planner import _smooth_cost_single, cp_cost, make_smooth_cp_value_and_grad, svgd_direction

logger = logging.getLogger(__name__)


def bounded_value_and_grad(objective):
    """Evaluate every particle without materializing all pair-distance tapes."""
    return jax.jit(lambda controls: jax.lax.map(jax.value_and_grad(objective), controls, batch_size=8))


def plan_double_integrator(start, goal, initial_controls, scorer, config, iterations, seed=42, record_particles=False, compare_cp_only=False):
    start, goal = np.asarray(start, np.float32), np.asarray(goal, np.float32)
    controls = np.asarray(initial_controls, np.float64).copy()
    if start.shape != goal.shape or start.ndim != 2 or start.shape[-1] != 2:
        raise ValueError("start and goal must have shape (robots, 2)")
    if controls.ndim != 4 or controls.shape[2:] != start.shape or min(controls.shape[:2]) < 2:
        raise ValueError("controls must have shape (particles >= 2, steps >= 2, robots, 2)")
    if not np.isfinite(controls).all() or not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("states and controls must be finite")
    if iterations < 1 or config.sampling_time <= 0 or config.maximum_acceleration <= 0 or config.maximum_speed <= 0:
        raise ValueError("iterations, sampling time and bounds must be positive")
    if config.fixed_goal:
        raise ValueError("fixed_goal is unsupported with double integrator: endpoints must follow the dynamics")
    if compare_cp_only and not config.collision_filter:
        raise ValueError("compare_cp_only requires the primary collision filter to be enabled")
    if config.robot_radius is not None:
        if not np.isfinite(config.robot_radius) or config.robot_radius <= 0:
            raise ValueError("robot_radius must be finite and positive")
        if config.minimum_separation < 2*config.robot_radius:
            raise ValueError("minimum_separation must be at least twice robot_radius")
    if scorer is None or config.cp_gradient not in ("spsa", "autodiff"):
        raise ValueError("a CP scorer and valid cp_gradient are required")
    if config.cp_directions < 1 or config.cp_smoothing <= 0 or not 0 <= config.required_robustness < 26:
        raise ValueError("invalid CP probe settings or robustness target")
    if np.asarray(config.initial_velocity).shape != (2,) or np.asarray(config.goal_velocity).shape != (2,):
        raise ValueError("initial_velocity and goal_velocity must be 2-D vectors")
    dt = config.sampling_time
    simulate = jax.jit(lambda u: rollout(u, start, jnp.asarray(config.initial_velocity), dt))
    project = jax.jit(lambda u: project_accelerations(u, config.maximum_acceleration))
    controls = np.asarray(project(controls), dtype=np.float64)
    position_config = replace(config, speed_weight=0.)
    def objective(u):
        p, v = rollout(u, start, jnp.asarray(config.initial_velocity), dt)
        speed = jnp.sqrt(jnp.sum(v*v, axis=-1) + 1e-12)
        return (_smooth_cost_single(p, start, goal, position_config)
                + config.speed_weight * jnp.mean(jnp.maximum(speed-config.maximum_speed, 0.)**2)
                + config.terminal_velocity_weight * jnp.mean((v[-1]-jnp.asarray(config.goal_velocity))**2)
                + config.control_weight * jnp.mean(u*u))
    base = bounded_value_and_grad(objective)
    learned = (make_smooth_cp_value_and_grad(scorer.model, scorer.quantile, config)
               if config.cp_weight and config.cp_gradient == "autodiff" else None)
    rng = np.random.default_rng(seed)
    snapshots, costs, bounds, history = [], [], [], []
    best = None
    best_cp_only = None
    cp_incumbents = []
    population_acceptance = []
    passing_count = 0
    cp_passing_count=0
    collision_rejected_count=0
    rejection_reasons={}
    def evaluate(u, population):
        nonlocal best, best_cp_only, passing_count,cp_passing_count,collision_rejected_count
        logger.info('Population %d: rollout and base cost/gradient', population)
        p, v = simulate(jnp.asarray(u, dtype=jnp.float32))
        total, gradient = base(jnp.asarray(u, dtype=jnp.float32))
        total, gradient = np.asarray(total, dtype=float), np.array(gradient, dtype=float)
        logger.info('Population %d: hard-graph CP bounds', population)
        hard_cost, lower = cp_cost(np.asarray(p), scorer, config)
        if learned is not None:
            logger.info('Population %d: smooth GNN cost/gradient', population)
            learned_cost, position_gradient = learned(p)
            _, pullback = jax.vjp(lambda control: simulate(control)[0], jnp.asarray(u, dtype=jnp.float32))
            gradient += np.asarray(pullback(position_gradient)[0])
            total += np.asarray(learned_cost)
        else:
            total += hard_cost
        eligible = (lower >= config.required_robustness) & np.isfinite(total)
        cp_passing_count += int(eligible.sum())
        indices = np.flatnonzero(eligible)
        previously_accepted=passing_count
        for index in indices:
            index=int(index)
            if compare_cp_only and (best_cp_only is None or total[index] < best_cp_only['cost']):
                best_cp_only = dict(index=index, cost=float(total[index]), minimum_lower_bound=int(lower[index]),
                    selected_population=population, positions=np.asarray(p[index]).copy(),
                    velocities=np.asarray(v[index]).copy(), controls=u[index].copy(), separation_check=None)
            separation=None
            if config.collision_filter:
                separation=separation_acceptance(start,np.asarray(p[index]),config.minimum_separation,
                    robot_radius=config.robot_radius,velocities=np.asarray(v[index]),controls=u[index],
                    dt=dt,initial_velocity=config.initial_velocity)
                if not separation['zero_collision']:
                    collision_rejected_count+=1
                    reason=separation['rejection_reason']
                    rejection_reasons[reason]=rejection_reasons.get(reason,0)+1
                    continue
            passing_count+=1
            if best is None or total[index] < best['cost']:
                best = dict(index=index, cost=float(total[index]), minimum_lower_bound=int(lower[index]),
                            selected_population=population, positions=np.asarray(p[index]).copy(),
                            velocities=np.asarray(v[index]).copy(), controls=u[index].copy(),
                            separation_check=separation)
        if record_particles:
            snapshots.append(np.asarray(p).copy()); costs.append(total.copy()); bounds.append(lower.copy())
        if compare_cp_only:
            cp_incumbents.append(None if best_cp_only is None else (best_cp_only['cost'],best_cp_only['minimum_lower_bound']))
        population_acceptance.append(dict(population=population,cp_passing_candidates=int(eligible.sum()),
            accepted_candidates=passing_count-previously_accepted))
        logger.info('Population %d: CP passing %d, joint accepted %d',population,int(eligible.sum()),passing_count-previously_accepted)
        return total, lower, gradient
    total, lower, gradient = evaluate(controls, 0)
    for iteration in range(iterations):
        began = perf_counter()
        if config.cp_weight and learned is None:
            estimate = np.zeros_like(controls)
            for _ in range(config.cp_directions):
                noise = rng.normal(size=controls.shape)
                positive = np.asarray(simulate(project(controls+config.cp_smoothing*noise))[0])
                negative = np.asarray(simulate(project(controls-config.cp_smoothing*noise))[0])
                slopes = (cp_cost(positive, scorer, config)[0]-cp_cost(negative, scorer, config)[0])/(2*config.cp_smoothing)
                estimate += slopes[:, None, None, None]*noise
            gradient += estimate/config.cp_directions
        direction = svgd_direction(controls, -gradient)
        updated = controls + config.svgd_step_size*direction
        if not np.isfinite(updated).all():
            raise FloatingPointError("non-finite double-integrator control update")
        controls = np.asarray(project(updated), dtype=float)
        total, lower, gradient = evaluate(controls, iteration+1)
        index = int(np.argmin(total))
        history.append(dict(best_cost=float(total[index]), best_lower_bound=int(lower[index]),
                            incumbent_cost=None if best is None else best['cost'],
                            incumbent_lower_bound=None if best is None else best['minimum_lower_bound'],
                            seconds=perf_counter()-began))
    report = dict(status="feasible" if best is not None else "no_feasible_trajectory",
                  dynamics="double_integrator", cp_gradient=config.cp_gradient,
                  optimization_cp_cost="continuous_expected_class_surrogate" if learned is not None else "hard_cp_shortfall",
                  history=history, passing_candidates=passing_count, evaluated_candidates=len(controls)*(iterations+1))
    report['population_acceptance']=population_acceptance
    report.update(collision_filter_enabled=config.collision_filter,cp_passing_candidates=cp_passing_count,
                  collision_rejected_candidates=collision_rejected_count,
                  collision_rejection_reasons=rejection_reasons,separation_threshold_m=config.minimum_separation)
    for key in ('index', 'cost', 'minimum_lower_bound', 'selected_population'):
        report[key] = None if best is None else best[key]
    if best is not None:
        if best['separation_check'] is not None:report['separation_check']=best['separation_check']
        report['accelerations'] = best['controls'].tolist()
        report['velocities'] = best['velocities'].tolist()
        report['maximum_acceleration_norm'] = float(np.linalg.norm(best['controls'], axis=-1).max())
        report['maximum_velocity_norm'] = float(np.linalg.norm(best['velocities'], axis=-1).max())
        report['terminal_velocity_mean_norm'] = float(np.linalg.norm(best['velocities'][-1], axis=-1).mean())
    if record_particles:
        report.update(particle_snapshots=np.asarray(snapshots), population_costs=np.asarray(costs), population_bounds=np.asarray(bounds))
    if compare_cp_only:
        # The final acceptance rule never feeds back into the optimizer. These
        # are exactly the CP-only and joint selections from identical populations.
        alternate = {k:v for k,v in report.items() if k not in
                     ('accelerations','velocities','maximum_acceleration_norm','maximum_velocity_norm',
                      'terminal_velocity_mean_norm','separation_check','particle_snapshots','population_costs','population_bounds')}
        alternate.update(status='feasible' if best_cp_only is not None else 'no_feasible_trajectory',
            collision_filter_enabled=False, passing_candidates=cp_passing_count,
            collision_rejected_candidates=0, collision_rejection_reasons={})
        alternate['population_acceptance']=[dict(item,accepted_candidates=item['cp_passing_candidates'])
                                            for item in population_acceptance]
        alternate['history'] = [dict(item,incumbent_cost=None if incumbent is None else incumbent[0],
            incumbent_lower_bound=None if incumbent is None else incumbent[1])
            for item,incumbent in zip(history,cp_incumbents[1:])]
        for key in ('index','cost','minimum_lower_bound','selected_population'):
            alternate[key] = None if best_cp_only is None else best_cp_only[key]
        if best_cp_only is not None:
            alternate.update(accelerations=best_cp_only['controls'].tolist(),
                velocities=best_cp_only['velocities'].tolist(),
                maximum_acceleration_norm=float(np.linalg.norm(best_cp_only['controls'],axis=-1).max()),
                maximum_velocity_norm=float(np.linalg.norm(best_cp_only['velocities'],axis=-1).max()),
                terminal_velocity_mean_norm=float(np.linalg.norm(best_cp_only['velocities'][-1],axis=-1).mean()))
        report['cp_only_comparison'] = dict(waypoints=None if best_cp_only is None else best_cp_only['positions'].tolist(),
                                           planner_report=alternate)
    return (None if best is None else best['positions']), report
