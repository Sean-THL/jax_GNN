"""SVGD over bounded acceleration sequences, with hard-graph CP acceptance."""
from dataclasses import replace
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from planning.dynamics import project_accelerations, rollout
from planning.svgd_cp_planner import _smooth_cost_single, cp_cost, make_smooth_cp_value_and_grad, svgd_direction


def plan_double_integrator(start, goal, initial_controls, scorer, config, iterations, seed=42, record_particles=False):
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
    base = jax.jit(jax.vmap(jax.value_and_grad(objective)))
    learned = (make_smooth_cp_value_and_grad(scorer.model, scorer.quantile, config)
               if config.cp_weight and config.cp_gradient == "autodiff" else None)
    rng = np.random.default_rng(seed)
    snapshots, costs, bounds, history = [], [], [], []
    best = None
    passing_count = 0
    def evaluate(u, population):
        nonlocal best, passing_count
        p, v = simulate(jnp.asarray(u, dtype=jnp.float32))
        total, gradient = base(jnp.asarray(u, dtype=jnp.float32))
        total, gradient = np.asarray(total, dtype=float), np.array(gradient, dtype=float)
        hard_cost, lower = cp_cost(np.asarray(p), scorer, config)
        if learned is not None:
            learned_cost, position_gradient = learned(p)
            _, pullback = jax.vjp(lambda control: simulate(control)[0], jnp.asarray(u, dtype=jnp.float32))
            gradient += np.asarray(pullback(position_gradient)[0])
            total += np.asarray(learned_cost)
        else:
            total += hard_cost
        eligible = (lower >= config.required_robustness) & np.isfinite(total)
        passing_count += int(eligible.sum())
        indices = np.flatnonzero(eligible)
        if len(indices):
            index = int(indices[np.argmin(total[indices])])
            if best is None or total[index] < best['cost']:
                best = dict(index=index, cost=float(total[index]), minimum_lower_bound=int(lower[index]),
                            selected_population=population, positions=np.asarray(p[index]).copy(),
                            velocities=np.asarray(v[index]).copy(), controls=u[index].copy())
        if record_particles:
            snapshots.append(np.asarray(p).copy()); costs.append(total.copy()); bounds.append(lower.copy())
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
    for key in ('index', 'cost', 'minimum_lower_bound', 'selected_population'):
        report[key] = None if best is None else best[key]
    if best is not None:
        report['accelerations'] = best['controls'].tolist()
        report['velocities'] = best['velocities'].tolist()
        report['maximum_acceleration_norm'] = float(np.linalg.norm(best['controls'], axis=-1).max())
        report['maximum_velocity_norm'] = float(np.linalg.norm(best['velocities'], axis=-1).max())
        report['terminal_velocity_mean_norm'] = float(np.linalg.norm(best['velocities'][-1], axis=-1).mean())
    if record_particles:
        report.update(particle_snapshots=np.asarray(snapshots), population_costs=np.asarray(costs), population_bounds=np.asarray(bounds))
    return (None if best is None else best['positions']), report
