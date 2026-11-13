"""Experimental SVGD planner using a GATv2 conformal robustness cost.

Particles are short 2-D joint robot trajectories with one waypoint per control
step. This is a prototype: the communication radius and simple point-mass
motion model must be replaced with the robot system's actual definitions.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import flax.nnx as nnx
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from conformal.scores import (
    calibrate_underbound,
    fit_lac,
    lac_prediction_sets,
    predict_underbounds,
)
from gnn.variants import model_class, predict_probabilities, prepare_records, read_records
from planning.connectivity import adjacency, smooth_adjacency
from planning.acceptance import separation_acceptance

MAX_PLANNING_ROBOTS = 100


@dataclass(frozen=True)
class PlannerConfig:
    communication_radius: float
    minimum_separation: float
    maximum_speed: float
    sampling_time: float
    required_robustness: int
    connectivity_mode: str = "smooth"
    connectivity_sigma: float | None = None
    cp_kind: str = "one_sided"  # or "lac"
    cp_coverage: float = 0.9
    cp_weight: float = 5.0
    goal_weight: float = 1.0
    path_weight: float = 0.05
    collision_weight: float = 10.0
    speed_weight: float = 10.0
    degree_weight: float = 2.0
    smooth_temperature: float = 0.1
    svgd_step_size: float = 0.05
    cp_smoothing: float = 0.1
    cp_directions: int = 2
    cp_gradient: str = "autodiff"  # continuous GNN surrogate; "spsa" for discrete probes
    fixed_goal: bool = False
    dynamics: str = "waypoints"  # legacy, double_integrator, or single_integrator
    maximum_acceleration: float = 1.0  # Euclidean norm, m/s^2
    initial_velocity: tuple = (0.0, 0.0)
    goal_velocity: tuple = (0.0, 0.0)
    terminal_velocity_weight: float = 2.0
    control_weight: float = 0.01
    robot_radius: float | None = None  # metres; None for historical point-only scenarios
    collision_filter: bool = True  # hard final separation acceptance, in addition to CP

    @property
    def maximum_step(self) -> float:
        return self.maximum_speed * self.sampling_time  # dt is sampling_time.


def communication_records(
    positions: np.ndarray,
    radius: float,
    mode: str = "hard",
    sigma: float | None = None,
) -> list[dict]:
    """Create binary compact graphs or weighted dense equation-(7) graphs."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape (graphs, robots, 2)")
    n = positions.shape[1]
    if not 1 <= n <= MAX_PLANNING_ROBOTS or radius <= 0:
        raise ValueError(f"robots must be 1..{MAX_PLANNING_ROBOTS} and radius must be positive")
    if mode not in ("hard", "smooth"):
        raise ValueError("connectivity mode must be 'hard' or 'smooth'")
    if sigma is not None and sigma <= 0:
        raise ValueError("connectivity sigma must be positive")
    weights = np.asarray(adjacency(positions, radius, mode, sigma))
    records = []
    for graph_weights in weights:
        if mode == "hard":
            source, target = np.nonzero(graph_weights)
            edge_weight = np.ones(len(source), dtype=np.float32)
        else:
            source, target = np.nonzero(~np.eye(n, dtype=bool))
            edge_weight = graph_weights[source, target].astype(np.float32)
        degree = graph_weights.sum(axis=1, dtype=np.float32)
        features = np.column_stack((degree, np.ones(n, dtype=np.float32)))
        records.append({
            "features": features,
            "edge_index": np.stack((source, target)).astype(np.int32),
            "edge_attr": edge_weight[:, None],
            "robustness": 0,  # batching requires a label; candidate labels are unknown
        })
    return records


class ConformalGraphScorer:
    def __init__(self, checkpoint: Path, calibration_data: Path, coverage: float, kind: str):
        if kind not in ("lac", "one_sided"):
            raise ValueError("kind must be 'lac' or 'one_sided'")
        metadata = json.loads(checkpoint.with_suffix(".json").read_text())
        if (metadata["representation"], metadata["convolution"]) != ("new", "gatv2"):
            raise ValueError("checkpoint must contain the new GATv2 model")
        self.batch_size = int(metadata["batch_size"])
        # Preserve inference minibatch boundaries while bounding graph-building
        # memory independently of particles × horizon.
        self.graph_chunk_size = max(self.batch_size, (256 // self.batch_size) * self.batch_size)
        self.cache_hard_batches=False
        self.hard_batch_cache=OrderedDict()
        self.hard_batch_cache_capacity=4096
        self.hard_batch_cache_hits=0
        self.hard_batch_cache_misses=0
        self.model = model_class("new", "gatv2")(
            2, 100, 26, nnx.Rngs(int(metadata["seed"])), self.batch_size
        )
        state = nnx.state(self.model)
        restored = serialization.from_bytes(nnx.to_pure_dict(state), checkpoint.read_bytes())
        nnx.replace_by_pure_dict(state, restored)
        nnx.update(self.model, state)
        self.model.eval()
        self.kind = kind
        calibration = prepare_records(read_records(calibration_data))
        probabilities, labels = predict_probabilities(
            self.model, calibration, self.batch_size, "new"
        )
        self.quantile = (
            fit_lac(probabilities, labels, 1 - coverage)
            if kind == "lac"
            else calibrate_underbound(probabilities.argmax(axis=1), labels, coverage)
        )

    def lower_bounds(
        self, positions: np.ndarray, radius: float, mode: str = "hard",
        sigma: float | None = None,
    ) -> np.ndarray:
        positions = np.asarray(positions)
        bounds = []
        for begin in range(0, len(positions), self.graph_chunk_size):
            records = communication_records(positions[begin:begin+self.graph_chunk_size], radius, mode, sigma)
            if mode=='hard' and getattr(self,'cache_hard_batches',False):
                for offset in range(0,len(records),self.batch_size):
                    batch=records[offset:offset+self.batch_size]
                    packed=[]
                    for record in batch:
                        n=len(record['features']);support=np.zeros((n,n),dtype=np.uint8)
                        support[record['edge_index'][0],record['edge_index'][1]]=1
                        packed.append(np.packbits(support).tobytes())
                    key=(n,len(batch),self.batch_size,self.kind,float(self.quantile),b''.join(packed))
                    if key in self.hard_batch_cache:
                        value=self.hard_batch_cache.pop(key);self.hard_batch_cache_hits+=1
                    else:
                        probabilities,_=predict_probabilities(self.model,batch,self.batch_size,'new')
                        value=(np.argmax(lac_prediction_sets(probabilities,self.quantile),axis=1)
                               if self.kind=='lac' else predict_underbounds(probabilities.argmax(axis=1),self.quantile,26))
                        self.hard_batch_cache_misses+=1
                    self.hard_batch_cache[key]=value
                    if len(self.hard_batch_cache)>self.hard_batch_cache_capacity:self.hard_batch_cache.popitem(last=False)
                    bounds.append(value.copy())
                continue
            probabilities, _ = predict_probabilities(self.model, records, self.batch_size, "new")
            if self.kind == "lac":
                bounds.append(np.argmax(lac_prediction_sets(probabilities, self.quantile), axis=1))
            else:
                bounds.append(predict_underbounds(probabilities.argmax(axis=1), self.quantile, 26))
        return np.concatenate(bounds) if bounds else np.empty(0, dtype=int)


def make_smooth_cp_value_and_grad(model, quantile, config: PlannerConfig):
    """Freeze the model and differentiate a continuous, non-certified surrogate.

    Expected class minus the one-sided calibration correction replaces integer
    argmax/ceil. Hard-graph CP bounds remain the final acceptance criterion.
    Map trajectories/waypoints sequentially to bound reverse-mode GPU memory.
    """
    if config.cp_kind != "one_sided" or not np.isfinite(quantile):
        raise ValueError("autodiff guidance requires one_sided CP with a finite quantile")
    graphdef, state = nnx.split(model)

    def expected_lower(positions):
        robots = positions.shape[0]
        source, target = np.nonzero(~np.eye(robots, dtype=bool))
        edges = jnp.asarray(np.stack((source, target)), dtype=jnp.int32)
        weights = smooth_adjacency(
            positions, config.communication_radius, config.connectivity_sigma
        )
        features = jnp.stack((weights.sum(axis=1), jnp.ones(robots)), axis=-1)
        frozen = nnx.merge(graphdef, state)
        frozen.batchsize = 1
        logits = frozen(features, edges, weights[source, target, None],
                        jnp.zeros(robots, dtype=jnp.int32))[0]
        probabilities = jax.nn.softmax(logits)
        return jnp.sum(probabilities * jnp.arange(logits.shape[0])) - quantile

    def single(trajectory):
        lower = jax.lax.map(jax.checkpoint(expected_lower), trajectory)
        shortfall = jnp.maximum(config.required_robustness - lower.min(), 0.0)
        return config.cp_weight * shortfall**2

    return jax.jit(lambda particles: jax.lax.map(jax.value_and_grad(single), particles))


def svgd_direction(particles: np.ndarray, log_density_gradient: np.ndarray) -> np.ndarray:
    """RBF Stein direction with the median bandwidth heuristic."""
    flat = np.asarray(particles, dtype=np.float64).reshape(len(particles), -1)
    gradient = np.asarray(log_density_gradient, dtype=np.float64).reshape(flat.shape)
    if len(flat) < 2:
        raise ValueError("SVGD needs at least two particles")
    # A full (particles, particles, control_dimensions) difference and its
    # square exceed 5 GB for 100 × 320 × 50 × 2 control populations. Compute
    # each upper-triangle row directly instead. Direct differences also avoid
    # cancellation for nearly identical particles with large coordinates.
    distance_squared = np.zeros((len(flat), len(flat)), dtype=np.float64)
    for index in range(len(flat)-1):
        difference = flat[index+1:] - flat[index]
        squared = np.sum(difference**2, axis=-1)
        distance_squared[index,index+1:] = squared
        distance_squared[index+1:,index] = squared
    upper = distance_squared[np.triu_indices(len(flat), k=1)]
    bandwidth = max(float(np.median(upper)) / np.log(len(flat) + 1), 1e-8)
    kernel = np.exp(-distance_squared / bandwidth)
    attraction = kernel @ gradient
    repulsion = 2.0 * (flat * kernel.sum(axis=1, keepdims=True) - kernel @ flat) / bandwidth
    return ((attraction + repulsion) / len(flat)).reshape(particles.shape)


def _smooth_cost_single(waypoints, start, goal, config: PlannerConfig):
    path = jnp.concatenate((start[None], waypoints), axis=0)
    motion = path[1:] - path[:-1]
    step_squared = jnp.sum(motion**2, axis=-1)
    # The derivative of norm(0) is undefined.  This epsilon keeps stationary
    # particles finite without perturbing the squared path-energy term.
    step_length = jnp.sqrt(step_squared + 1e-12)
    displacement = path[:, :, None, :] - path[:, None, :, :]
    distance = jnp.linalg.norm(displacement + 1e-8, axis=-1)
    off_diagonal = 1.0 - jnp.eye(start.shape[0])
    collision = jnp.maximum(config.minimum_separation - distance, 0.0) ** 2
    collision = (collision * off_diagonal).sum() / (start.shape[0] * (start.shape[0] - 1))
    # A differentiable necessary condition for r-robustness: degree >= r.
    # Preserve the sigmoid heuristic in hard mode as the original baseline.
    if config.connectivity_mode == "smooth":
        soft_edges = smooth_adjacency(
            path, config.communication_radius, config.connectivity_sigma
        )
    else:
        soft_edges = jax.nn.sigmoid(
            (config.communication_radius - distance) / config.smooth_temperature
        )
    soft_degree = (soft_edges * off_diagonal).sum(axis=-1)
    degree_shortfall = jnp.maximum(config.required_robustness - soft_degree[1:], 0.0)
    return (
        config.goal_weight * jnp.mean((waypoints[-1] - goal) ** 2)
        + config.path_weight * jnp.mean(step_squared)
        + config.collision_weight * collision
        + config.speed_weight * jnp.mean(jnp.maximum(step_length - config.maximum_step, 0.0) ** 2)
        + config.degree_weight * jnp.mean(degree_shortfall**2)
    )


def cp_cost(particles: np.ndarray, scorer: ConformalGraphScorer, config: PlannerConfig):
    p, steps, robots, _ = particles.shape
    # The checkpoint and calibration data contain binary radius graphs, and
    # exact r-robustness is defined on that hard support. Smooth connectivity
    # remains the differentiable planning surrogate, not the CP input format.
    bounds = scorer.lower_bounds(
        particles.reshape(p * steps, robots, 2), config.communication_radius,
        "hard", None,
    )
    minimum = bounds.reshape(p, steps).min(axis=1)
    shortfall = np.maximum(config.required_robustness - minimum, 0)
    return config.cp_weight * shortfall.astype(float) ** 2, minimum


def cp_spsa_gradient(
    particles: np.ndarray,
    scorer: ConformalGraphScorer,
    config: PlannerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate the gradient of a Gaussian-smoothed CP cost with 2-sided probes."""
    # Collect all +/- probes before graph construction and GNN inference.
    # cp_cost already flattens particles x waypoints; inference still uses
    # bounded graph minibatches rather than one GPU allocation for all probes.
    noise = rng.normal(size=(config.cp_directions, *particles.shape))
    probes = np.stack((particles + config.cp_smoothing * noise,
                       particles - config.cp_smoothing * noise), axis=1)
    values, _ = cp_cost(probes.reshape(-1, *particles.shape[1:]), scorer, config)
    values = values.reshape(config.cp_directions, 2, len(particles))
    slopes = (values[:, 0] - values[:, 1]) / (2 * config.cp_smoothing)
    return np.mean(slopes[..., None, None, None] * noise, axis=0)


def plan(
    start: np.ndarray,
    goal: np.ndarray,
    initial_particles: np.ndarray,
    scorer: ConformalGraphScorer,
    config: PlannerConfig,
    iterations: int,
    seed: int = 42,
    record_particles: bool = False,
):
    """Return the cheapest CP-passing candidate, or (None, report) if none pass."""
    if config.dynamics == "double_integrator":
        from planning.double_integrator_planner import plan_double_integrator
        return plan_double_integrator(start, goal, initial_particles, scorer, config,
                                      iterations, seed, record_particles)
    if config.dynamics == 'single_integrator':
        from planning.single_integrator_planner import plan_single_integrator
        return plan_single_integrator(start,goal,initial_particles,scorer,config,iterations,seed,record_particles)
    if config.dynamics != "waypoints":
        raise ValueError("dynamics must be waypoints, double_integrator or single_integrator")
    start, goal = np.asarray(start, np.float32), np.asarray(goal, np.float32)
    particles = np.asarray(initial_particles, np.float64).copy()
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        raise ValueError("start and goal must have shape (robots, 2)")
    if particles.ndim != 4 or particles.shape[2:] != start.shape:
        raise ValueError("particles must have shape (particles, steps, robots, 2)")
    if len(start) < 2 or len(particles) < 2 or particles.shape[1] < 1:
        raise ValueError("at least two robots, two particles, and one step are required")
    if config.fixed_goal:
        particles[:, -1] = goal
    if not 0 <= config.required_robustness < 26:
        raise ValueError("required_robustness must be between zero and 25")
    if config.maximum_speed <= 0 or config.sampling_time <= 0:
        raise ValueError("maximum_speed and sampling_time must be positive")
    if config.connectivity_mode not in ("hard", "smooth"):
        raise ValueError("connectivity_mode must be 'hard' or 'smooth'")
    if config.connectivity_sigma is not None and config.connectivity_sigma <= 0:
        raise ValueError("connectivity_sigma must be positive")
    if iterations < 1 or config.cp_directions < 1 or config.cp_smoothing <= 0:
        raise ValueError("iterations, cp_directions, and cp_smoothing must be positive")
    if scorer is None:
        raise ValueError("a graph scorer is required for the final CP filter")
    if config.cp_gradient not in ("spsa", "autodiff"):
        raise ValueError("cp_gradient must be 'spsa' or 'autodiff'")
    if config.cp_gradient == "autodiff" and config.connectivity_mode != "smooth":
        raise ValueError("autodiff guidance requires smooth connectivity")
    learned = None
    learned_gradient = None
    if config.cp_weight and config.cp_gradient == "autodiff":
        learned = make_smooth_cp_value_and_grad(scorer.model, scorer.quantile, config)
    smooth = jax.jit(jax.vmap(lambda w: _smooth_cost_single(w, jnp.asarray(start), jnp.asarray(goal), config)))
    smooth_gradient = jax.jit(jax.vmap(jax.grad(lambda w: _smooth_cost_single(w, jnp.asarray(start), jnp.asarray(goal), config))))
    rng = np.random.default_rng(seed)
    history = []
    particle_snapshots = []
    population_costs = []
    population_bounds = []
    candidates = []
    candidate_costs = []
    candidate_bounds = []

    def evaluated_population(current_particles):
        nonlocal learned_gradient
        total = np.array(smooth(jnp.asarray(current_particles)), dtype=np.float64)
        cp_values, lower = cp_cost(current_particles, scorer, config)
        if learned is not None:
            cp_values, learned_gradient = learned(jnp.asarray(current_particles, dtype=jnp.float32))
            cp_values = np.asarray(cp_values)
            learned_gradient = np.asarray(learned_gradient, dtype=np.float64)
        total += cp_values
        candidates.append(current_particles.copy())
        candidate_costs.append(total.copy())
        candidate_bounds.append(lower.copy())
        return total, lower

    # Preserve the best candidate seen. SVGD updates and particle repulsion are
    # not guaranteed to improve the discrete CP objective monotonically.
    total, lower = evaluated_population(particles)
    if record_particles:
        particle_snapshots.append(particles.copy())
        population_costs.append(total.copy())
        population_bounds.append(None if lower is None else lower.copy())
    best_index = int(np.argmin(total))
    best_waypoints = particles[best_index].copy()
    best_cost = float(total[best_index])
    best_bound = int(lower[best_index]) if lower is not None else None

    for _ in range(iterations):
        t0 = perf_counter()
        gradient = np.array(smooth_gradient(jnp.asarray(particles)), dtype=np.float64)
        if config.cp_weight:
            gradient += (learned_gradient if learned is not None else
                         cp_spsa_gradient(particles, scorer, config, rng))
        if config.fixed_goal:
            gradient[:, -1] = 0.0
        particles += config.svgd_step_size * svgd_direction(particles, -gradient)
        if config.fixed_goal:
            particles[:, -1] = goal
        total, lower = evaluated_population(particles)
        if record_particles:
            particle_snapshots.append(particles.copy())
            population_costs.append(total.copy())
            population_bounds.append(None if lower is None else lower.copy())
        iteration_index = int(np.argmin(total))
        iteration_cost = float(total[iteration_index])
        iteration_bound = int(lower[iteration_index]) if lower is not None else None
        if iteration_cost < best_cost:
            best_index = iteration_index
            best_waypoints = particles[iteration_index].copy()
            best_cost = iteration_cost
            best_bound = iteration_bound
        history.append({"best_cost": iteration_cost,
                        "best_lower_bound": iteration_bound,
                        "incumbent_cost": best_cost,
                        "incumbent_lower_bound": best_bound,
                        "seconds": perf_counter() - t0})
    # Final acceptance is independent of the optimization penalty weight.
    # Include earlier populations, since SVGD can move a passing particle away.
    costs = np.concatenate(candidate_costs)
    bounds = np.concatenate(candidate_bounds)
    cp_passing = np.flatnonzero(
        (bounds >= config.required_robustness) & np.isfinite(costs)
    )
    # Check ALL CP-passing candidates before ranking, including earlier ones.
    # An unsafe low-cost candidate must not hide a more expensive safe plan.
    passing=[]
    selected_separation=None
    rejected=0
    for selected in sorted(cp_passing, key=lambda index: costs[index]):
        population,index=divmod(int(selected),len(particles))
        if config.collision_filter:
            separation=separation_acceptance(start,candidates[population][index],config.minimum_separation,
                                             robot_radius=config.robot_radius)
            if not separation['zero_collision']:
                rejected+=1
                continue
            if not passing:selected_separation=separation
        passing.append(int(selected))
    passing=np.asarray(passing,dtype=int)
    selected_population = None
    if len(passing):
        selected = int(passing[np.argmin(costs[passing])])
        selected_population, best_index = divmod(selected, len(particles))
        best_waypoints = candidates[selected_population][best_index].copy()
        best_cost = float(costs[selected])
        best_bound = int(bounds[selected])
    else:
        best_waypoints = best_cost = best_bound = best_index = None
    report = {"index": best_index, "cost": best_cost,
              "minimum_lower_bound": best_bound, "history": history,
              "status": "feasible" if len(passing) else "no_feasible_trajectory",
              "selected_population": selected_population,
              "passing_candidates": int(len(passing)),
              "evaluated_candidates": int(len(costs))}
    report.update(collision_filter_enabled=config.collision_filter,
                  cp_passing_candidates=int(len(cp_passing)),collision_rejected_candidates=rejected,
                  separation_threshold_m=config.minimum_separation)
    if selected_separation is not None:
        report['separation_check']=selected_separation
    report["cp_gradient"] = config.cp_gradient
    report["optimization_cp_cost"] = (
        "continuous_expected_class_surrogate" if learned is not None else "hard_cp_shortfall"
    )
    if record_particles:
        report.update(
            particle_snapshots=np.asarray(particle_snapshots),
            population_costs=np.asarray(population_costs),
            population_bounds=(None if population_bounds[0] is None
                               else np.asarray(population_bounds)),
        )
    return best_waypoints, report
