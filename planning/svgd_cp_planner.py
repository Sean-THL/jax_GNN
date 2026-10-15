"""Experimental SVGD planner using a GATv2 conformal robustness cost.

Particles are short 2-D joint robot trajectories with one waypoint per control
step. This is a prototype: the communication radius and simple point-mass
motion model must be replaced with the robot system's actual definitions.
"""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class PlannerConfig:
    communication_radius: float
    minimum_separation: float
    maximum_step: float
    required_robustness: int
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


def communication_records(positions: np.ndarray, radius: float) -> list[dict]:
    """Create the same degree/edge representation used by the trained GNN."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape (graphs, robots, 2)")
    n = positions.shape[1]
    if not 1 <= n <= 50 or radius <= 0:
        raise ValueError("robots must be 1..50 and radius must be positive")
    records = []
    for graph in positions:
        distance_squared = np.sum((graph[:, None] - graph[None, :]) ** 2, axis=-1)
        adjacent = (distance_squared <= radius**2) & ~np.eye(n, dtype=bool)
        source, target = np.nonzero(adjacent)
        degree = np.bincount(source, minlength=n).astype(np.float32)
        features = np.column_stack((degree, np.ones(n, dtype=np.float32)))
        records.append({
            "features": features,
            "edge_index": np.stack((source, target)).astype(np.int32),
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

    def lower_bounds(self, positions: np.ndarray, radius: float) -> np.ndarray:
        records = communication_records(positions, radius)
        probabilities, _ = predict_probabilities(
            self.model, records, self.batch_size, "new"
        )
        if self.kind == "lac":
            return np.argmax(lac_prediction_sets(probabilities, self.quantile), axis=1)
        return predict_underbounds(probabilities.argmax(axis=1), self.quantile, 26)


def svgd_direction(particles: np.ndarray, log_density_gradient: np.ndarray) -> np.ndarray:
    """RBF Stein direction with the median bandwidth heuristic."""
    flat = np.asarray(particles, dtype=np.float64).reshape(len(particles), -1)
    gradient = np.asarray(log_density_gradient, dtype=np.float64).reshape(flat.shape)
    if len(flat) < 2:
        raise ValueError("SVGD needs at least two particles")
    difference = flat[:, None, :] - flat[None, :, :]
    distance_squared = np.sum(difference**2, axis=-1)
    upper = distance_squared[np.triu_indices(len(flat), k=1)]
    bandwidth = max(float(np.median(upper)) / np.log(len(flat) + 1), 1e-8)
    kernel = np.exp(-distance_squared / bandwidth)
    attraction = kernel @ gradient
    repulsion = 2.0 * (flat * kernel.sum(axis=1, keepdims=True) - kernel @ flat) / bandwidth
    return ((attraction + repulsion) / len(flat)).reshape(particles.shape)


def _smooth_cost_single(waypoints, start, goal, config: PlannerConfig):
    path = jnp.concatenate((start[None], waypoints), axis=0)
    motion = path[1:] - path[:-1]
    step_length = jnp.linalg.norm(motion, axis=-1)
    displacement = path[:, :, None, :] - path[:, None, :, :]
    distance = jnp.linalg.norm(displacement + 1e-8, axis=-1)
    off_diagonal = 1.0 - jnp.eye(start.shape[0])
    collision = jnp.maximum(config.minimum_separation - distance, 0.0) ** 2
    collision = (collision * off_diagonal).sum() / (start.shape[0] * (start.shape[0] - 1))
    # A differentiable necessary condition for r-robustness: degree >= r.
    soft_edges = jax.nn.sigmoid((config.communication_radius - distance) / config.smooth_temperature)
    soft_degree = (soft_edges * off_diagonal).sum(axis=-1)
    degree_shortfall = jnp.maximum(config.required_robustness - soft_degree[1:], 0.0)
    return (
        config.goal_weight * jnp.mean((waypoints[-1] - goal) ** 2)
        + config.path_weight * jnp.mean(step_length**2)
        + config.collision_weight * collision
        + config.speed_weight * jnp.mean(jnp.maximum(step_length - config.maximum_step, 0.0) ** 2)
        + config.degree_weight * jnp.mean(degree_shortfall**2)
    )


def cp_cost(particles: np.ndarray, scorer: ConformalGraphScorer, config: PlannerConfig):
    p, steps, robots, _ = particles.shape
    bounds = scorer.lower_bounds(particles.reshape(p * steps, robots, 2), config.communication_radius)
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
    gradient = np.zeros_like(particles, dtype=np.float64)
    for _ in range(config.cp_directions):
        noise = rng.normal(size=particles.shape)
        positive, _ = cp_cost(particles + config.cp_smoothing * noise, scorer, config)
        negative, _ = cp_cost(particles - config.cp_smoothing * noise, scorer, config)
        gradient += ((positive - negative) / (2 * config.cp_smoothing))[:, None, None, None] * noise
    return gradient / config.cp_directions


def plan(
    start: np.ndarray,
    goal: np.ndarray,
    initial_particles: np.ndarray,
    scorer: ConformalGraphScorer | None,
    config: PlannerConfig,
    iterations: int,
    seed: int = 42,
):
    """Optimize joint trajectories and return the lowest-cost candidate."""
    start, goal = np.asarray(start, np.float32), np.asarray(goal, np.float32)
    particles = np.asarray(initial_particles, np.float64).copy()
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        raise ValueError("start and goal must have shape (robots, 2)")
    if particles.ndim != 4 or particles.shape[2:] != start.shape:
        raise ValueError("particles must have shape (particles, steps, robots, 2)")
    if len(start) < 2 or len(particles) < 2 or particles.shape[1] < 1:
        raise ValueError("at least two robots, two particles, and one step are required")
    if not 0 <= config.required_robustness < len(start):
        raise ValueError("required_robustness must be between zero and robots - 1")
    if iterations < 1 or config.cp_directions < 1 or config.cp_smoothing <= 0:
        raise ValueError("iterations, cp_directions, and cp_smoothing must be positive")
    if config.cp_weight and scorer is None:
        raise ValueError("a graph scorer is required when cp_weight is nonzero")
    smooth = jax.jit(jax.vmap(lambda w: _smooth_cost_single(w, jnp.asarray(start), jnp.asarray(goal), config)))
    smooth_gradient = jax.jit(jax.vmap(jax.grad(lambda w: _smooth_cost_single(w, jnp.asarray(start), jnp.asarray(goal), config))))
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(iterations):
        t0 = perf_counter()
        gradient = np.array(smooth_gradient(jnp.asarray(particles)), dtype=np.float64)
        if config.cp_weight:
            gradient += cp_spsa_gradient(particles, scorer, config, rng)
        particles += config.svgd_step_size * svgd_direction(particles, -gradient)
        smooth_values = np.asarray(smooth(jnp.asarray(particles)))
        if config.cp_weight:
            cost, lower = cp_cost(particles, scorer, config)
            total = cost + smooth_values
            best_bound = int(lower[np.argmin(total)])
        else:
            total = smooth_values
            best_bound = None
        history.append({"best_cost": float(np.min(total)),
                        "best_lower_bound": best_bound,
                        "seconds": perf_counter() - t0})
    total = np.array(smooth(jnp.asarray(particles)), dtype=np.float64)
    if config.cp_weight:
        cp_values, lower = cp_cost(particles, scorer, config)
        total += cp_values
    best = int(np.argmin(total))
    if config.cp_weight:
        best_bound = int(lower[best])
    else:
        # Baseline optimization must not query the GNN. The runner evaluates
        # every selected trajectory with the same scorer after timing the plan.
        best_bound = None
    return particles[best], {"index": best, "cost": float(total[best]),
                             "minimum_lower_bound": best_bound, "history": history}
