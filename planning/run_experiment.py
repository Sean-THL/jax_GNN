"""Compare three SVGD objectives from the same initial trajectories.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python run_svgd_experiment.py \
        --scenario planning/examples/demo_scenario.json --output /tmp/svgd_results.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from planning.svgd_cp_planner import (
    ConformalGraphScorer,
    PlannerConfig,
    _smooth_cost_single,
    communication_records,
    plan,
)


PROJECT = Path(__file__).resolve().parent.parent


def initial_trajectories(start, goal, count, steps, noise, seed):
    """Make diverse particles around a straight-line joint trajectory."""
    if count < 2 or steps < 1 or noise < 0:
        raise ValueError("particles >= 2, steps >= 1, and noise >= 0 are required")
    fractions = np.linspace(1 / steps, 1, steps, dtype=np.float64)
    baseline = start[None] + fractions[:, None, None] * (goal - start)[None]
    perturbations = np.random.default_rng(seed).normal(
        0.0, noise, size=(count, steps, *start.shape)
    )
    return baseline[None] + perturbations


def comparison_configs(config):
    """Keep starts, particles, and all other weights fixed across the ablation."""
    return (
        ("simple", replace(config, degree_weight=0.0, cp_weight=0.0)),
        ("soft_degree", replace(config, cp_weight=0.0)),
        ("gnn_cp", config),
    )


def exact_robustness(record):
    """Exhaustive r-robustness for a small directed graph (at most 10 nodes).

    A subset is r-reachable if one of its nodes has at least r incoming
    neighbors outside the subset. The graph's robustness is the minimum,
    across disjoint nonempty subset pairs, of their larger reachability.
    """
    n = len(record["features"])
    if not 2 <= n <= 10:
        return None
    full = (1 << n) - 1
    incoming = [0] * n
    for source, target in record["edge_index"].T:
        incoming[int(target)] |= 1 << int(source)
    reachability = [0] * (1 << n)
    for subset in range(1, full):
        outside = full ^ subset
        remaining = subset
        while remaining:
            bit = remaining & -remaining
            node = bit.bit_length() - 1
            reachability[subset] = max(
                reachability[subset], (incoming[node] & outside).bit_count()
            )
            remaining ^= bit
    robustness = n
    for first in range(1, full):
        remaining = full ^ first
        second = remaining
        while second:
            if first < second:
                robustness = min(
                    robustness,
                    max(reachability[first], reachability[second]),
                )
                if robustness == 0:
                    return 0
            second = (second - 1) & remaining
    return robustness


def trajectory_summary(waypoints, start, goal, scorer, config, wall_seconds):
    records = communication_records(waypoints, config.communication_radius)
    lower_bounds = scorer.lower_bounds(waypoints, config.communication_radius)
    actual = [exact_robustness(record) for record in records]
    path = np.concatenate((start[None], waypoints), axis=0)
    movement = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    differences = path[:, :, None, :] - path[:, None, :, :]
    separations = np.linalg.norm(differences, axis=-1)
    robot_count = len(start)
    off_diagonal = separations[:, ~np.eye(robot_count, dtype=bool)]
    smooth_cost = float(_smooth_cost_single(
        jnp.asarray(waypoints), jnp.asarray(start), jnp.asarray(goal), config
    ))
    minimum_bound = int(np.min(lower_bounds))
    cp_penalty = config.cp_weight * max(0, config.required_robustness - minimum_bound) ** 2
    return {
        "lower_bounds_by_step": [int(x) for x in lower_bounds],
        "minimum_lower_bound": minimum_bound,
        "exact_robustness_by_step": actual,
        "minimum_exact_robustness": None if any(x is None for x in actual) else min(actual),
        "meets_predicted_target": bool(minimum_bound >= config.required_robustness),
        "goal_mean_distance": float(np.linalg.norm(waypoints[-1] - goal, axis=-1).mean()),
        "largest_step": float(movement.max()),
        "smallest_separation": float(off_diagonal.min()),
        "smooth_cost": smooth_cost,
        "cp_penalty_under_shared_objective": float(cp_penalty),
        "total_shared_objective": float(smooth_cost + cp_penalty),
        "wall_seconds": float(wall_seconds),
        "waypoints": np.asarray(waypoints).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=PROJECT / "planning/examples/demo_scenario.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "checkpoints/full_run_xK0R3I/new_gatv2.msgpack")
    parser.add_argument("--calibration", type=Path, default=PROJECT / "calib_data.json")
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON file for both trajectories and metrics")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text())
    start = np.asarray(scenario["start"], dtype=np.float32)
    goal = np.asarray(scenario["goal"], dtype=np.float32)
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        parser.error("scenario start and goal must have matching (robots, 2) arrays")
    config = PlannerConfig(**scenario["config"])
    if config.degree_weight <= 0 or config.cp_weight <= 0:
        parser.error("three-way comparison requires positive degree_weight and cp_weight")
    particles = initial_trajectories(
        start, goal, args.particles, args.steps, args.noise, args.seed
    )
    scorer = ConformalGraphScorer(
        args.checkpoint, args.calibration, config.cp_coverage, config.cp_kind
    )

    results = {}
    for name, run_config in comparison_configs(config):
        began = perf_counter()
        trajectory, planner_report = plan(
            start, goal, particles,
            scorer if run_config.cp_weight else None,
            run_config, args.iterations, args.seed,
        )
        elapsed = perf_counter() - began
        results[name] = trajectory_summary(
            trajectory, start, goal, scorer, config, elapsed
        )
        results[name]["planner_report"] = planner_report

    print(f"Scenario: {scenario.get('description', args.scenario.name)}")
    print(f"Robots: {len(start)}; particles: {args.particles}; future steps: {args.steps}; iterations: {args.iterations}")
    print(f"CP method: {config.cp_kind}; target coverage: {config.cp_coverage:.0%}; required robustness: {config.required_robustness}")
    print("Method        CP bound   Exact r   Goal error   Max step   Min separation   Shared cost   Seconds")
    for name, _ in comparison_configs(config):
        value = results[name]
        exact = "n/a" if value["minimum_exact_robustness"] is None else str(value["minimum_exact_robustness"])
        print(f"{name:13} {value['minimum_lower_bound']:8d} {exact:>9} "
              f"{value['goal_mean_distance']:12.3f} {value['largest_step']:10.3f} "
              f"{value['smallest_separation']:16.3f} {value['total_shared_objective']:13.3f} "
              f"{value['wall_seconds']:9.3f}")
    print("Exact r uses exhaustive subset checks only for scenarios with at most 10 robots.")
    print("Seconds measures SVGD optimization only; scorer setup and evaluation are excluded.")
    print("Maximum step and minimum separation are soft penalties; inspect them against the scenario limits.")
    print("The CP bound is a prediction, not a guarantee for the trajectory selected by search.")

    if args.output:
        payload = {
            "scenario": str(args.scenario),
            "checkpoint": str(args.checkpoint),
            "calibration": str(args.calibration),
            "parameters": {"particles": args.particles, "steps": args.steps,
                           "iterations": args.iterations, "noise": args.noise, "seed": args.seed},
            "config": asdict(config),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved trajectories and metrics: {args.output}")


if __name__ == "__main__":
    main()
