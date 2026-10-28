"""Compare three SVGD objectives from the same initial trajectories.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python -m planning \
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


def config_from_scenario(config_values):
    """Build planner settings, deriving the robustness target from F."""
    values = dict(config_values)
    if "adversary_bound" not in values:
        raise ValueError("scenario config must define adversary_bound (F)")
    adversary_bound = values.pop("adversary_bound")
    if isinstance(adversary_bound, bool) or not isinstance(adversary_bound, int):
        raise ValueError("adversary_bound (F) must be a nonnegative integer")
    if adversary_bound < 0:
        raise ValueError("adversary_bound (F) must be a nonnegative integer")
    if "required_robustness" in values:
        raise ValueError(
            "set adversary_bound (F), not required_robustness; "
            "required_robustness is derived as 2 * F + 1"
        )
    required_robustness = 2 * adversary_bound + 1
    return PlannerConfig(required_robustness=required_robustness, **values), adversary_bound


def _sample_degree_constrained_pool(
    count, lower, upper, minimum_separation, radius, minimum_degree, seed_size, rng
):
    """Grow a pool whose feasible prefixes have minimum degree at least r."""
    if seed_size < minimum_degree + 1 or seed_size > count:
        raise ValueError("robust_seed_size must be between r + 1 and robot_count")
    # Build required edges inside 80% of the hard radius. The unused 20%
    # provides clearance for the small waypoint perturbations added later.
    protected_radius = 0.8 * radius
    seed_radius = 0.5 * protected_radius
    center_lower, center_upper = lower + seed_radius, upper - seed_radius
    if np.any(center_lower >= center_upper):
        raise ValueError("random position bounds are too small for the communication radius")

    for _ in range(100):
        center = rng.uniform(center_lower, center_upper)
        points = []
        for _ in range(seed_size):
            for _ in range(10_000):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                distance = seed_radius * np.sqrt(rng.uniform())
                candidate = center + distance * np.array([np.cos(angle), np.sin(angle)])
                if not points or np.min(np.linalg.norm(np.asarray(points) - candidate, axis=1)) >= minimum_separation:
                    points.append(candidate)
                    break
            else:
                break
        if len(points) != seed_size:
            continue

        while len(points) < count:
            for _ in range(20_000):
                candidate = rng.uniform(lower, upper)
                distances = np.linalg.norm(np.asarray(points) - candidate, axis=1)
                if np.min(distances) < minimum_separation:
                    continue
                if np.count_nonzero(distances <= protected_radius) >= minimum_degree:
                    points.append(candidate)
                    break
            else:
                break
        if len(points) == count:
            return np.asarray(points, dtype=np.float32)
    raise ValueError("could not generate the degree-constrained robot pool")


def scenario_positions(
    scenario, requested_robots=None, required_robustness=None, communication_radius=None
):
    """Load explicit positions or reproducibly sample a configured robot pool."""
    if "random_positions" in scenario:
        spec = scenario["random_positions"]
        pool_size = int(spec["robot_count"])
        robot_count = pool_size if requested_robots is None else requested_robots
        if not 2 <= robot_count <= pool_size <= 50:
            raise ValueError("robots must be between 2 and the configured pool size (at most 50)")
        lower = np.asarray(spec.get("lower", [0.0, 0.0]), dtype=np.float32)
        upper = np.asarray(spec.get("upper", [1.0, 1.0]), dtype=np.float32)
        if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
            raise ValueError("random_positions lower and upper must define a valid 2-D box")
        minimum_separation = float(scenario["config"]["minimum_separation"])
        seed = int(spec.get("seed", 42))
        target_degree = (
            int(required_robustness)
            if required_robustness is not None
            else int(spec.get("minimum_degree", 0))
        )
        radius = (
            float(communication_radius)
            if communication_radius is not None
            else float(scenario["config"]["communication_radius"])
        )
        seed_size = int(spec.get("robust_seed_size", max(target_degree + 1, 2)))
        start_pool = _sample_degree_constrained_pool(
            pool_size, lower, upper, minimum_separation, radius,
            target_degree, seed_size, np.random.default_rng(seed),
        )
        translation = np.asarray(spec.get("goal_translation", [0.0, 0.0]), dtype=np.float32)
        if translation.shape != (2,):
            raise ValueError("goal_translation must be a 2-D vector")
        goal_pool = start_pool + translation
        return start_pool[:robot_count], goal_pool[:robot_count]

    start = np.asarray(scenario["start"], dtype=np.float32)
    goal = np.asarray(scenario["goal"], dtype=np.float32)
    if requested_robots is not None:
        if requested_robots > len(start):
            raise ValueError("--robots exceeds the number of explicit scenario positions")
        start, goal = start[:requested_robots], goal[:requested_robots]
    return start, goal


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
    # Exact r-robustness remains defined on the hard support graph.
    exact_records = communication_records(waypoints, config.communication_radius)
    # Certify the same binary support graph used by exact r-robustness and by
    # the checkpoint's training/calibration distribution. Smooth adjacency is
    # used only inside the differentiable planning objective.
    lower_bounds = scorer.lower_bounds(
        waypoints, config.communication_radius, "hard", None,
    )
    actual = [exact_robustness(record) for record in exact_records]
    minimum_hard_degree = min(
        int(np.min(np.asarray(record["features"])[:, 0])) for record in exact_records
    )
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
    target_is_feasible = config.required_robustness <= (robot_count + 1) // 2
    cp_penalty = config.cp_weight * max(0, config.required_robustness - minimum_bound) ** 2
    return {
        "lower_bounds_by_step": [int(x) for x in lower_bounds],
        "minimum_lower_bound": minimum_bound,
        "exact_robustness_by_step": actual,
        "minimum_exact_robustness": None if any(x is None for x in actual) else min(actual),
        "minimum_hard_degree": minimum_hard_degree,
        "target_is_theoretically_feasible": target_is_feasible,
        "meets_predicted_target": bool(
            target_is_feasible
            and minimum_bound >= config.required_robustness
            and minimum_hard_degree >= config.required_robustness
        ),
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
    parser.add_argument("--robots", type=int, choices=range(5, 51),
                        help="Use the first N robots from the generated pool")
    parser.add_argument("--steps", type=int, default=10,
                        help="Planning horizon in discrete sampling steps (default: 10)")
    parser.add_argument("--sampling-time", "--dt", dest="sampling_time", type=float,
                        help="Sampling time dt in seconds (overrides the scenario)")
    parser.add_argument("--maximum-speed", type=float,
                        help="Maximum speed in distance units/second (overrides the scenario)")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--noise", type=float,
                        help="Absolute waypoint noise (overrides --noise-fraction)")
    parser.add_argument("--noise-fraction", type=float, default=0.1,
                        help="Default noise as a fraction of maximum_speed * dt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connectivity-mode", choices=("hard", "smooth"),
                        help="Override the scenario connectivity mode (default: hard)")
    parser.add_argument("--connectivity-sigma", type=float,
                        help="Smooth sigma (default: R^4 / ln(2))")
    parser.add_argument("--output", type=Path, help="Optional JSON file for both trajectories and metrics")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text())
    try:
        config, adversary_bound = config_from_scenario(scenario["config"])
        start, goal = scenario_positions(
            scenario, args.robots, config.required_robustness, config.communication_radius
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        parser.error("scenario start and goal must have matching (robots, 2) arrays")
    if args.sampling_time is not None:
        config = replace(config, sampling_time=args.sampling_time)
    if args.maximum_speed is not None:
        config = replace(config, maximum_speed=args.maximum_speed)
    if args.connectivity_mode is not None:
        config = replace(config, connectivity_mode=args.connectivity_mode)
    if args.connectivity_sigma is not None:
        config = replace(config, connectivity_sigma=args.connectivity_sigma)
    if config.degree_weight <= 0 or config.cp_weight <= 0:
        parser.error("three-way comparison requires positive degree_weight and cp_weight")
    if config.connectivity_sigma is not None and config.connectivity_sigma <= 0:
        parser.error("connectivity-sigma must be positive")
    if config.sampling_time <= 0 or config.maximum_speed <= 0:
        parser.error("sampling-time and maximum-speed must be positive")
    if args.noise_fraction < 0:
        parser.error("noise-fraction must be nonnegative")
    trajectory_noise = (
        args.noise if args.noise is not None
        else args.noise_fraction * config.maximum_step
    )
    if trajectory_noise < 0:
        parser.error("noise must be nonnegative")
    particles = initial_trajectories(
        start, goal, args.particles, args.steps, trajectory_noise, args.seed
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
    print(f"CP method: {config.cp_kind}; target coverage: {config.cp_coverage:.0%}; "
          f"F: {adversary_bound}; required robustness (2F+1): {config.required_robustness}")
    print(f"Sampling time dt: {config.sampling_time:g}; maximum speed: {config.maximum_speed:g}; "
          f"maximum step = speed * dt: {config.maximum_step:g}")
    print(f"Initial trajectory noise: {trajectory_noise:g} "
          f"({trajectory_noise / config.maximum_step:.0%} of maximum step)")
    if config.required_robustness > (len(start) + 1) // 2:
        print(f"WARNING: r={config.required_robustness} is infeasible for {len(start)} robots; "
              "this run will report Success=NO.")
    sigma_text = "R^4 / ln(2)" if config.connectivity_sigma is None else str(config.connectivity_sigma)
    print(f"Connectivity: {config.connectivity_mode}; sigma: {sigma_text if config.connectivity_mode == 'smooth' else 'n/a'}")
    print("CP certification graph: hard radius support (matches training and exact r)")
    print("Method        CP bound   Min degree   Success   Exact r   Goal error   Max step   Min separation   Shared cost   Seconds")
    for name, _ in comparison_configs(config):
        value = results[name]
        exact = "n/a" if value["minimum_exact_robustness"] is None else str(value["minimum_exact_robustness"])
        success = "YES" if value["meets_predicted_target"] else "NO"
        print(f"{name:13} {value['minimum_lower_bound']:8d} "
              f"{value['minimum_hard_degree']:12d} {success:>9} {exact:>9} "
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
            "parameters": {"robots": len(start), "particles": args.particles, "steps": args.steps,
                           "iterations": args.iterations, "noise": trajectory_noise,
                           "noise_fraction": trajectory_noise / config.maximum_step,
                           "seed": args.seed},
            "config": {"adversary_bound": adversary_bound,
                       "required_robustness": config.required_robustness,
                       **asdict(config)},
            "start": start.tolist(),
            "goal": goal.tolist(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved trajectories and metrics: {args.output}")


if __name__ == "__main__":
    main()
