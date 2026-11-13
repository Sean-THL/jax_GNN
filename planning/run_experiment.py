"""Compare four SVGD objectives from the same initial particles.

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
    MAX_PLANNING_ROBOTS,
    ConformalGraphScorer,
    PlannerConfig,
    _smooth_cost_single,
    communication_records,
    plan,
)
from planning.dynamics import initial_controls


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
        raise ValueError(
            f"r={minimum_degree} requires robust_seed_size between {minimum_degree+1} "
            f"and robot_count={count}; configured robust_seed_size={seed_size}. "
            "Update random_positions.robust_seed_size in the scenario JSON."
        )
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
        if not 2 <= robot_count <= pool_size <= MAX_PLANNING_ROBOTS:
            raise ValueError(f"robots must be between 2 and the configured pool size (at most {MAX_PLANNING_ROBOTS})")
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
        if "random_goals" in scenario:
            goals = scenario["random_goals"]
            goal_lower = np.asarray(goals["lower"], dtype=np.float32)
            goal_upper = np.asarray(goals["upper"], dtype=np.float32)
            if goal_lower.shape != (2,) or goal_upper.shape != (2,) or np.any(goal_lower >= goal_upper):
                raise ValueError("random_goals bounds must define a valid 2-D box")
            goal_pool = _sample_degree_constrained_pool(
                pool_size, goal_lower, goal_upper, minimum_separation, radius,
                target_degree, seed_size, np.random.default_rng(int(goals.get("seed", seed+1))),
            )
        else:
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
        ("gnn_cp_spsa", replace(config, cp_gradient="spsa")),
        ("gnn_cp_autodiff", replace(config, cp_gradient="autodiff", connectivity_mode="smooth")),
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
    if waypoints is None:
        return {
            "status": "no_feasible_trajectory", "waypoints": None,
            "meets_predicted_target": False, "wall_seconds": float(wall_seconds),
            "minimum_lower_bound": None, "minimum_exact_robustness": None,
            "minimum_hard_degree": None, "goal_mean_distance": None,
            "largest_step": None, "smallest_separation": None,
            "total_shared_objective": None,
        }
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
    parser.add_argument("--robots", type=int, choices=range(5, MAX_PLANNING_ROBOTS+1),
                        help="Use the first N robots from the generated pool")
    parser.add_argument("--steps", type=int,
                        help="Planning horizon (default: scenario horizon_steps, otherwise 10)")
    parser.add_argument("--sampling-time", "--dt", dest="sampling_time", type=float,
                        help="Sampling time dt in seconds (overrides the scenario)")
    parser.add_argument("--maximum-speed", type=float,
                        help="Maximum speed in distance units/second (overrides the scenario)")
    parser.add_argument("--maximum-acceleration", type=float,
                        help="Hard acceleration norm bound for double-integrator dynamics")
    parser.add_argument('--dynamics',choices=('waypoints','double_integrator','single_integrator'),help='Override the scenario dynamics')
    parser.add_argument("--collision-weight", type=float,
                        help="Override the soft pairwise collision penalty weight")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--noise", type=float,
                        help="Absolute waypoint noise (overrides --noise-fraction)")
    parser.add_argument("--noise-fraction", type=float, default=0.1,
                        help="Default noise as a fraction of maximum_speed * dt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+",
                        choices=("simple", "soft_degree", "gnn_cp_spsa", "gnn_cp_autodiff"),
                        default=["gnn_cp_autodiff"],
                        help="Run these planners (default: gnn_cp_autodiff)")
    parser.add_argument("--connectivity-mode", choices=("hard", "smooth"),
                        help="Override the scenario connectivity mode (default: smooth)")
    parser.add_argument("--connectivity-sigma", type=float,
                        help="Smooth sigma (default: R^4 / ln(2))")
    parser.add_argument("--output", type=Path, help="Optional JSON file for both trajectories and metrics")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text())
    if args.steps is None:
        args.steps = int(scenario.get("horizon_steps", 10))
    try:
        config, adversary_bound = config_from_scenario(scenario["config"])
        start, goal = scenario_positions(
            scenario, args.robots, config.required_robustness, config.communication_radius
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        parser.error("scenario start and goal must have matching (robots, 2) arrays")
    if args.dynamics is not None:
        config=replace(config,dynamics=args.dynamics)
    if args.sampling_time is not None:
        config = replace(config, sampling_time=args.sampling_time)
    if args.maximum_speed is not None:
        config = replace(config, maximum_speed=args.maximum_speed)
    if args.maximum_acceleration is not None:
        config = replace(config, maximum_acceleration=args.maximum_acceleration)
    if args.collision_weight is not None:
        if not np.isfinite(args.collision_weight) or args.collision_weight < 0:
            parser.error("collision-weight must be finite and nonnegative")
        config = replace(config, collision_weight=args.collision_weight)
    if args.connectivity_mode is not None:
        config = replace(config, connectivity_mode=args.connectivity_mode)
    if args.connectivity_sigma is not None:
        config = replace(config, connectivity_sigma=args.connectivity_sigma)
    if config.degree_weight <= 0 or config.cp_weight <= 0:
        parser.error("comparison requires positive degree_weight and cp_weight")
    if config.connectivity_sigma is not None and config.connectivity_sigma <= 0:
        parser.error("connectivity-sigma must be positive")
    if config.sampling_time <= 0 or config.maximum_speed <= 0:
        parser.error("sampling-time and maximum-speed must be positive")
    if config.dynamics == "double_integrator" and config.maximum_acceleration <= 0:
        parser.error("maximum-acceleration must be positive")
    if args.noise_fraction < 0:
        parser.error("noise-fraction must be nonnegative")
    trajectory_noise = (
        args.noise if args.noise is not None
        else args.noise_fraction * (config.maximum_acceleration if config.dynamics == "double_integrator" else config.maximum_speed if config.dynamics=='single_integrator' else config.maximum_step)
    )
    if trajectory_noise < 0:
        parser.error("noise must be nonnegative")
    if config.dynamics in ("double_integrator","single_integrator"):
        particles = initial_controls(start, goal, args.particles, args.steps, config, trajectory_noise, args.seed)
        distance = np.linalg.norm(goal-start, axis=-1).max()
        if config.dynamics=='double_integrator' and np.allclose(config.initial_velocity, 0) and distance > 0.5*config.maximum_acceleration*(args.steps*config.sampling_time)**2:
            parser.error("horizon too short to reach these goals from rest under the acceleration bound")
        # SI arrival is soft: short horizons remain runnable and report goal error.
    else:
        particles = initial_trajectories(start, goal, args.particles, args.steps, trajectory_noise, args.seed)
    scorer = ConformalGraphScorer(
        args.checkpoint, args.calibration, config.cp_coverage, config.cp_kind
    )

    runs = [(name, settings) for name, settings in comparison_configs(config)
            if args.methods is None or name in args.methods]
    if any(name == "gnn_cp_autodiff" for name, _ in runs) and (
        scorer.kind != "one_sided" or not np.isfinite(scorer.quantile)
    ):
        parser.error("gnn_cp_autodiff requires one_sided CP with a finite quantile; "
                     "use --methods to select the other planners")
    results = {}
    for name, run_config in runs:
        began = perf_counter()
        trajectory, planner_report = plan(
            start, goal, particles,
            scorer,
            run_config, args.iterations, args.seed,
        )
        elapsed = perf_counter() - began
        results[name] = trajectory_summary(
            trajectory, start, goal, scorer, config, elapsed
        )
        results[name]["planner_report"] = planner_report
        results[name]["planner_config"] = asdict(run_config)
        results[name]["meets_separation_target"] = (
            planner_report.get("separation_check",{}).get("zero_collision")
            if trajectory is not None else False
        )
        results[name]["cp_scale_status"] = (
            "experimental_above_50_training_calibration_nodes" if len(start) > 50
            else "within_original_node_count_range"
        )
        if run_config.dynamics == "double_integrator" and trajectory is not None:
            results[name]["dynamics_metrics"] = {
                key: planner_report[key] for key in (
                    "maximum_acceleration_norm", "maximum_velocity_norm", "terminal_velocity_mean_norm"
                )
            }
        if run_config.dynamics=='single_integrator' and trajectory is not None:
            results[name]['dynamics_metrics']={key:planner_report[key] for key in ('maximum_velocity_norm','terminal_velocity_mean_norm')}

    print(f"Scenario: {scenario.get('description', args.scenario.name)}")
    print(f"Robots: {len(start)}; particles: {args.particles}; future steps: {args.steps}; iterations: {args.iterations}")
    print(f"CP method: {config.cp_kind}; target coverage: {config.cp_coverage:.0%}; "
          f"F: {adversary_bound}; required robustness (2F+1): {config.required_robustness}")
    print(f"Sampling time dt: {config.sampling_time:g}; maximum speed: {config.maximum_speed:g}; "
          f"maximum step = speed * dt: {config.maximum_step:g}")
    if config.dynamics == "double_integrator":
        print(f"Dynamics: double integrator; hard acceleration norm <= {config.maximum_acceleration:g} m/s^2; "
              f"horizon {args.steps * config.sampling_time:g} s; speed is a soft cost")
    elif config.dynamics=='single_integrator':
        print(f'Dynamics: single integrator; hard velocity norm <= {config.maximum_speed:g} m/s; horizon {args.steps*config.sampling_time:g} s; position is the state')
    if config.robot_radius is not None:
        print(f"Robot radius: {config.robot_radius:g} m; body diameter: {2*config.robot_radius:g} m; "
              f"separation target: {config.minimum_separation:g} m; surface gap target: {config.minimum_separation-2*config.robot_radius:g} m")
    print(f"Initial particle noise: {trajectory_noise:g} "
          f"({'acceleration units' if config.dynamics == 'double_integrator' else 'velocity units' if config.dynamics=='single_integrator' else 'position units'})")
    if config.required_robustness > (len(start) + 1) // 2:
        print(f"WARNING: r={config.required_robustness} is infeasible for {len(start)} robots; "
              "this run will report Success=NO.")
    sigma_text = "R^4 / ln(2)" if config.connectivity_sigma is None else str(config.connectivity_sigma)
    print(f"Connectivity: {config.connectivity_mode}; sigma: {sigma_text if config.connectivity_mode == 'smooth' else 'n/a'}")
    print("CP certification graph: hard radius support (matches training and exact r)")
    if len(start) > 50:
        print("CP scale: experimental; 100-robot support does not extend the checkpoint's <=50-node calibration.")
    print("Method        CP bound   Min degree   Success   Exact r   Goal error   Max step   Min separation   Shared cost   Seconds")
    for name, _ in runs:
        value = results[name]
        if value["waypoints"] is None:
            report=value['planner_report']
            print(f"{name:15} no feasible trajectory (Success=NO; "
                  f"CP-pass={report.get('cp_passing_candidates',0)}, "
                  f"collision-rejected={report.get('collision_rejected_candidates',0)})")
            continue
        exact = "n/a" if value["minimum_exact_robustness"] is None else str(value["minimum_exact_robustness"])
        success = "YES" if value["meets_predicted_target"] else "NO"
        print(f"{name:15} {value['minimum_lower_bound']:8d} "
              f"{value['minimum_hard_degree']:12d} {success:>9} {exact:>9} "
              f"{value['goal_mean_distance']:12.3f} {value['largest_step']:10.3f} "
              f"{value['smallest_separation']:16.3f} {value['total_shared_objective']:13.3f} "
              f"{value['wall_seconds']:9.3f}")
    print("Exact r uses exhaustive subset checks only for scenarios with at most 10 robots.")
    print("Seconds measures optimization and final CP filtering; scorer setup and evaluation are excluded.")
    print('Terminal goals are soft; SI velocity and DI acceleration are hard bounded. DI speed is a soft cost.')
    print(f"Final collision filter: {'ON' if config.collision_filter else 'OFF'}; "
          f"requires center separation >= {config.minimum_separation:g} m throughout each motion interval.")
    print("The CP bound is a prediction, not a guarantee for the trajectory selected by search.")

    if args.output:
        payload = {
            "scenario": str(args.scenario),
            "checkpoint": str(args.checkpoint),
            "calibration": str(args.calibration),
            "parameters": {"robots": len(start), "particles": args.particles, "steps": args.steps,
                           "iterations": args.iterations, "noise": trajectory_noise,
                           "noise_fraction": trajectory_noise / (config.maximum_acceleration if config.dynamics == "double_integrator" else config.maximum_speed if config.dynamics=='single_integrator' else config.maximum_step),
                           "noise_units": "m/s^2" if config.dynamics == "double_integrator" else "m/s" if config.dynamics=='single_integrator' else "position units",
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
