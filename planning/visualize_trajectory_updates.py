"""Record real SVGD updates and animate optimization separately from motion.

Run from jax_GNN with python -m planning.visualize_trajectory_updates.
Use --load to re-render an existing output directory without rerunning GNNs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-planner-matplotlib")
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from planning.run_experiment import (
    PROJECT, comparison_configs, config_from_scenario, initial_trajectories,
    scenario_positions, trajectory_summary,
)
from planning.svgd_cp_planner import ConformalGraphScorer, plan
from planning.dynamics import initial_controls

TITLES = {
    "simple": "Simple",
    "soft_degree": "Soft degree",
    "gnn_cp_spsa": "GNN + CP · SPSA",
    "gnn_cp_autodiff": "GNN + CP · autodiff",
}
COLORS = plt.colormaps["tab10"](np.arange(10))


def record(args):
    scenario = json.loads(args.scenario.read_text())
    config, _ = config_from_scenario(scenario["config"])
    if args.steps is None:
        args.steps = int(scenario.get("horizon_steps", 10))
    if args.fixed_goal:
        config = replace(config, fixed_goal=True)
    start, goal = scenario_positions(scenario, args.robots, config.required_robustness,
                                     config.communication_radius)
    if config.dynamics == "double_integrator":
        particles = initial_controls(start, goal, args.particles, args.steps, config,
                                     0.1 * config.maximum_acceleration, args.seed)
    else:
        particles = initial_trajectories(start, goal, args.particles, args.steps,
                                         0.1 * config.maximum_step, args.seed)
    scorer = ConformalGraphScorer(args.checkpoint, args.calibration,
                                  config.cp_coverage, config.cp_kind)
    arrays = {"start": start, "goal": goal}
    metadata = {
        "robots": args.robots, "steps": args.steps, "iterations": args.iterations,
        "particles": args.particles, "seed": args.seed,
        "scenario": str(args.scenario), "methods": {},
    }
    for name, settings in comparison_configs(config):
        if name not in args.methods:
            continue
        began = perf_counter()
        selected, report = plan(start, goal, particles, scorer, settings,
                                args.iterations, args.seed, record_particles=True)
        elapsed = perf_counter() - began
        for key in ("particle_snapshots", "population_costs", "population_bounds"):
            arrays[f"{name}__{key}"] = report.pop(key)
        # Follow one fixed particle identity across updates, avoiding artificial
        # path changes from switching to a different best particle each frame.
        tracked = report["index"]
        if tracked is None:
            costs = arrays[f"{name}__population_costs"][-1]
            tracked = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        summary = trajectory_summary(selected, start, goal, scorer, settings, elapsed)
        metadata["methods"][name] = {
            "config": asdict(settings), "planner_report": report,
            "tracked_particle": tracked, "metrics": summary,
        }
        if selected is not None:
            arrays[f"{name}__selected"] = selected
        print(f"Recorded {name}: {report['status']}, {elapsed:.3f} s", flush=True)
    np.savez_compressed(args.output / "planner_snapshots.npz", **arrays)
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")


def scene_limits(start, goal, arrays, *, method=None, selected_only=False):
    points = [start, goal]
    for name in arrays.files:
        include = (name.endswith("__selected") if selected_only else
                   name.endswith("__particle_snapshots"))
        if include and (method is None or name.startswith(method + "__")):
            points.append(arrays[name].reshape(-1, 2))
    points = np.concatenate(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    half = max(float(np.ptp(points, axis=0).max()) / 2 + 0.045, 0.15)
    return center - half, center + half


def figure_axes(methods, lower, upper, title, subtitle):
    columns = min(2, len(methods))
    rows = math.ceil(len(methods) / columns)
    fig, grid = plt.subplots(rows, columns, figsize=(12, 5 * rows), squeeze=False)
    fig.patch.set_facecolor("#f4f7fb")
    fig.subplots_adjust(left=0.065, right=0.97, top=0.82, bottom=0.11,
                        wspace=0.16, hspace=0.30)
    fig.suptitle(title, fontsize=20, fontweight="bold", color="#172b46", y=0.98)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=11, color="#52637a")
    clock = fig.text(0.5, 0.89, "", ha="center", fontsize=12, color="#126782")
    axes = grid.flat
    result = {}
    for ax, name in zip(axes, methods):
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel("x (scenario units)"); ax.set_ylabel("y (scenario units)")
        ax.grid(alpha=0.16)
        for spine in ax.spines.values():
            spine.set_color("#c8d3e0")
        result[name] = ax
    for ax in list(grid.flat)[len(methods):]:
        ax.set_visible(False)
    return fig, result, clock


def make_optimization(arrays, metadata):
    start, goal = arrays["start"], arrays["goal"]
    methods = list(metadata["methods"])
    lower, upper = scene_limits(start, goal, arrays)
    fig, axes, clock = figure_axes(
        methods, lower, upper, "How candidate trajectories change during optimization",
        f"{metadata['robots']} robots · {metadata['steps']} future waypoints · same initial particles and seed",
    )
    for ax in axes.values():
        ax.scatter(*start.T, s=36, c=COLORS[:len(start)], edgecolors="white", zorder=4)
        ax.scatter(*goal.T, s=95, marker="*", c=COLORS[:len(start)], edgecolors="#22354a", linewidths=0.4, zorder=4)
    artists = {}
    for name, ax in axes.items():
        # Keep small updates legible without hiding large excursions in other
        # methods. Each panel uses a fixed scale spanning its own populations.
        lo, hi = scene_limits(start, goal, arrays, method=name)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        cloud = LineCollection([], colors="#718096", alpha=0.10, linewidths=0.6)
        ax.add_collection(cloud)
        initial = arrays[f"{name}__particle_snapshots"][0, metadata["methods"][name]["tracked_particle"]]
        initial = np.concatenate((start[None], initial))
        for robot in range(len(start)):
            ax.plot(*initial[:, robot].T, color=COLORS[robot], ls="--", alpha=0.25, lw=1)
        lines = [ax.plot([], [], color=COLORS[r], lw=1.8, marker=".", ms=4)[0]
                 for r in range(len(start))]
        artists[name] = cloud, lines
    handles = [Line2D([], [], color="#64748b", marker="o", ls="none", label="Start"),
               Line2D([], [], color="#64748b", marker="*", ls="none", label="Goal"),
               Line2D([], [], color="#64748b", ls="--", label="Initial tracked trajectory"),
               Line2D([], [], color="#126782", label="Updated tracked trajectory")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.018, "Colors identify robots; gray paths are other particles. Panel scales differ to show full updates. Earlier candidates may be selected.", ha="center", fontsize=9)

    def update(population):
        clock.set_text(f"SVGD update {population} / {metadata['iterations']}  ·  0 = initialization")
        for name, ax in axes.items():
            snapshots = arrays[f"{name}__particle_snapshots"][population]
            paths = np.concatenate((np.broadcast_to(start, (len(snapshots), 1, *start.shape)), snapshots), axis=1)
            cloud, lines = artists[name]
            cloud.set_segments(paths.transpose(0, 2, 1, 3).reshape(-1, paths.shape[1], 2))
            info = metadata["methods"][name]
            index = info["tracked_particle"]
            for robot, line in enumerate(lines):
                line.set_data(*paths[index, :, robot].T)
            bounds = arrays[f"{name}__population_bounds"][population]
            costs = arrays[f"{name}__population_costs"][population]
            target = info["config"]["required_robustness"]
            passing = np.count_nonzero((bounds >= target) & np.isfinite(costs))
            ax.set_title(f"{TITLES[name]}  |  {passing}/{len(bounds)} candidates pass CP\n"
                         f"Tracked particle {index}: CP min {bounds[index]:.0f}, cost {costs[index]:.4f}", fontsize=11)
        return []

    return fig, update


def make_rollout(arrays, metadata):
    start, goal = arrays["start"], arrays["goal"]
    methods = list(metadata["methods"])
    lower, upper = scene_limits(start, goal, arrays, selected_only=True)
    fig, axes, clock = figure_axes(
        methods, lower, upper, "Selected trajectories: waypoint-by-waypoint playback",
        "Playback of the planned waypoints · no closed-loop controller or interpolated safety certification",
    )
    artists, paths = {}, {}
    for name, ax in axes.items():
        info = metadata["methods"][name]
        ax.set_title(f"{TITLES[name]}  |  solver {info['metrics']['wall_seconds']:.3f} s", fontsize=12)
        ax.scatter(*goal.T, s=100, marker="*", c=COLORS[:len(start)], edgecolors="#22354a", linewidths=0.4)
        if f"{name}__selected" not in arrays:
            ax.text(0.5, 0.5, "No CP-feasible trajectory", transform=ax.transAxes, ha="center", color="#b91c1c")
            continue
        path = np.concatenate((start[None], arrays[f"{name}__selected"]))
        paths[name] = path
        for robot in range(len(start)):
            ax.plot(*path[:, robot].T, color=COLORS[robot], ls="--", alpha=0.28)
        links = LineCollection([], colors="#64748b", alpha=0.22, linewidths=0.8, zorder=1)
        ax.add_collection(links)
        trails = [ax.plot([], [], color=COLORS[r], lw=2)[0] for r in range(len(start))]
        dots = ax.scatter(*start.T, s=90, c=COLORS[:len(start)], edgecolors="white", linewidths=1, zorder=4)
        labels = [ax.text(*start[r], str(r + 1), fontsize=6, color="white", ha="center", va="center", zorder=5) for r in range(len(start))]
        status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                         bbox=dict(facecolor="white", edgecolor="#d3dce8", alpha=0.9))
        artists[name] = links, trails, dots, labels, status
    fig.legend(handles=[Line2D([], [], color="#64748b", label="Hard communication links"),
                        Line2D([], [], color="#126782", label="Waypoint trail"),
                        Line2D([], [], marker="*", color="#64748b", ls="none", label="Assigned goal")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04), frameon=False)
    fig.text(0.5, 0.018, "CP acceptance concerns sampled graphs. Speed and collision separation remain soft penalties.", ha="center", fontsize=10)

    def update(step):
        dt = next(iter(metadata["methods"].values()))["config"]["sampling_time"]
        clock.set_text(f"Waypoint {step} / {metadata['steps']}  ·  planned time {step * dt:.2f} s")
        for name, path in paths.items():
            links, trails, dots, labels, status = artists[name]
            info = metadata["methods"][name]
            positions = path[step]
            distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            pairs = np.transpose(np.nonzero(np.triu(distance <= info["config"]["communication_radius"], k=1)))
            links.set_segments(positions[pairs] if len(pairs) else [])
            dots.set_offsets(positions)
            for r, (trail, label) in enumerate(zip(trails, labels)):
                trail.set_data(*path[:step + 1, r].T)
                label.set_position(positions[r])
            if step == 0:
                status.set_text("Initial configuration\nCP filter checks future waypoints")
            else:
                cp = info["metrics"]["lower_bounds_by_step"][step - 1]
                separation = distance[~np.eye(len(start), dtype=bool)].min()
                maximum_step = np.linalg.norm(path[step] - path[step - 1], axis=-1).max()
                status.set_text(f"Hard CP bound: {cp}  |  target: {info['config']['required_robustness']}\n"
                                f"Min separation: {separation:.3f}  |  max move: {maximum_step:.3f}")
        return []

    return fig, update


def export(fig, update, count, output, fps):
    update(count - 1)
    fig.savefig(output.with_suffix(".png"), dpi=150, facecolor=fig.get_facecolor())
    animation = FuncAnimation(fig, update, frames=count, interval=1000 / fps, blit=False, repeat=False)
    animation.save(output.with_suffix(".mp4"), writer=FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=100)
    with matplotlib.rc_context({"animation.embed_limit": 100}):
        output.with_suffix(".html").write_text(animation.to_jshtml(fps=fps, default_mode="once"))
    print(f"Saved {output.name}: PNG, MP4, interactive HTML", flush=True)
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=PROJECT / "planning/examples/demo_scenario.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints/full_run_xK0R3I/new_gatv2.msgpack")
    parser.add_argument("--calibration", type=Path, default=PROJECT / "calib_data.json")
    parser.add_argument("--robots", type=int, default=10, choices=range(2, 101))
    parser.add_argument("--steps", type=int, help="Default: scenario horizon_steps, otherwise 10")
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=tuple(TITLES), default=["gnn_cp_autodiff"])
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/visualizations/11-02/double_integrator_10m")
    parser.add_argument("--load", action="store_true", help="Render saved snapshots without running planning again")
    parser.add_argument("--fixed-goal", action="store_true", help="Keep final waypoints exactly at goals")
    parser.add_argument("--record-only", action="store_true", help="Save planning data without rendering comparison views")
    parser.add_argument("--show", action="store_true", help="Also open Matplotlib windows on a desktop")
    parser.add_argument("--fps", type=float, default=2., help="Recorded states per playback second")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.robots > 10 and not args.record_only:
        parser.error("use --record-only for teams above 10, then render with planning.visualize_network_closeup")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.load:
        record(args)
    if args.record_only:
        return
    metadata = json.loads((args.output / "metrics.json").read_text())
    with np.load(args.output / "planner_snapshots.npz") as arrays:
        fig, update = make_optimization(arrays, metadata)
        optimization = export(fig, update, metadata["iterations"] + 1, args.output / "optimization_updates", args.fps)
        motion_fig, motion_update = make_rollout(arrays, metadata)
        playback = export(motion_fig, motion_update, metadata["steps"] + 1, args.output / "selected_motion", args.fps)
        if args.show:
            plt.show()
        else:
            plt.close("all")
    return optimization, playback


if __name__ == "__main__":
    main()
