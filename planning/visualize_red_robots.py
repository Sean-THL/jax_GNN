"""Single red-robot / green-goal video with exact graph diagnostics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-planner-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from planning.run_experiment import PROJECT, exact_robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "reports/visualizations/11-02/10robots_50steps_fixed_goal")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--show-inputs", action="store_true", help="Show acceleration arrows, communication radius and full workspace")
    args = parser.parse_args()
    folder = args.input
    metadata = json.loads((folder / "metrics.json").read_text())
    info = metadata["methods"]["gnn_cp_autodiff"]
    config = info["config"]
    with np.load(folder / "planner_snapshots.npz") as data:
        start, goal = data["start"], data["goal"]
        path = np.concatenate((start[None], data["gnn_cp_autodiff__selected"]))
    steps, robots = len(path) - 1, len(start)
    if robots > 10:
        raise ValueError("Exact robustness diagnostics are limited to 10 robots")
    dt, radius, target = config["sampling_time"], config["communication_radius"], config["required_robustness"]
    points = np.concatenate((path.reshape(-1, 2), goal))
    center = (points.min(0) + points.max(0)) / 2
    half = np.ptp(points, axis=0).max() / 2 + 0.06
    fig, ax = plt.subplots(figsize=(12, 8) if args.show_inputs else (8, 8), dpi=120)
    fig.patch.set_facecolor("#fafbfc")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.77, bottom=0.20 if args.show_inputs else 0.14)
    fig.suptitle(f"GNN–CP–autodiff · {robots} robots", y=0.97, fontsize=20, fontweight="bold")
    goal_mode = "fixed goals" if config.get("fixed_goal", False) else "soft goal cost (free endpoints)"
    fig.text(0.5, 0.925, f"{steps} steps · {steps * dt:.1f} s horizon · {goal_mode}", ha="center", fontsize=12, color="#52637a")
    clock = fig.text(0.5, 0.88, "", ha="center", fontsize=12)
    diagnostics = fig.text(0.5, 0.84, "", ha="center", fontsize=12)
    cp_text = fig.text(0.5, 0.80, "", ha="center", fontsize=10, color="#52637a")
    ax.set_aspect("equal")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    if args.show_inputs:
        if config.get("dynamics") != "double_integrator":
            raise ValueError("--show-inputs requires double-integrator data")
        scenario = json.loads(Path(metadata["scenario"]).read_text())
        workspace = scenario.get("workspace", {})
        lower = workspace.get("lower", points.min(0)-1)
        upper = workspace.get("upper", points.max(0)+1)
        ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1])
        ax.scatter(*start.T, s=40, facecolors="none", edgecolors="#df2929", alpha=0.35, zorder=2)
        y = upper[1] - 0.8
        ax.annotate("", xy=(goal[0,0], y), xytext=(start[0,0], y),
                    arrowprops=dict(arrowstyle="<->", color="#52637a", lw=1.2))
        displacement = goal-start
        distances = np.linalg.norm(displacement, axis=1)
        transfer_label = (f"Independent assigned goals: {distances.min():.0f}–{distances.max():.0f} m away"
                          if np.ptp(displacement, axis=0).max() > 0.01 else
                          f"{distances[0]:.0f} m start-to-goal transfer")
        ax.text((start[0,0]+goal[0,0])/2, y+0.15,
                transfer_label, ha="center", fontsize=10, color="#52637a")
    unit = "m" if config.get("dynamics") == "double_integrator" else "scenario units"
    ax.set_xlabel(f"x ({unit})"); ax.set_ylabel(f"y ({unit})")
    ax.grid(alpha=0.10)
    # Larger goal dots leave a visible green rim when robots arrive.
    ax.scatter(*goal.T, s=240, color="#159447", edgecolors="white", linewidths=1, zorder=3)
    for r in range(robots):
        ax.annotate(str(r + 1), goal[r], xytext=(9, 8), textcoords="offset points", color="#086b2c", fontsize=9, fontweight="bold", zorder=6)
    links = LineCollection([], colors="gray", alpha=0.5, linestyles="--", linewidths=0.9, zorder=1)
    ax.add_collection(links)
    dots = ax.scatter(*start.T, s=120, color="#df2929", edgecolors="white", linewidths=0.8, zorder=5)
    labels = [ax.text(*start[r], str(r + 1), ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=6) for r in range(robots)]
    input_arrows = radius_circle = None
    if args.show_inputs:
        input_arrows = ax.quiver(start[:,0], start[:,1], np.zeros(robots), np.zeros(robots),
                                 color="#e58a12", angles="xy", scale_units="xy", scale=1., width=0.004, zorder=4)
        radius_circle = Circle(start[0], radius, fill=False, edgecolor="#2478bc", linestyle=":", linewidth=1.5, alpha=0.75, zorder=2)
        ax.add_patch(radius_circle)
        fig.text(0.5, 0.11, f"Blue circle: R = {radius:g} m around robot 1   ·   Orange arrows: acceleration u (1 m on plot = 1 m/s²)", ha="center", fontsize=10, color="#52637a")
    fig.legend(handles=[Line2D([], [], marker="o", color="#df2929", ls="none", label="Robot"), Line2D([], [], marker="o", color="#159447", ls="none", label="Assigned goal"), Line2D([], [], color="gray", alpha=0.5, ls="--", label="Communication link")], loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    fig.text(0.5, 0.025, "½-speed playback · exact r and degree checked on displayed graphs" +
             (" · workspace outline is a reference, not a wall constraint" if args.show_inputs else ""), ha="center", fontsize=9, color="#52637a")
    cache, graph_metrics = {}, []
    def graph_at(positions):
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        edges = (distance <= radius) & ~np.eye(robots, dtype=bool)
        key = edges.tobytes()
        if key not in cache:
            source, target_edges = np.nonzero(edges)
            record = {"features": np.column_stack((edges.sum(1), np.ones(robots))), "edge_index": np.stack((source, target_edges))}
            cache[key] = exact_robustness(record)
        return edges, int(edges.sum(1).min()), cache[key]
    for step, positions in enumerate(path):
        edges, degree, robustness = graph_at(positions)
        graph_metrics.append({"step": step, "time": step * dt, "exact_r": robustness, "minimum_degree": degree, "cp_lower_bound": None if step == 0 else info["metrics"]["lower_bounds_by_step"][step - 1]})
    fps, hold = args.fps, args.fps
    moving = max(1, round(steps * dt * 2 * fps))
    def update(frame):
        progress = np.clip((frame - hold) / moving, 0., 1.) * steps
        step = min(int(progress), steps - 1)
        fraction = progress - step
        if config.get("dynamics") == "double_integrator":
            report = info["planner_report"]
            velocity = np.asarray(config["initial_velocity"] if step == 0 else report["velocities"][step-1])
            acceleration = np.asarray(report["accelerations"][step])
            elapsed = fraction*dt
            positions = path[step] + elapsed*velocity + 0.5*elapsed**2*acceleration
        else:
            positions = (1 - fraction) * path[step] + fraction * path[step + 1]
        edges, degree, robustness = graph_at(positions)
        pairs = np.transpose(np.nonzero(np.triu(edges, 1)))
        links.set_segments(positions[pairs] if len(pairs) else [])
        dots.set_offsets(positions)
        for r, label in enumerate(labels):
            label.set_position(positions[r])
        if input_arrows is not None:
            input_arrows.set_offsets(positions)
            input_arrows.set_UVC(acceleration[:,0], acceleration[:,1])
            radius_circle.center = positions[0]
        shown = min(int(progress), steps)
        clock.set_text(f"Time {progress * dt:.2f} / {steps * dt:.2f} s   ·   Step {shown} / {steps}")
        if args.show_inputs:
            maximum_u = np.linalg.norm(acceleration, axis=1).max()
            clock.set_text(clock.get_text() + f"   ·   dt = {dt:g} s   ·   max |u| = {maximum_u:.2f} / {config['maximum_acceleration']:g} m/s²")
        diagnostics.set_text(f"Exact r-robustness: {robustness}   |   Min degree: {degree}   |   Target r: {target}")
        diagnostics.set_color("#086b2c" if robustness >= target else "#bc2525")
        cp = graph_metrics[shown]["cp_lower_bound"]
        cp_text.set_text("Initial state" if cp is None else f"CP lower bound at waypoint {shown}: {cp}  (sampled waypoint only)")
        return []
    output = folder / ("double_integrator_explained.mp4" if args.show_inputs else f"red_robots_{steps}steps.mp4")
    animation = FuncAnimation(fig, update, frames=moving + 2 * hold + 1, interval=1000 / fps, blit=False)
    animation.save(output, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=3500, extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    update(moving + hold)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    (folder / "graph_metrics_by_waypoint.json").write_text(json.dumps(graph_metrics, indent=2) + "\n")
    (folder / "playback_graph_summary.json").write_text(json.dumps({"minimum_exact_r_across_rendered_frames": min(cache.values()), "distinct_rendered_graphs": len(cache), "endpoint_max_error": float(np.linalg.norm(path[-1] - goal, axis=1).max())}, indent=2) + "\n")
    plt.close(fig)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
