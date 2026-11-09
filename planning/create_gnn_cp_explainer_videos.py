"""Create three MP4 explainers for the GNN+CP SVGD planning workflow."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from planning.run_experiment import PROJECT, config_from_scenario, initial_trajectories, scenario_positions
from planning.svgd_cp_planner import ConformalGraphScorer, plan


BG = "#07111f"
PANEL = "#0d1b2b"
GRID = "#29445f"
TEXT = "#ecf4ff"
MUTED = "#9db0c5"
CYAN = "#53d8fb"
GREEN = "#4fd1a1"
ORANGE = "#f2a65a"
GOLD = "#ffd166"
PURPLE = "#c792ea"


def setup_axis(axis, bounds):
    axis.set_facecolor(PANEL)
    axis.set_aspect("equal")
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.grid(color=GRID, alpha=0.25, linewidth=0.7)
    axis.tick_params(colors="#8297ad", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#35506b")


def hard_edges(points, radius):
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return [[points[i], points[j]] for i in range(len(points)) for j in range(i + 1, len(points))
            if distances[i, j] <= radius]


def video_writer(fps=30):
    return FFMpegWriter(fps=fps, codec="libx264", bitrate=5000,
                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def box(axis, xy, width, height, title, body="", edge=CYAN, alpha=1.0, size=12):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018",
        transform=axis.transAxes, facecolor="#102237", edgecolor=edge,
        linewidth=2, alpha=alpha,
    )
    axis.add_patch(patch)
    title_artist = axis.text(xy[0] + width / 2, xy[1] + height * 0.67, title,
                             transform=axis.transAxes, ha="center", va="center",
                             color=TEXT, fontsize=size, fontweight="bold", alpha=alpha)
    body_artist = None
    if body:
        body_artist = axis.text(xy[0] + width / 2, xy[1] + height * 0.28, body,
                                transform=axis.transAxes, ha="center", va="center",
                                color=MUTED, fontsize=size - 2, alpha=alpha)
    patch._box_texts = tuple(x for x in (title_artist, body_artist) if x is not None)
    return patch


def arrow(axis, start, end, color=CYAN, alpha=1.0):
    patch = FancyArrowPatch(start, end, transform=axis.transAxes, arrowstyle="-|>",
                            mutation_scale=18, linewidth=2, color=color, alpha=alpha)
    axis.add_patch(patch)
    return patch


def create_planner_data():
    scenario = json.loads((PROJECT / "planning/examples/legacy_waypoint_scenario.json").read_text())
    base, _ = config_from_scenario(scenario["config"])
    # Five robots cannot satisfy the project's F=2 target. These explainers use
    # the mathematically feasible illustrative setting F=1 -> r_required=3.
    config = replace(base, required_robustness=3, connectivity_mode="smooth",
                     sampling_time=0.1, cp_directions=2)
    start, goal = scenario_positions(
        scenario, 5, config.required_robustness, config.communication_radius
    )
    particles = initial_trajectories(start, goal, 8, 3, 0.025, 42)
    scorer = ConformalGraphScorer(
        PROJECT / "checkpoints/full_run_xK0R3I/new_gatv2.msgpack",
        PROJECT / "calib_data.json", config.cp_coverage, config.cp_kind,
    )
    selected, report = plan(
        start, goal, particles, scorer, config, iterations=4, seed=42,
        record_particles=True,
    )
    return start, goal, config, scorer, selected, report


def workspace_bounds(start, goal, snapshots):
    points = np.concatenate((start, goal, snapshots.reshape(-1, 2)))
    lo, hi = points.min(axis=0) - 0.08, points.max(axis=0) + 0.08
    span = max(hi - lo)
    center = 0.5 * (lo + hi)
    lo, hi = center - span / 2, center + span / 2
    return lo[0], hi[0], lo[1], hi[1]


def make_update_video(output, start, goal, config, report):
    fps, seconds_per_iteration = 30, 2.0
    snapshots = np.asarray(report["particle_snapshots"])
    costs = np.asarray(report["population_costs"])
    bounds = np.asarray(report["population_bounds"])
    best_snapshot, particle = np.unravel_index(np.argmin(costs), costs.shape)
    paths = snapshots[:, particle]
    initial = paths[0]
    bounds_xy = workspace_bounds(start, goal, snapshots)
    colors = plt.colormaps["tab10"](np.arange(5))

    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.55, 0.75], left=0.055, right=0.96,
                               bottom=0.09, top=0.82, wspace=0.12)
    axis = figure.add_subplot(grid[0, 0])
    info = figure.add_subplot(grid[0, 1]); info.axis("off"); info.set_facecolor(BG)
    setup_axis(axis, bounds_xy)
    figure.suptitle("Video 1 — How SVGD updates one GNN+CP trajectory particle",
                    color=TEXT, fontsize=23, fontweight="bold", y=0.96)
    subtitle = figure.text(
        0.5, 0.905,
        "5-robot educational example • illustrative F = 1, r_required = 3 • three future waypoints",
        ha="center", color=MUTED, fontsize=13,
    )
    stage_text = figure.text(0.5, 0.855, "", ha="center", color=CYAN,
                             fontsize=16, fontweight="bold")
    axis.scatter(goal[:, 0], goal[:, 1], s=150, marker="*", facecolors="none",
                 edgecolors=GOLD, linewidths=1.8, zorder=5)
    axis.scatter(start[:, 0], start[:, 1], s=42, color="#718399", alpha=0.7, zorder=4)
    initial_lines = [axis.plot(np.r_[start[i, 0], initial[:, i, 0]],
                               np.r_[start[i, 1], initial[:, i, 1]],
                               linestyle="--", color=colors[i], alpha=0.28, linewidth=1.5)[0]
                     for i in range(5)]
    current_lines = [axis.plot([], [], color=colors[i], linewidth=2.8, zorder=4)[0] for i in range(5)]
    previous_lines = [axis.plot([], [], color="#c8d5e5", linewidth=1.0, alpha=0.22, zorder=2)[0]
                      for _ in range(5)]
    edge_collection = LineCollection([], colors=CYAN, linewidths=1.4, alpha=0.45, zorder=2)
    axis.add_collection(edge_collection)
    robots = axis.scatter(start[:, 0], start[:, 1], s=130, c=colors, edgecolors="white",
                          linewidths=1.2, zorder=7)
    labels = [axis.text(*start[i], str(i + 1), ha="center", va="center", color=BG,
                        fontsize=8, fontweight="bold", zorder=8) for i in range(5)]
    waypoint_dots = axis.scatter([], [], s=24, color=PURPLE, alpha=0.8, zorder=6)

    box(info, (0.05, 0.76), 0.9, 0.16, "1. Build graph at every waypoint",
        "Hard edge if distance ≤ R = 0.30", CYAN)
    box(info, (0.05, 0.54), 0.9, 0.16, "2. GNN inference + CP lower bound",
        "Checkpoint is trained offline; q = 0 here", PURPLE)
    box(info, (0.05, 0.32), 0.9, 0.16, "3. Cost and SPSA gradient",
        "Goal + motion + separation + degree + CP", ORANGE)
    box(info, (0.05, 0.10), 0.9, 0.16, "4. SVGD particle update",
        "Attraction lowers cost; kernel repulsion keeps diversity", GREEN)
    status = info.text(0.5, 0.01, "", transform=info.transAxes, ha="center", va="bottom",
                       color=TEXT, fontsize=12,
                       bbox={"boxstyle": "round,pad=0.6", "facecolor": PANEL,
                             "edgecolor": ORANGE, "linewidth": 1.5})

    iteration_frames = round(seconds_per_iteration * fps)
    total_frames = len(paths) * iteration_frames + fps

    def update(frame):
        iteration = min(frame // iteration_frames, len(paths) - 1)
        local = (frame % iteration_frames) / max(1, iteration_frames - 1)
        if frame >= len(paths) * iteration_frames:
            iteration, local = len(paths) - 1, 1.0
        previous = paths[max(iteration - 1, 0)]
        eased = local * local * (3 - 2 * local)
        path = (1 - eased) * previous + eased * paths[iteration]
        active_step = min(2, int(local * 3))
        current = path[active_step]
        edge_collection.set_segments(hard_edges(current, config.communication_radius))
        robots.set_offsets(current)
        waypoint_dots.set_offsets(path.reshape(-1, 2))
        for i in range(5):
            full = np.concatenate((start[i][None], path[:, i]), axis=0)
            old = np.concatenate((start[i][None], previous[:, i]), axis=0)
            current_lines[i].set_data(full[:, 0], full[:, 1])
            previous_lines[i].set_data(old[:, 0], old[:, 1])
            labels[i].set_position(current[i])
        cp = int(bounds[iteration, particle])
        cost = float(costs[iteration, particle])
        retained = "  ← lowest cost retained" if iteration == best_snapshot else ""
        stage_text.set_text(
            f"SVGD iteration {iteration} of {len(paths)-1}  •  displaying waypoint {active_step+1} of 3"
        )
        status.set_text(
            f"Particle {particle + 1}\nCP lower bound: {cp} / 3\n"
            f"Total cost: {cost:.2f}{retained}\n"
            f"Best-ever snapshot: iteration {best_snapshot}"
        )
        status.get_bbox_patch().set_edgecolor(GREEN if cp >= 3 else ORANGE)
        return [stage_text, edge_collection, robots, waypoint_dots, status,
                *current_lines, *previous_lines, *labels, *initial_lines]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=video_writer(fps))
    plt.close(figure)


def make_selection_video(output, start, goal, config, report):
    fps = 30
    snapshots = np.asarray(report["particle_snapshots"])
    costs = np.asarray(report["population_costs"])
    bounds = np.asarray(report["population_bounds"])
    snapshot, winner = np.unravel_index(np.argmin(costs), costs.shape)
    particles = snapshots[snapshot]
    particle_costs = costs[snapshot]
    particle_bounds = bounds[snapshot]
    order = np.argsort(particle_costs)
    bounds_xy = workspace_bounds(start, goal, snapshots)
    colors = plt.colormaps["tab10"](np.arange(5))

    figure, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=120, facecolor=BG)
    figure.subplots_adjust(left=0.04, right=0.97, bottom=0.20, top=0.80, wspace=0.10, hspace=0.25)
    figure.suptitle("Video 2 — How GNN+CP ranks several SVGD trajectory particles",
                    color=TEXT, fontsize=23, fontweight="bold", y=0.96)
    figure.text(0.5, 0.905,
                "At planning time the GNN performs inference; it is not retrained for each trajectory",
                ha="center", color=MUTED, fontsize=13)
    headline = figure.text(0.5, 0.845, "", ha="center", color=CYAN,
                           fontsize=15, fontweight="bold")
    panels = []
    for index, axis in enumerate(axes.flat):
        setup_axis(axis, bounds_xy)
        axis.set_xticks([]); axis.set_yticks([])
        axis.scatter(goal[:, 0], goal[:, 1], s=55, marker="*", facecolors="none",
                     edgecolors=GOLD, linewidths=1.1, zorder=5)
        path = particles[index]
        for robot in range(5):
            full = np.concatenate((start[robot][None], path[:, robot]), axis=0)
            axis.plot(full[:, 0], full[:, 1], color=colors[robot], linewidth=1.7, alpha=0.9)
        final = path[-1]
        edge_lines = LineCollection(hard_edges(final, config.communication_radius),
                                    colors=CYAN, linewidths=0.8, alpha=0.35)
        axis.add_collection(edge_lines)
        axis.scatter(final[:, 0], final[:, 1], s=45, c=colors, edgecolors="white", linewidths=0.7, zorder=6)
        title = axis.set_title(
            f"Particle {index+1}  •  CP {int(particle_bounds[index])}/3  •  cost {particle_costs[index]:.2f}",
            color=MUTED, fontsize=10, fontweight="bold",
        )
        panels.append((axis, title))

    pipeline = figure.text(
        0.5, 0.115,
        "Positions at 3 waypoints  →  3 hard-radius graphs  →  offline-trained GNN predictions  "
        "→  CP correction q = 0  →  minimum bound over time  →  combined trajectory cost",
        ha="center", color=TEXT, fontsize=12,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": PANEL, "edgecolor": CYAN, "linewidth": 1.5},
    )
    footer = figure.text(0.5, 0.045, "", ha="center", color=TEXT, fontsize=13, fontweight="bold")
    scan_frames, hold_frames = 24, 75
    total_frames = len(particles) * scan_frames + hold_frames

    def update(frame):
        if frame < len(particles) * scan_frames:
            current = frame // scan_frames
            headline.set_text(f"Evaluate particle {current+1}: graph → GNN → CP → cost")
            footer.set_text("Scanning the SVGD population…")
        else:
            current = winner
            headline.set_text(f"Select minimum-cost certified particle: particle {winner+1}")
            footer.set_text(
                f"Winner at SVGD snapshot {snapshot}: cost {particle_costs[winner]:.2f}, "
                f"CP lower bound {int(particle_bounds[winner])} ≥ 3"
            )
        for index, (axis, title) in enumerate(panels):
            active = index == current
            winner_now = frame >= len(particles) * scan_frames and index == winner
            color = GREEN if winner_now else (ORANGE if active else "#35506b")
            width = 3.5 if (active or winner_now) else 1.0
            for spine in axis.spines.values():
                spine.set_color(color); spine.set_linewidth(width)
            title.set_color(GREEN if winner_now else (TEXT if active else MUTED))
        return [headline, footer, pipeline, *(title for _, title in panels)]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=video_writer(fps))
    plt.close(figure)


def make_pipeline_video(output):
    fps, seconds = 30, 14
    figure, axis = plt.subplots(figsize=(16, 9), dpi=120, facecolor=BG)
    axis.set_facecolor(BG); axis.axis("off")
    figure.suptitle("Video 3 — Complete GNN + conformal + SVGD planning workflow",
                    color=TEXT, fontsize=24, fontweight="bold", y=0.965)
    axis.text(0.03, 0.90, "OFFLINE — done once", transform=axis.transAxes,
              color=PURPLE, fontsize=15, fontweight="bold")
    axis.text(0.03, 0.54, "ONLINE — repeated while planning", transform=axis.transAxes,
              color=CYAN, fontsize=15, fontweight="bold")

    offline_specs = [
        ((0.04, 0.70), 0.18, 0.13, "Graph dataset", "binary radius graphs"),
        ((0.29, 0.70), 0.18, 0.13, "Exact r labels", "subset enumeration"),
        ((0.54, 0.70), 0.18, 0.13, "Train GATv2", "learn graph → r"),
        ((0.79, 0.70), 0.17, 0.13, "Calibrate CP", "one-sided q = 0"),
    ]
    online_specs = [
        ((0.03, 0.35), 0.15, 0.12, "Starts + goals", "N robots"),
        ((0.22, 0.35), 0.15, 0.12, "Sample particles", "3-step trajectories"),
        ((0.41, 0.35), 0.15, 0.12, "Build graph", "at every waypoint"),
        ((0.60, 0.35), 0.15, 0.12, "GNN inference", "predict r class"),
        ((0.79, 0.35), 0.17, 0.12, "CP lower bound", "prediction − q"),
        ((0.13, 0.10), 0.19, 0.12, "Trajectory cost", "goal + safety + CP"),
        ((0.405, 0.10), 0.19, 0.12, "SPSA + SVGD", "gradient + diversity"),
        ((0.68, 0.10), 0.19, 0.12, "Select best-ever", "verify and execute"),
    ]
    specs = offline_specs + online_specs
    patches = [box(axis, *spec, edge=(PURPLE if i < 4 else CYAN), alpha=0.13)
               for i, spec in enumerate(specs)]
    arrow_specs = [
        ((0.22, 0.765), (0.29, 0.765), PURPLE),
        ((0.47, 0.765), (0.54, 0.765), PURPLE),
        ((0.72, 0.765), (0.79, 0.765), PURPLE),
        ((0.18, 0.41), (0.22, 0.41), CYAN),
        ((0.37, 0.41), (0.41, 0.41), CYAN),
        ((0.56, 0.41), (0.60, 0.41), CYAN),
        ((0.75, 0.41), (0.79, 0.41), CYAN),
        ((0.875, 0.35), (0.27, 0.22), ORANGE),
        ((0.32, 0.16), (0.405, 0.16), GREEN),
        ((0.595, 0.16), (0.68, 0.16), GREEN),
        ((0.50, 0.10), (0.30, 0.35), GREEN),
        ((0.875, 0.70), (0.675, 0.47), PURPLE),
    ]
    arrows = [arrow(axis, s, e, c, alpha=0.10) for s, e, c in arrow_specs]
    narration = axis.text(
        0.5, 0.015, "", transform=axis.transAxes, ha="center", va="bottom",
        color=TEXT, fontsize=13,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": PANEL, "edgecolor": "#35506b"},
    )
    messages = [
        "Create binary communication graphs and exact robustness labels.",
        "Train GATv2 offline to predict the graph robustness class.",
        "Calibrate a one-sided conformal correction on held-out binary graphs.",
        "At runtime, sample diverse multi-robot trajectory particles.",
        "Convert every future waypoint into a hard-radius graph for GNN+CP certification.",
        "Combine goal, motion, separation, degree, and CP-shortfall costs.",
        "Use SPSA for the discrete CP term and SVGD for low cost plus particle diversity.",
        "Repeat updates, retain the best-ever particle, and verify before execution.",
    ]
    total_frames = seconds * fps

    def update(frame):
        progress = frame / max(1, total_frames - 1)
        stage = min(len(messages) - 1, int(progress * len(messages)))
        narration.set_text(messages[stage])
        active_boxes = min(len(patches), int(progress * (len(patches) + 2)) + 1)
        active_arrows = min(len(arrows), max(0, active_boxes - 1))
        for i, patch in enumerate(patches):
            alpha = 0.96 if i < active_boxes else 0.13
            patch.set_alpha(alpha)
            for text_artist in patch._box_texts:
                text_artist.set_alpha(alpha)
            patch.set_linewidth(2.8 if i == active_boxes - 1 else 1.6)
        for i, patch in enumerate(arrows):
            patch.set_alpha(0.9 if i < active_arrows else 0.10)
        return [narration, *patches, *arrows]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=video_writer(fps))
    plt.close(figure)


def main():
    output_dir = PROJECT / "reports/videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    start, goal, config, scorer, selected, report = create_planner_data()
    make_update_video(output_dir / "01_gnn_cp_svgd_trajectory_update.mp4",
                      start, goal, config, report)
    print("Saved video 1", flush=True)
    make_selection_video(output_dir / "02_gnn_cp_particle_selection.mp4",
                         start, goal, config, report)
    print("Saved video 2", flush=True)
    make_pipeline_video(output_dir / "03_project_pipeline_workflow.mp4")
    print("Saved video 3", flush=True)


if __name__ == "__main__":
    main()
