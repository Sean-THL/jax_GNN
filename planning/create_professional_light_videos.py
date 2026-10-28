"""Create three professional light-theme conceptual explainer videos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/videos"
BG = "#F5F7FB"
WHITE = "#FFFFFF"
NAVY = "#17324D"
MUTED = "#66788A"
LINE = "#D9E2EC"
BLUE = "#2563EB"
TEAL = "#0F9D8A"
ORANGE = "#F59E0B"
RED = "#DC5A5A"
PURPLE = "#7C5CFC"
GREEN = "#23A36D"
ROBOT_COLORS = ["#2563EB", "#0F9D8A", "#F59E0B", "#7C5CFC", "#DC5A5A"]


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def writer(fps=30):
    return FFMpegWriter(fps=fps, codec="libx264", bitrate=5000,
                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def card(axis, xy, width, height, title="", subtitle="", edge=LINE, face=WHITE,
         transform=None, radius=0.02, linewidth=1.4, zorder=2):
    transform = axis.transAxes if transform is None else transform
    patch = FancyBboxPatch(xy, width, height, boxstyle=f"round,pad={radius}",
                           transform=transform, facecolor=face, edgecolor=edge,
                           linewidth=linewidth, zorder=zorder)
    axis.add_patch(patch)
    title_artist = axis.text(xy[0] + width / 2, xy[1] + height * 0.64, title,
                             transform=transform, ha="center", va="center",
                             color=NAVY, fontsize=12, fontweight="bold", zorder=zorder + 1)
    subtitle_artist = axis.text(xy[0] + width / 2, xy[1] + height * 0.28, subtitle,
                                transform=transform, ha="center", va="center",
                                color=MUTED, fontsize=9.5, zorder=zorder + 1)
    patch._texts = (title_artist, subtitle_artist)
    return patch


def arrow(axis, start, end, color=BLUE, alpha=1.0, transform=None, zorder=3):
    transform = axis.transAxes if transform is None else transform
    patch = FancyArrowPatch(start, end, transform=transform, arrowstyle="-|>",
                            mutation_scale=17, linewidth=2.0, color=color,
                            alpha=alpha, zorder=zorder)
    axis.add_patch(patch)
    return patch


def hard_edges(points, radius):
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return [[points[i], points[j]] for i in range(len(points)) for j in range(i + 1, len(points))
            if distances[i, j] <= radius]


def title_block(figure, title, subtitle):
    figure.text(0.065, 0.935, title, color=NAVY, fontsize=25, fontweight="bold", ha="left")
    figure.text(0.065, 0.895, subtitle, color=MUTED, fontsize=12.5, ha="left")
    figure.add_artist(plt.Line2D([0.065, 0.935], [0.865, 0.865], transform=figure.transFigure,
                                 color=LINE, linewidth=1.2))


def conceptual_paths():
    start = np.array([[0.15, 0.35], [0.15, 0.50], [0.15, 0.65],
                      [0.30, 0.42], [0.30, 0.58]])
    goal = start + np.array([0.52, 0.02])
    fractions = np.array([0.34, 0.67, 1.0])[:, None, None]
    ideal = start[None] + fractions * (goal - start)[None]
    offsets = np.array([
        [[0.00, 0.11], [0.02, -0.10], [-0.01, 0.08], [0.02, 0.11], [-0.02, -0.12]],
        [[0.01, 0.08], [-0.01, -0.08], [0.02, 0.10], [-0.02, 0.09], [0.01, -0.09]],
        [[0.00, 0.06], [0.00, -0.06], [0.00, 0.07], [0.00, 0.06], [0.00, -0.07]],
    ])
    initial = ideal + offsets
    snapshots = np.stack([
        initial,
        0.62 * initial + 0.38 * ideal,
        0.28 * initial + 0.72 * ideal,
        ideal,
    ])
    return start, goal, snapshots


def make_video_1(path):
    fps, duration = 30, 12
    start, goal, snapshots = conceptual_paths()
    radius = 0.235
    bounds = (0.07, 0.78, 0.20, 0.80)
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    title_block(figure, "How GNN + conformal prediction guides an SVGD update",
                "A simplified five-robot example with three future waypoints • illustrative target r = 3")
    axis = figure.add_axes([0.06, 0.10, 0.62, 0.72], facecolor=WHITE)
    axis.set_xlim(bounds[:2]); axis.set_ylim(bounds[2:]); axis.set_aspect("equal")
    axis.grid(color="#EAF0F6", linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values(): spine.set_color(LINE)
    side = figure.add_axes([0.72, 0.10, 0.23, 0.72]); side.axis("off")

    axis.scatter(start[:, 0], start[:, 1], s=65, color="#C7D2E0", edgecolors=WHITE, zorder=4)
    axis.scatter(goal[:, 0], goal[:, 1], s=145, marker="*", facecolors="none",
                 edgecolors=ORANGE, linewidths=1.8, zorder=5)
    axis.text(0.09, 0.755, "START", color=MUTED, fontsize=9, fontweight="bold")
    axis.text(0.64, 0.755, "GOALS", color=ORANGE, fontsize=9, fontweight="bold")
    initial_lines = []
    current_lines = []
    update_arrows = []
    for robot in range(5):
        baseline = np.concatenate((start[robot][None], snapshots[0, :, robot]), axis=0)
        initial_lines.append(axis.plot(baseline[:, 0], baseline[:, 1], "--",
                                       color=ROBOT_COLORS[robot], alpha=0.28, linewidth=1.6)[0])
        current_lines.append(axis.plot([], [], color=ROBOT_COLORS[robot], linewidth=3.0, zorder=5)[0])
        update_arrows.extend([
            arrow(axis, (0, 0), (0, 0), color=PURPLE, alpha=0.0,
                  transform=axis.transData, zorder=4) for _ in range(3)
        ])
    graph = LineCollection([], colors=TEAL, linewidths=1.5, alpha=0.55, zorder=2)
    axis.add_collection(graph)
    robots = axis.scatter(start[:, 0], start[:, 1], s=145, c=ROBOT_COLORS,
                          edgecolors=WHITE, linewidths=1.4, zorder=7)
    labels = [axis.text(*start[i], str(i + 1), ha="center", va="center", color=WHITE,
                        fontsize=8, fontweight="bold", zorder=8) for i in range(5)]
    waypoints = axis.scatter([], [], s=27, color=PURPLE, alpha=0.65, zorder=6)
    stage = figure.text(0.37, 0.835, "", ha="center", color=BLUE, fontsize=15, fontweight="bold")

    stages = [
        ("1", "Build a graph", "At each waypoint, connect robots within R", BLUE),
        ("2", "GNN predicts robustness", "The trained model evaluates each graph", PURPLE),
        ("3", "CP creates a lower bound", "Use the worst bound over the trajectory", ORANGE),
        ("4", "SVGD updates the path", "Lower cost while keeping particles diverse", TEAL),
    ]
    stage_cards = []
    for index, (number, name, text, color) in enumerate(stages):
        y = 0.80 - index * 0.19
        patch = card(side, (0.04, y), 0.92, 0.145, f"{number}. {name}", text, edge=color)
        stage_cards.append(patch)
    metric = side.text(0.5, 0.015, "", transform=side.transAxes, ha="center", va="bottom",
                       color=NAVY, fontsize=11.5,
                       bbox={"boxstyle": "round,pad=0.65", "facecolor": WHITE,
                             "edgecolor": LINE, "linewidth": 1.4})
    cp_values = [1, 2, 3, 3]
    costs = [18.6, 7.4, 2.3, 1.2]
    total_frames = fps * duration

    def update(frame):
        progress = frame / max(1, total_frames - 1)
        scaled = progress * (len(snapshots) - 1)
        left = min(int(scaled), len(snapshots) - 1)
        right = min(left + 1, len(snapshots) - 1)
        blend = ease(scaled - left)
        trajectory = (1 - blend) * snapshots[left] + blend * snapshots[right]
        active_step = min(2, int((progress * 9) % 3))
        current = trajectory[active_step]
        graph.set_segments(hard_edges(current, radius))
        robots.set_offsets(current)
        waypoints.set_offsets(trajectory.reshape(-1, 2))
        for robot in range(5):
            full = np.concatenate((start[robot][None], trajectory[:, robot]), axis=0)
            current_lines[robot].set_data(full[:, 0], full[:, 1])
            labels[robot].set_position(current[robot])
            for step_index in range(3):
                item = update_arrows[robot * 3 + step_index]
                old = snapshots[left, step_index, robot]
                new = snapshots[right, step_index, robot]
                item.set_positions(old, new)
                item.set_alpha(0.65 if right != left and np.linalg.norm(new - old) > 0.004 else 0.0)
        stage.set_text(f"SVGD update {left} → {right}   •   graph at waypoint {active_step + 1} of 3")
        metric.set_text(
            f"CP lower bound   {cp_values[right]} / 3\n"
            f"Combined cost    {costs[right]:.1f}\n\n"
            f"{'CERTIFIED' if cp_values[right] >= 3 else 'update again'}"
        )
        metric.get_bbox_patch().set_edgecolor(GREEN if cp_values[right] >= 3 else ORANGE)
        active_card = min(3, int((progress * 8) % 4))
        for i, patch in enumerate(stage_cards):
            patch.set_facecolor("#EEF5FF" if i == active_card else WHITE)
            patch.set_linewidth(2.6 if i == active_card else 1.4)
        return [graph, robots, waypoints, stage, metric, *current_lines, *labels,
                *update_arrows, *initial_lines, *stage_cards]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=writer(fps)); plt.close(figure)


def candidate_trajectory(index):
    start, goal, snapshots = conceptual_paths()
    base = snapshots[-1]
    patterns = [0.11, 0.025, 0.07, 0.012, 0.15, 0.045]
    amplitude = patterns[index]
    signs = np.array([1, -1, 1, -1, 1])[None, :, None]
    offsets = np.zeros_like(base)
    offsets[..., 1:] = amplitude * signs * np.array([0.6, -0.2, 0.7])[:, None, None]
    return start, goal, base + offsets


def make_video_2(path):
    fps, duration = 30, 11
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    title_block(figure, "How SVGD + GNN + CP selects one trajectory",
                "Six candidate particles are evaluated consistently; the lowest-cost certified path is retained")
    pipeline_axis = figure.add_axes([0.07, 0.73, 0.86, 0.12]); pipeline_axis.axis("off")
    labels = [("Trajectory", "3 waypoints"), ("Graphs", "one per waypoint"),
              ("GNN", "offline-trained"), ("CP bound", "worst over time"),
              ("Total cost", "rank candidates")]
    pipeline_cards, pipeline_arrows = [], []
    for i, (name, sub) in enumerate(labels):
        x = i * 0.205
        pipeline_cards.append(card(pipeline_axis, (x, 0.12), 0.16, 0.72, name, sub,
                                   edge=[BLUE, TEAL, PURPLE, ORANGE, GREEN][i]))
        if i < len(labels) - 1:
            pipeline_arrows.append(arrow(pipeline_axis, (x + 0.165, 0.48), (x + 0.202, 0.48),
                                          color=MUTED))
    figure.text(0.93, 0.865, "GNN training happens offline", ha="right", color=PURPLE,
                fontsize=10.5, fontweight="bold")

    card_positions = [(0.07, 0.40), (0.375, 0.40), (0.68, 0.40),
                      (0.07, 0.08), (0.375, 0.08), (0.68, 0.08)]
    cp = [1, 3, 2, 3, 0, 3]
    costs = [22.4, 3.8, 10.2, 1.4, 35.0, 5.6]
    winner = 3
    candidate_axes, borders, metric_texts = [], [], []
    for i, (x, y) in enumerate(card_positions):
        border = FancyBboxPatch((x, y), 0.25, 0.255, boxstyle="round,pad=0.012",
                                transform=figure.transFigure, facecolor=WHITE, edgecolor=LINE,
                                linewidth=1.4, zorder=1)
        figure.add_artist(border); borders.append(border)
        ax = figure.add_axes([x + 0.015, y + 0.045, 0.15, 0.18], facecolor=WHITE, zorder=2)
        ax.set_xlim(0.08, 0.76); ax.set_ylim(0.23, 0.76); ax.set_aspect("equal"); ax.axis("off")
        start, goal, trajectory = candidate_trajectory(i)
        ax.scatter(goal[:, 0], goal[:, 1], marker="*", s=40, facecolors="none",
                   edgecolors=ORANGE, linewidths=1.0)
        for robot in range(5):
            full = np.concatenate((start[robot][None], trajectory[:, robot]), axis=0)
            ax.plot(full[:, 0], full[:, 1], color=ROBOT_COLORS[robot], linewidth=1.6)
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=34, c=ROBOT_COLORS,
                   edgecolors=WHITE, linewidths=0.6)
        candidate_axes.append(ax)
        figure.text(x + 0.018, y + 0.225, f"PARTICLE {i+1}", color=NAVY, fontsize=10.5,
                    fontweight="bold", zorder=3)
        metric_texts.append(figure.text(x + 0.175, y + 0.145, "", color=MUTED,
                                        fontsize=10.5, ha="left", va="center", zorder=3))
    headline = figure.text(0.5, 0.69, "", ha="center", color=BLUE, fontsize=14, fontweight="bold")
    footer = figure.text(0.5, 0.028, "", ha="center", color=NAVY, fontsize=12.5,
                         fontweight="bold")
    total_frames = duration * fps

    def update(frame):
        progress = frame / max(1, total_frames - 1)
        pipeline_stage = min(4, int(progress * 6))
        scan_progress = np.clip((progress - 0.22) / 0.53, 0, 1)
        current = min(5, int(scan_progress * 6))
        final = progress > 0.78
        for i, patch in enumerate(pipeline_cards):
            patch.set_facecolor("#EEF5FF" if i <= pipeline_stage else WHITE)
            patch.set_linewidth(2.5 if i == pipeline_stage else 1.3)
        for i, border in enumerate(borders):
            active = i == current and not final
            chosen = i == winner and final
            border.set_edgecolor(GREEN if chosen else (BLUE if active else LINE))
            border.set_linewidth(3.5 if (active or chosen) else 1.4)
            alpha = np.clip((progress - 0.12 - i * 0.025) / 0.12, 0, 1)
            border.set_alpha(alpha)
            candidate_axes[i].set_visible(alpha > 0.02)
            if progress > 0.42:
                metric_texts[i].set_text(f"CP   {cp[i]} / 3\nCOST {costs[i]:4.1f}")
                metric_texts[i].set_color(GREEN if cp[i] >= 3 else ORANGE)
            else:
                metric_texts[i].set_text("forming\ngraphs…")
        if final:
            headline.set_text("Minimum cost among certified candidates")
            footer.set_text("SELECT PARTICLE 4   •   CP 3/3   •   COST 1.4")
        else:
            headline.set_text(f"Evaluate candidate {current + 1}: graph → GNN → CP → cost")
            footer.set_text("SVGD keeps several alternatives, then retains the best certified trajectory")
        return [headline, footer, *pipeline_cards, *pipeline_arrows, *borders, *metric_texts]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=writer(fps)); plt.close(figure)


def make_video_3(path):
    fps, duration = 30, 14
    figure, axis = plt.subplots(figsize=(16, 9), dpi=120, facecolor=BG)
    axis.set_facecolor(BG); axis.axis("off")
    title_block(figure, "Project workflow: GNN + conformal prediction + SVGD",
                "Separate offline learning from online trajectory optimization and final verification")
    axis.text(0.06, 0.78, "OFFLINE  •  PREPARE THE CERTIFIER", transform=axis.transAxes,
              color=PURPLE, fontsize=13, fontweight="bold")
    axis.text(0.06, 0.45, "ONLINE  •  PLAN AND UPDATE", transform=axis.transAxes,
              color=BLUE, fontsize=13, fontweight="bold")
    axis.text(0.06, 0.13, "OUTPUT  •  VERIFY BEFORE EXECUTION", transform=axis.transAxes,
              color=GREEN, fontsize=13, fontweight="bold")
    specs = [
        ((0.07, 0.61), 0.18, 0.12, "Graph dataset", "binary radius graphs", PURPLE),
        ((0.31, 0.61), 0.18, 0.12, "Exact r labels", "ground truth", PURPLE),
        ((0.55, 0.61), 0.18, 0.12, "Train GATv2", "offline model", PURPLE),
        ((0.79, 0.61), 0.15, 0.12, "Calibrate CP", "held-out graphs", PURPLE),
        ((0.07, 0.29), 0.16, 0.12, "Starts + goals", "robot task", BLUE),
        ((0.28, 0.29), 0.16, 0.12, "Particles", "candidate paths", BLUE),
        ((0.49, 0.29), 0.16, 0.12, "Waypoint graphs", "hard support", TEAL),
        ((0.70, 0.29), 0.11, 0.12, "GNN", "inference", PURPLE),
        ((0.86, 0.29), 0.10, 0.12, "CP", "lower bound", ORANGE),
        ((0.20, 0.00), 0.18, 0.10, "Combined cost", "goal + safety + CP", ORANGE),
        ((0.45, 0.00), 0.18, 0.10, "SPSA + SVGD", "update + diversity", BLUE),
        ((0.70, 0.00), 0.18, 0.10, "Best-ever path", "verify + execute", GREEN),
    ]
    boxes = [card(axis, xy, w, h, title, sub, edge=color) for xy, w, h, title, sub, color in specs]
    connections = [
        ((0.25, 0.67), (0.31, 0.67), PURPLE), ((0.49, 0.67), (0.55, 0.67), PURPLE),
        ((0.73, 0.67), (0.79, 0.67), PURPLE), ((0.23, 0.35), (0.28, 0.35), BLUE),
        ((0.44, 0.35), (0.49, 0.35), BLUE), ((0.65, 0.35), (0.70, 0.35), TEAL),
        ((0.81, 0.35), (0.86, 0.35), PURPLE), ((0.91, 0.29), (0.31, 0.10), ORANGE),
        ((0.38, 0.05), (0.45, 0.05), BLUE), ((0.63, 0.05), (0.70, 0.05), GREEN),
        ((0.54, 0.00), (0.36, 0.29), BLUE), ((0.86, 0.61), (0.755, 0.41), PURPLE),
    ]
    arrows = [arrow(axis, a, b, color=c, alpha=0.18) for a, b, c in connections]
    pulse = Circle((0.07, 0.67), 0.008, transform=axis.transAxes, facecolor=BLUE,
                   edgecolor=WHITE, linewidth=1.0, zorder=10)
    axis.add_patch(pulse)
    narration = figure.text(0.5, 0.035, "", ha="center", color=NAVY, fontsize=13,
                             bbox={"boxstyle": "round,pad=0.6", "facecolor": WHITE,
                                   "edgecolor": LINE, "linewidth": 1.3})
    messages = [
        "Build binary communication graphs and compute exact robustness labels.",
        "Train GATv2 once; calibration is also an offline step.",
        "At runtime, create several candidate multi-robot trajectories.",
        "Every waypoint becomes a hard-support graph for GNN inference.",
        "Conformal prediction turns the GNN output into a lower bound.",
        "Combine the CP shortfall with goal, motion, and safety costs.",
        "SPSA estimates the discrete CP gradient; SVGD updates all particles.",
        "Repeat, retain the best-ever path, then verify before execution.",
    ]
    route = np.array([[0.07, 0.67], [0.40, 0.67], [0.64, 0.67], [0.865, 0.67],
                      [0.15, 0.35], [0.36, 0.35], [0.57, 0.35], [0.755, 0.35],
                      [0.91, 0.35], [0.29, 0.05], [0.54, 0.05], [0.79, 0.05]])
    total_frames = fps * duration

    def update(frame):
        progress = frame / max(1, total_frames - 1)
        scaled = progress * (len(route) - 1)
        left = min(int(scaled), len(route) - 1); right = min(left + 1, len(route) - 1)
        pos = (1 - ease(scaled - left)) * route[left] + ease(scaled - left) * route[right]
        pulse.center = pos
        active = min(len(boxes) - 1, int(progress * len(boxes)))
        for i, patch in enumerate(boxes):
            patch.set_facecolor("#EEF5FF" if i == active else WHITE)
            patch.set_linewidth(2.8 if i == active else 1.3)
            patch.set_alpha(1.0 if i <= active else 0.24)
            for text in patch._texts: text.set_alpha(1.0 if i <= active else 0.24)
        for i, item in enumerate(arrows): item.set_alpha(0.85 if i < active else 0.14)
        narration.set_text(messages[min(len(messages) - 1, int(progress * len(messages)))])
        return [pulse, narration, *boxes, *arrows]

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=writer(fps)); plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_video_1(OUT / "01_gnn_cp_svgd_update_professional_light.mp4")
    print("Saved professional video 1", flush=True)
    make_video_2(OUT / "02_gnn_cp_selection_professional_light.mp4")
    print("Saved professional video 2", flush=True)
    make_video_3(OUT / "03_project_workflow_professional_light.mp4")
    print("Saved professional video 3", flush=True)


if __name__ == "__main__":
    main()
