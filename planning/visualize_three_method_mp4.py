"""Animate the three-method planner comparison from a saved JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np


METHODS = ("simple", "soft_degree", "gnn_cp")
METHOD_TITLES = {
    "simple": "Simple objective",
    "soft_degree": "Soft-degree objective",
    "gnn_cp": "GNN + conformal objective",
}


def smooth_weights(positions, radius, sigma):
    delta = positions[:, None, :] - positions[None, :, :]
    distance_squared = np.sum(delta * delta, axis=-1)
    gap = np.maximum(radius * radius - distance_squared, 0.0)
    weights = np.expm1(gap * gap / sigma)
    np.fill_diagonal(weights, 0.0)
    return weights, np.sqrt(distance_squared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds-per-step", type=float, default=2.0)
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    config = report["config"]
    start = np.asarray(report["start"], dtype=float)
    goal = np.asarray(report["goal"], dtype=float)
    paths = {
        name: np.concatenate(
            (start[None], np.asarray(report["results"][name]["waypoints"], dtype=float)),
            axis=0,
        )
        for name in METHODS
    }
    steps = report["parameters"]["steps"]
    if steps != 3:
        raise ValueError("this explanatory animation expects exactly three planning steps")

    radius = float(config["communication_radius"])
    sigma = config["connectivity_sigma"]
    sigma = radius**4 / np.log(2.0) if sigma is None else float(sigma)
    required = int(config["required_robustness"])
    fps = args.fps
    segment_frames = max(2, round(args.seconds_per_step * fps))
    total_frames = steps * segment_frames + fps

    all_points = np.concatenate([start, goal, *(path.reshape(-1, 2) for path in paths.values())])
    lower = all_points.min(axis=0) - 0.08
    upper = all_points.max(axis=0) + 0.08
    span = max(upper - lower)
    center = 0.5 * (lower + upper)
    lower, upper = center - 0.5 * span, center + 0.5 * span

    colors = plt.colormaps["tab10"](np.arange(len(start)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120, sharex=True, sharey=True)
    figure.patch.set_facecolor("#07111f")
    figure.subplots_adjust(left=0.045, right=0.985, bottom=0.12, top=0.82, wspace=0.08)
    figure.suptitle(
        "How 10 robots move over three planning steps",
        color="white", fontsize=24, fontweight="bold", y=0.965,
    )
    figure.text(
        0.5, 0.915,
        "Smooth connectivity • brighter links are stronger • dashed circle shows communication radius R = 0.30",
        ha="center", color="#b8c7db", fontsize=13,
    )
    clock = figure.text(0.5, 0.865, "", ha="center", color="#53d8fb", fontsize=16, fontweight="bold")
    figure.text(
        0.5, 0.045,
        "● robot position     ☆ assigned goal     ··· traveled path     — smooth communication link",
        ha="center", color="#cdd8e7", fontsize=12,
    )
    figure.text(
        0.5, 0.018,
        "Success requires CP lower bound ≥ 5 and hard minimum degree ≥ 5; exact r is shown separately.",
        ha="center", color="#8fa4bd", fontsize=10,
    )

    artists = {}
    for axis, method in zip(axes, METHODS):
        axis.set_facecolor("#0d1b2b")
        axis.set_aspect("equal")
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.grid(color="#29445f", alpha=0.24, linewidth=0.7)
        axis.tick_params(colors="#8297ad", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#35506b")
        axis.set_title(METHOD_TITLES[method], color="white", fontsize=15, fontweight="bold", pad=14)
        axis.scatter(start[:, 0], start[:, 1], s=28, color="#78899d", alpha=0.55, zorder=2)
        axis.scatter(
            goal[:, 0], goal[:, 1], s=120, marker="*", facecolors="none",
            edgecolors="#ffd166", linewidths=1.5, zorder=3,
        )
        edges = LineCollection([], zorder=1)
        axis.add_collection(edges)
        trails = [axis.plot([], [], color=colors[i], alpha=0.82, linewidth=2.0, zorder=2)[0]
                  for i in range(len(start))]
        robots = axis.scatter(start[:, 0], start[:, 1], s=105, c=colors, edgecolors="white",
                              linewidths=1.1, zorder=5)
        labels = [axis.text(*start[i], str(i + 1), ha="center", va="center", color="#07111f",
                            fontsize=7.5, fontweight="bold", zorder=6) for i in range(len(start))]
        range_circle = Circle(start[0], radius, fill=False, linestyle=(0, (4, 4)),
                              linewidth=1.1, edgecolor="#53d8fb", alpha=0.65, zorder=1)
        axis.add_patch(range_circle)
        metrics = axis.text(
            0.03, 0.03, "", transform=axis.transAxes, ha="left", va="bottom",
            fontsize=10, color="white",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#07111f", "edgecolor": "#35506b", "alpha": 0.92},
            zorder=8,
        )
        artists[method] = (edges, trails, robots, labels, range_circle, metrics)

    def frame_state(frame):
        if frame >= steps * segment_frames:
            return steps - 1, steps, 1.0
        segment = frame // segment_frames
        fraction = (frame % segment_frames) / (segment_frames - 1)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        return segment, segment + 1, eased

    def update(frame):
        left, right, fraction = frame_state(frame)
        clock.set_text(f"Planning step {right} of {steps}     •     t = {right * config['sampling_time']:.1f} s")
        changed = [clock]
        for method in METHODS:
            result = report["results"][method]
            path = paths[method]
            current = (1.0 - fraction) * path[left] + fraction * path[right]
            weights, distances = smooth_weights(current, radius, sigma)
            segments, edge_colors, widths = [], [], []
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if distances[i, j] <= radius:
                        weight = float(np.clip(weights[i, j], 0.0, 1.0))
                        segments.append([current[i], current[j]])
                        edge_colors.append((0.33, 0.85, 0.98, 0.10 + 0.62 * weight))
                        widths.append(0.5 + 2.2 * weight)
            edges, trails, robots, labels, range_circle, metrics = artists[method]
            edges.set_segments(segments)
            edges.set_colors(edge_colors)
            edges.set_linewidths(widths)
            robots.set_offsets(current)
            range_circle.center = current[0]
            for i, (trail, label) in enumerate(zip(trails, labels)):
                completed = np.concatenate((path[:right, i], current[i][None]), axis=0)
                trail.set_data(completed[:, 0], completed[:, 1])
                label.set_position(current[i])
            hard_degree = ((distances <= radius) & ~np.eye(len(current), dtype=bool)).sum(axis=1)
            step_index = min(right - 1, len(result["lower_bounds_by_step"]) - 1)
            cp_bound = result["lower_bounds_by_step"][step_index]
            exact_values = result["exact_robustness_by_step"]
            exact = exact_values[step_index] if exact_values[step_index] is not None else "n/a"
            certificate = "PASS" if cp_bound >= required else "conservative / below target"
            metrics.set_text(
                f"Current CP bound: {cp_bound} / {required}  ({certificate})\n"
                f"Current exact r: {exact}\n"
                f"Current hard min degree: {int(hard_degree.min())}\n"
                f"Final goal error: {result['goal_mean_distance']:.3f}"
            )
            metrics.get_bbox_patch().set_edgecolor("#4fd1a1" if cp_bound >= required else "#f2a65a")
            changed.extend([edges, robots, range_circle, metrics, *trails, *labels])
        return changed

    animation = FuncAnimation(figure, update, frames=total_frames, interval=1000 / fps, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=5000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    animation.save(args.output, writer=writer)
    plt.close(figure)
    print(f"Saved animation: {args.output}")


if __name__ == "__main__":
    main()
