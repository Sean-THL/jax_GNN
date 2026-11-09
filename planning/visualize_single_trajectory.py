"""Render one video of the saved GNN-CP-autodiff selected trajectory."""
import os
from pathlib import Path
import argparse
import json
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-planner-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import numpy as np
from planning.run_experiment import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "reports/visualizations/11-02/10robots_10steps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "gnn_cp_autodiff_10robots_motion.mp4"
    metadata = json.loads((args.input / "metrics.json").read_text())
    info = metadata["methods"]["gnn_cp_autodiff"]
    with np.load(args.input / "planner_snapshots.npz") as data:
        if "gnn_cp_autodiff__selected" not in data:
            raise ValueError("No CP-feasible trajectory selected")
        start, goal = data["start"], data["goal"]
        path = np.concatenate((start[None], data["gnn_cp_autodiff__selected"]))
    steps, robots = len(path) - 1, len(start)
    dt = info["config"]["sampling_time"]
    colors = plt.colormaps["tab10"](np.arange(robots))
    points = np.concatenate((path.reshape(-1, 2), goal))
    center = (points.min(0) + points.max(0)) / 2
    half = np.ptp(points, axis=0).max() / 2 + 0.055
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor("#f6f8fc")
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=0.81)
    fig.suptitle(f"GNN–CP–autodiff · {robots} robots", fontsize=21, fontweight="bold", color="#172b46", y=0.96)
    fig.text(0.5, 0.91, f"Selected trajectory · {steps} waypoints · {steps * dt:.1f} s planned motion", ha="center", fontsize=12, color="#52637a")
    clock = fig.text(0.5, 0.86, "", ha="center", fontsize=12, color="#126782")
    ax.set_aspect("equal")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xlabel("x (scenario units)"); ax.set_ylabel("y (scenario units)")
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.scatter(*goal.T, marker="*", s=200, c=colors, edgecolors="#314158", linewidths=0.7, zorder=3)
    for r in range(robots):
        ax.plot(*path[:, r].T, color=colors[r], lw=1, alpha=0.15, ls="--")
    trails = [ax.plot([], [], color=colors[r], lw=2, alpha=0.8)[0] for r in range(robots)]
    dots = ax.scatter(*start.T, s=180, c=colors, edgecolors="white", linewidths=1.5, zorder=5)
    labels = [ax.text(*start[r], str(r + 1), color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6) for r in range(robots)]
    fig.legend(handles=[Line2D([], [], marker="o", color="#52637a", ls="none", label="Robot (numbered)"), Line2D([], [], marker="*", markersize=12, color="#52637a", ls="none", label="Assigned goal"), Line2D([], [], color="#52637a", label="Robot trail")], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.065), frameon=False)
    fig.text(0.5, 0.035, "¼-speed playback · straight-line interpolation between planned waypoints", ha="center", fontsize=9, color="#52637a")
    fps = 24
    movement_frames = max(1, round(steps * dt * 4 * fps))
    hold = fps
    def update(frame):
        progress = np.clip((frame - hold) / movement_frames, 0., 1.) * steps
        step = min(int(progress), steps - 1)
        fraction = progress - step
        position = (1 - fraction) * path[step] + fraction * path[step + 1]
        dots.set_offsets(position)
        for r, (trail, label) in enumerate(zip(trails, labels)):
            past = np.concatenate((path[:step + 1, r], position[r, None]))
            trail.set_data(*past.T)
            label.set_position(position[r])
        clock.set_text(f"Planned time: {progress * dt:.2f} / {steps * dt:.2f} s")
        return []
    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=movement_frames + 2 * hold + 1, interval=1000 / fps, blit=False)
    animation.save(output, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    update(movement_frames + hold)
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
