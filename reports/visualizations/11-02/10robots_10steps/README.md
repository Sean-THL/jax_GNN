# Ten robots, ten-step horizon: trajectory visualization

Generated November 2, 2026 with CUDA planning and Matplotlib rendering.

## Open the visualizations

- [Optimization updates, interactive](optimization_updates.html): move the slider through actual SVGD populations.
- [Optimization updates, video](optimization_updates.mp4)
- [Selected motion, interactive](selected_motion.html): play the chosen trajectories through their sampled waypoints.
- [Selected motion, video](selected_motion.mp4)

Ten future waypoints at dt=0.2 s cover 2 seconds of planned motion. There are ten separate optimization updates, plus the initial population. Each method receives identical starts, goals, 16 initial particles, and seed 42.

Optimization panels follow one fixed particle index per method, corresponding to the selected candidate. A selected candidate may come from an earlier population. Initial trajectories are dashed, updated trajectories are colored, and other candidates are faint gray. Each optimization panel has its own fixed scale spanning all its populations, so large excursions remain visible. Playback panels share a scale based only on the selected paths.

| Method | Selected update | Passing / evaluated candidates | Selected minimum CP bound | Solving time (s) |
| --- | ---: | ---: | ---: | ---: |
| simple | 0 | 176 / 176 | 5 | 3.402 |
| soft_degree | 10 | 176 / 176 | 5 | 1.500 |
| gnn_cp_spsa | 0 | 17 / 176 | 5 | 6.565 |
| gnn_cp_autodiff | 7 | 67 / 176 | 5 | 4.320 |

All CP bounds above concern the selected sampled trajectory. Inter-waypoint motion is not certified. Collision separation and speed remain soft penalties. This is planned waypoint playback, not a robot controller or physics simulation. Single-run solving times include compilation within planning and exclude model loading/calibration, post-run evaluation, and visualization rendering.

## Reproduce

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --robots 10 --steps 10 --iterations 10
```

Re-render the saved arrays without rerunning planning:

```bash
.venv/bin/python -m planning.visualize_trajectory_updates --load
```

Add `--show` to open desktop Matplotlib windows. Open either HTML file in a browser for playback and frame-slider controls.

`planner_snapshots.npz` contains positions with axes `[population, particle, waypoint, robot, xy]`, population costs and minimum bounds with axes `[population, particle]`, and selected paths. `metrics.json` contains per-step selected CP bounds and per-method settings.

## Numerical stability fix

The first recording exposed underflow in attention normalization when zero-weight edges had dominant attention logits. The weighted GATv2 layer now applies the edge prior in log space before softmax. This preserves the intended normalization mathematically, keeps silent edges inactive, and avoids 0/0 attention. Binary-edge compatibility and finite-gradient regression checks pass. All regenerated candidate positions are finite. The data are not from the earlier unstable recording.

## Visualization references

The [distributed-MBD project](https://distributed-mbd.github.io/dmbd.github.io/) presents refinement separately from optimized-trajectory playback. The [DPCBF project](https://www.taekyung.me/dpcbf) presents method comparisons and motion diagnostics. Those presentation ideas informed these views; their planning/control algorithms were not copied.
