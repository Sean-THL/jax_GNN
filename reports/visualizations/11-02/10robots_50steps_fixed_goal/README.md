# Red robots, green goals: 50-step GNN–CP–autodiff playback

The single video is `red_robots_50steps.mp4`.

- Ten red, numbered robot dots; green, numbered assigned-goal dots.
- Dashed gray lines at opacity 0.5 show links within the communication radius.
- Fifty future waypoints at dt=0.2 s give a 10-second horizon.
- Playback is at half speed with linear interpolation between waypoints.
- Exact r-robustness and minimum binary degree are recomputed on each displayed graph. CP bounds are explicitly labeled as sampled-waypoint values.
- Start and final goal positions are fixed for this run. Only intermediate positions can change. Final endpoint error is zero.

The run uses 16 particles, ten optimization updates, seed 42, smooth equation-(7) guidance, and the hard-graph CP acceptance filter with target 5. Fixed goals are enabled via the new opt-in `fixed_goal` setting. Earlier experiments retain their original soft endpoint behavior.

From `jax_GNN/`, reproduce:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --robots 10 --steps 50 --iterations 10 --methods gnn_cp_autodiff \
  --fixed-goal --record-only \
  --output reports/visualizations/11-02/10robots_50steps_fixed_goal

.venv/bin/python -m planning.visualize_red_robots
```

`graph_metrics_by_waypoint.json` records exact robustness, minimum degree, and CP bounds at each stored waypoint. `playback_graph_summary.json` summarizes the rendered graphs and endpoint error. This is playback of a planned trajectory, not a physics simulation. Finite frame sampling does not certify every instant of continuous motion. Collision separation and speed remain soft penalties.
