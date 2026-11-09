# Red robots with soft goals — 50 steps

This is the matching experiment without fixed endpoint constraints.
`fixed_goal` is false. Goals contribute a cost but the endpoint can move.
The video is `red_robots_50steps.mp4`.

Same scenario, 10 robots, 50 waypoints, dt 0.2 s, 16 particles, 10 SVGD updates,
seed 42, and GNN–CP–autodiff as the fixed-goal comparison. Red numbered robots,
green numbered goals, and dashed gray links at opacity 0.5 are unchanged.
The video interpolates between the actual selected waypoints; it does not
append a segment to the goal.

- Mean final goal distance: 0.022237 scenario units.
- Maximum final goal distance: 0.044845 scenario units.
- Minimum CP bound over selected future waypoints: 5.
- Minimum exact robustness at those waypoints: 5.

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --robots 10 --steps 50 --iterations 10 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-02/10robots_50steps_soft_goal

.venv/bin/python -m planning.visualize_red_robots \
  --input reports/visualizations/11-02/10robots_50steps_soft_goal
```

Exact robustness and minimum degree are checked on displayed graphs. CP bounds
refer to stored waypoints only. Frame sampling does not certify every instant
of continuous motion; speed and collision separation remain soft costs.
