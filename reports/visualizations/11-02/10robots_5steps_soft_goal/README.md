# Ten robots, five steps, soft goals

Single GNN–CP–autodiff video: `red_robots_5steps.mp4`.

Same scenario and display style as the 50-step soft-goal experiment:
10 robots, 16 particles, 10 optimization updates, seed 42, dt=0.2 s.
The horizon is now five future waypoints, covering 1 second of planned motion.
Endpoints are free; goals contribute a cost and are not fixed constraints.

Red numbered robots, green numbered goals, and 50%-opacity dashed gray
communication links. Exact robustness and minimum degree refer to displayed
graphs; CP values refer to stored waypoints. Playback is half speed with
straight-line interpolation between waypoints. The video does not append a
segment to force arrival at the goals.

Mean final goal error: 0.090656 scenario units.
Maximum final goal error: 0.137039 scenario units.

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --robots 10 --steps 5 --iterations 10 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-02/10robots_5steps_soft_goal

.venv/bin/python -m planning.visualize_red_robots \
  --input reports/visualizations/11-02/10robots_5steps_soft_goal
```
