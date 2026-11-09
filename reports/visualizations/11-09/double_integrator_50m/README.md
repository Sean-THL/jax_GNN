# Fifty-metre double-integrator visualization

Open `double_integrator_explained.mp4`.

Same 10-robot GNN–CP–autodiff setup as the 10 m video, using
`planning/examples/double_integrator_largest.json`. Goals are translated 50 m
along x. The reference workspace is 60 x 10 m. It is not enforced as walls.

Sampling time: 0.05 s. Horizon: 640 steps / 32 seconds. Hard acceleration norm
bound: 1 m/s². Communication radius: 3 m. Initial velocity: zero. The speed
cost uses 3 m/s; goals and terminal velocity remain soft costs.

The video has red numbered robots, green numbered goals, dashed gray
communication links at 50% opacity, orange acceleration arrows, and one blue
3 m radius circle around robot 1. All robots use the same radius. Start markers
remain faintly visible. Header values show time, step, acceleration, graph
robustness and minimum degree. CP values refer to stored waypoints. Playback
is half speed with constant-acceleration motion between waypoints.

Recorded with 8 particles, 5 SVGD updates and seed 42. Selected candidate came
from population 0 (0 means initialization). Subsequent
updates did not improve on its accepted optimization cost.

- Maximum selected acceleration: 0.292512 m/s².
- Maximum selected speed: 2.343752 m/s.
- Mean goal error: 0.00002899 m.
- Minimum CP lower bound over future waypoints: 5.
- Minimum exact robustness over future waypoints: 5.

The nearly exact endpoint comes from the dynamically feasible initial seed,
not a hard goal constraint or a segment appended by the renderer. Full
continuous-time safety is not certified by frame sampling.

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --scenario planning/examples/double_integrator_largest.json \
  --robots 10 --particles 8 --iterations 5 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-09/double_integrator_50m

.venv/bin/python -m planning.visualize_red_robots \
  --input reports/visualizations/11-09/double_integrator_50m --show-inputs
```
