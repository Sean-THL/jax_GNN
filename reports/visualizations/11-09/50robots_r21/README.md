# Fifty robots, robustness target 21

Video: `network_closeup_50robots_r21.mp4`.
Preview: `network_closeup_preview.png`.

This is a CP-accepted trajectory for 50 robots with required robustness 21.
Minimum predicted CP lower bound: 22.
Minimum hard degree: 31.
Exact robustness is not computed for 50 robots. CP acceptance is not an exact
robustness certificate or a guarantee of continuous-time collision safety.

The close-up follows the formation with a fixed scale spanning the selected
paths; the small map shows the full start-to-goal route. Red numbered markers
are robots, green numbered markers are assigned goals, and gray dashed edges
show communication within R=3 m. The chart plots sampled CP bounds and minimum
degree. It does not claim to plot exact robustness.

Settings: 50 robots from the existing 100-robot pool; 8 particles; 5 SVGD
updates; 320 control steps; dt=0.05 s; 16-second horizon; F=10; acceleration
norm <=1 m/s². Goals and collision separation remain soft objectives.
Maximum selected acceleration: 0.902071 m/s².
Mean terminal goal error: 0.00000872 m.
Minimum separation: 0.002951 m, below the 0.45 m target.
The motion therefore is not collision-safe.

Selected population: 0 (0 means the initial candidate).
Later optimization updates did not produce a cheaper CP-accepted candidate.
The renderer follows the double-integrator equations using the saved controls
and states; it does not force a terminal segment to the goals.

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --scenario planning/examples/double_integrator_100robots.json \
  --robots 50 --particles 8 --iterations 5 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-09/50robots_r21
.venv/bin/python -m planning.visualize_network_closeup \
  --input reports/visualizations/11-09/50robots_r21
```

## Physical-radius update

`network_closeup_50robots_r21_radius0.2m.mp4` draws the robot footprints as
world-coordinate disks of radius 0.2 m. The original video used symbolic
markers. The new disk video reuses the same stored trajectory and does not
change its optimization or endpoints. It flags negative surface gaps. The
0.45 m center-distance target corresponds to 0.05 m surface clearance.
Overview-map dots and goal points are still symbolic markers.
