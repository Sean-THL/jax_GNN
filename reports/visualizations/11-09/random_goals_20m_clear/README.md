# Readable 20 m random-goal network video

Open `network_closeup_20m.mp4`. Preview: `network_closeup_preview.png`.

The main view follows the team with a fixed 4 x 4 m camera, keeping numbered
robots and communication links readable. Axes are world coordinates. A small
route map shows the full trip, starts, independent goals and the camera
footprint. The graph plot reveals exact robustness and minimum degree up to
the current time. CP in the statistics refers to the last stored waypoint.
No CBF values or malicious robots are simulated or added from the references.

10 robots, independent random endpoint formations about 19–22 m apart,
R=3 m, double integrator with ||u|| <=1 m/s², dt=0.05 s, 320 steps/16 seconds,
8 particles, five SVGD updates. Scenario:
`planning/examples/double_integrator_random_goals_20m.json`.
Endpoint separation sampling is 0.45 m. Goals remain soft costs. The selected
trajectory has minimum separation about 0.025 m, so it is not collision-safe.
Robustness and degree are constant in this recorded example because every
sampled graph is complete. Their curves are not fabricated to show variation.

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --scenario planning/examples/double_integrator_random_goals_20m.json \
  --robots 10 --particles 8 --iterations 5 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-09/random_goals_20m_clear
.venv/bin/python -m planning.visualize_network_closeup \
  --input reports/visualizations/11-09/random_goals_20m_clear
```

Playback uses the constant-acceleration equations between states. The scene
is not a physics/controller simulation. Frame sampling is not a certificate
of full continuous-time safety.
