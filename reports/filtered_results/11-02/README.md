# Double-integrator validation — November 2, 2026

10 robots, 4 particles, 3 SVGD updates, 200 steps, dt=0.05 s.
10 m formation transfer, R=3 m, hard acceleration norm bound 1 m/s².

| Method | CP success | Exact r | Mean goal error (m) | Max acceleration (m/s²) | Solving time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| simple | YES | 5 | 0.0000 | 0.5970 | 4.683 |
| soft_degree | YES | 5 | 0.0455 | 0.6016 | 2.602 |
| gnn_cp_spsa | YES | 5 | 0.0455 | 0.6016 | 8.134 |
| gnn_cp_autodiff | YES | 5 | 0.0728 | 0.6044 | 5.327 |

Independent checks using the saved controls/states confirm position residuals below 2e-6 m and velocity residuals below 1e-6 m/s (float32 rounding), and acceleration norms below 1 m/s². See `double_integrator_validation.json`.

This is an unobstructed formation transfer. All four methods start from the same physically rolled-out control seeds. This run establishes implementation validity, not a statistical comparison of planner quality. Speed, collision separation, terminal position, and terminal velocity remain soft objectives. CP success refers to waypoint predictions; exact r is reported separately.

The 25 m and 50 m scenarios are configured but have not been benchmarked on CUDA in this validation.
