# Differentiable GNN planner — November 1, 2026

`gnn_cp_autodiff` uses equation-(7) weights and the frozen GNN's expected class
minus the one-sided calibration correction as a continuous optimization
surrogate. Automatic differentiation reaches robot positions without SPSA.
The continuous value is not itself a certified CP bound. Hard-graph CP bounds
still determine final acceptance for all saved candidates.

## CUDA validation

50 robots, 16 particles, 10 waypoints, five SVGD iterations, seed 42, dt 0.2 s.

| Planner | CP success | Minimum hard CP bound | Minimum hard degree | Solving time (s) |
| --- | --- | ---: | ---: | ---: |
| gnn_cp_autodiff | YES | 11 | 10 | 7.872 |

Solving time includes optimization, final filtering, and compilation incurred
inside the planner call. Model loading/calibration and post-run evaluation are
excluded. This is one validation run, not a speed or success-rate comparison.
Exact robustness is not computed at 50 robots. Minimum robot separation was
0.004, below the configured 0.03; collision separation remains a soft penalty.

The JSON records the selected trajectory, hard CP bounds per waypoint,
per-method configuration, optimization cost type, candidate counts, and history.
The binary-trained model's guidance on fractional edge weights still requires
validation, potentially retraining. Existing calibration and adaptive-selection
limitations remain relevant to interpreting final acceptance.
