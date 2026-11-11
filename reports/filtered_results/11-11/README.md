# Joint CP and collision filter validation — November 11, 2026

The saved 50-robot r=21 geometry was rerun with the same 15 particles,
320 steps, dt=0.05 s, five SVGD updates and seed 42. GNN–CP–autodiff remains
the default cost. Goal arrival remains soft. Robot radius is 0.2 m and the
hard acceptance threshold is 0.45 m between centers (0.05 m surface gap).

| Result | Count |
| --- | ---: |
| Evaluated candidates | 90 |
| CP-passing candidates | 15 |
| Rejected for separation violations | 15 |
| Passing both checks | 0 |

Status: `no_feasible_trajectory`; returned waypoints: null. The optimizer did
not find a valid plan under both acceptance rules, so no rejected fallback
is exposed as a successful trajectory. This result does not prove that a
safe solution is impossible. Optimization cost terms are unchanged.

The deterministic separation check includes the initial positions and all
constant-acceleration intervals. It first screens waypoint violations, then
checks interval endpoints and stationary points of squared pair distance.
CP bounds are still evaluated on hard graphs at future waypoints, with their
existing calibration limitations. The geometric check does not extend CP
coverage to continuous time; it is an independent modeled-motion constraint.

The test suite passed all 62 tests, including clear endpoints with an
intermediate collision, rejection of initial overlap, and selection of a
more expensive collision-free candidate over a cheaper colliding one.

From `jax_GNN/`, reproduce:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario reports/filtered_results/11-10/joint_filter_scenario.json \
  --robots 50 --particles 15 --steps 320 --iterations 5 \
  --methods gnn_cp_autodiff \
  --output reports/filtered_results/11-11/50robots_r21_joint_filter.json
```

Historical videos and the collision-weight sweep used CP-only acceptance;
they are preserved as historical results, and do not pass the new filter.
