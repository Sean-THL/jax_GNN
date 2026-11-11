# Extended collision-weight trials: 50 robots, r = 21

Historical acceptance policy: CP only. New planning runs enable the joint CP/separation filter by default and reject these colliding plans. The best-tested scenario file preserves costs and geometry; rerunning it with new defaults may correctly return no feasible trajectory.

**No tested weight eliminates collisions.** 18 tested weights cover zero to one million. The lowest tested weight tied for the fewest violating pairs is **6900**. It does not improve the best count from the earlier search; it selects the same initial trajectory at a smaller weight.

The best group has 310 pairs violating the 0.45 m center-distance target, versus 328 at weight 10. With the user-defined body radius of 0.2 m, the best group has 270 pairs with actual disk overlap, versus 287 at the baseline. Its closest center distance is 0.002784 m and its minimum surface gap is -0.397216 m. These remain unsafe trajectories.

Weights tied at the best safety-violation count: 6900, 7500, 10000, 15000, 30000.

## Geometry and success definitions

- Robot body radius: 0.2 m. Body diameter: 0.4 m.
- Physical body overlap: center distance below 0.4 m.
- Safety-distance violation: center distance below 0.45 m; this includes a 0.05 m surface gap.
- CP success: accepted predicted hard-graph lower bounds at least 21, with the existing degree/feasibility reporting checks.
- CP success does not certify collision safety or exact graph robustness.
- The body radius was specified after the trials. Body-overlap metrics were recomputed on the saved trajectories; the optimization separation target stays 0.45 m.

A violating pair is a unique robot pair that violates at least once. Pair-intervals count individual robot-pair/control intervals with a violation. For each constant-acceleration interval, relative position is quadratic. Minimum separation is evaluated at endpoints and real stationary points obtained from a cubic. These checks include the motion between stored waypoints. They do not simulate contact forces or impacts.

## Fixed experiment settings

50 robots, GNN–CP–autodiff, 15 particles, 320 steps, dt=0.05 s (16 seconds), five optimization updates, seed 42, the same saved starts/goals and initial control noise 0.1. R=3 m, r=21, hard acceleration norm <=1 m/s². Goals and collision separation remain soft objectives. Only collision weight changes. Source: `reports/filtered_results/11-09/100robots_r21_four_methods.json`; its historical filename says 100 but its saved parameters specify 50 robots.

## Results

| Weight | CP success | Min CP | Safety-violating pairs (<0.45 m) | Body-overlap pairs (<0.4 m) | Closest centers (m) | Mean goal error (m) | Solve time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | YES | 22 | 328 | 287 | 0.002951 | 0.00000872 | 47.834 |
| 3 | YES | 22 | 328 | 287 | 0.002951 | 0.00000872 | 47.193 |
| 10 | YES | 22 | 328 | 287 | 0.002951 | 0.00000872 | 43.343 |
| 30 | YES | 22 | 328 | 287 | 0.002951 | 0.00000872 | 44.006 |
| 100 | YES | 22 | 328 | 287 | 0.002951 | 0.00000872 | 43.719 |
| 300 | YES | 22 | 316 | 283 | 0.000558 | 0.00001039 | 42.358 |
| 1000 | YES | 22 | 316 | 283 | 0.000558 | 0.00001039 | 41.118 |
| 2000 | YES | 22 | 316 | 283 | 0.000558 | 0.00001039 | 44.427 |
| 3000 | YES | 22 | 316 | 283 | 0.000558 | 0.00001039 | 39.100 |
| 5000 | YES | 22 | 316 | 283 | 0.000558 | 0.00001039 | 38.643 |
| 6900 | YES | 22 | 310 | 270 | 0.002784 | 0.00001190 | 42.145 |
| 7500 | YES | 22 | 310 | 270 | 0.002784 | 0.00001190 | 45.275 |
| 10000 | YES | 22 | 310 | 270 | 0.002784 | 0.00001190 | 42.793 |
| 15000 | YES | 22 | 310 | 270 | 0.002784 | 0.00001190 | 36.589 |
| 30000 | YES | 22 | 310 | 270 | 0.002784 | 0.00001190 | 35.058 |
| 31200 | YES | 22 | 331 | 279 | 0.001619 | 0.00001166 | 38.632 |
| 100000 | YES | 22 | 331 | 279 | 0.001619 | 0.00001166 | 38.069 |
| 1e+06 | YES | 22 | 331 | 279 | 0.001619 | 0.00001166 | 33.960 |

Weight 10 uses the existing saved baseline; other weights are new CUDA trials. Timings include planner compilation, exclude setup and collision analysis, and are not a controlled runtime benchmark.

## Why further weight tuning plateaued

All selected plans came from initialization. Every run accepted exactly 15 out of 90 evaluated candidates. `initial_population_cp.json` confirms that all 15 initial candidates pass CP with bound 22, so none of the 75 updated candidates passed in any trial. The weight changes which initial candidate is selected rather than producing a better CP-accepted updated plan.

For fixed initial trajectories, each cost is affine in collision weight: `cost_i(w) = base_i + w * collision_penalty_i`. `initial_weight_frontier.json` identifies every initial-candidate selection region. Its predicted indices and costs match the saved trials. The bounds below are approximate floating-point transition values, not exact guarantees for the full optimizer.

| Weight range (approximately) | Selected initial particle | Safety-violating pairs |
| --- | ---: | ---: |
| 0.0 to 277.9 | 1 | 328 |
| 277.9 to 6889.2 | 12 | 316 |
| 6889.2 to 31134.3 | 10 | 310 |
| 31134.3 to infinity | 4 | 331 |

No initial selection region is collision-free. This establishes the limitation of ranking these fixed initial candidates, not impossibility of finding a collision-free optimized trajectory at an untested weight or with other optimization settings. The squared-depth penalty also does not minimize collision count directly, explaining why very large weights increase the count.

## Reproduction

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight --legacy-cp-only \
  --weights 0 3 30 100 300 1000 2000 3000 5000 6900 7500 10000 15000 30000 31200 100000 1000000 \
  --robot-radius 0.2
```

The baseline weight 10 is imported automatically. Existing trial files are reused. The default scenario collision weight remains 10. To reproduce the best tested setting without changing it:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario reports/trials/11-09/collision_weights_r21/best_tested_scenario.json \
  --robots 50 --particles 15 --steps 320 --iterations 5 \
  --output reports/trials/11-09/collision_weights_r21/best_rerun.json
```

`summary.csv` and `summary.json` contain both body-overlap and safety-distance metrics. The main CLI also supports `--collision-weight 6900`. This is an experimental best count, not a safe recommended setting.
