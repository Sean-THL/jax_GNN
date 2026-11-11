# Collision-weight trials: 50 robots, r = 21

No tested collision weight achieved zero separation violations. Four weights (7,500, 10,000, 15,000 and 30,000) tied for the best collision count. The lowest tested weight in that group is **7,500**. This is a relative improvement, not a collision-safe recommendation.

The baseline has 328 violating pairs; the best group has 310, a 5.49% reduction. Their closest separation is 0.002784 m, which is slightly worse than the baseline's 0.002951 m and far below the required 0.45 m. The largest tested weight, 100,000, worsened the count to 331. Increasing the squared-shortfall penalty does not monotonically reduce the number of violating pairs.

## Controlled setup

Source: `reports/filtered_results/11-09/100robots_r21_four_methods.json`. Despite its historical filename, its saved parameters specify **50 robots**. The trials use GNN–CP–autodiff, 15 particles, 320 steps, dt=0.05 s (16 s horizon), five SVGD updates, seed 42, initial acceleration noise 0.1, R=3 m, r=21, and the same saved start/goal positions. Acceleration norm is hard bounded by 1 m/s²; soft goals remain enabled. Only `collision_weight` changes. The default scenario is not modified.

The collision threshold is the configured center-to-center separation of 0.45 m. A violating pair is a unique robot pair that goes below this threshold at any time. A violating pair-interval is a pair/control interval with such a violation. These are separation violations, not separately simulated physical impacts. For each interval, relative position is quadratic in time; minima are checked at interval endpoints and real stationary points obtained from a cubic. This checks between stored waypoints, not just animation frames. Three analytic regression tests cover linear and accelerated crossings and a clear parallel-motion case.

## Results

| Weight | CP success | Minimum CP bound | Violating pairs | Violating pair-intervals | Closest distance (m) | Mean goal error (m) | Selected population | Solving time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | YES | 22 | 328 | 20882 | 0.002951 | 0.00000872 | 0 | 47.193 |
| 10 | YES | 22 | 328 | 20882 | 0.002951 | 0.00000872 | 0 | 43.343 |
| 30 | YES | 22 | 328 | 20882 | 0.002951 | 0.00000872 | 0 | 44.006 |
| 100 | YES | 22 | 328 | 20882 | 0.002951 | 0.00000872 | 0 | 43.719 |
| 300 | YES | 22 | 316 | 20186 | 0.000558 | 0.00001039 | 0 | 42.358 |
| 1000 | YES | 22 | 316 | 20186 | 0.000558 | 0.00001039 | 0 | 41.118 |
| 2000 | YES | 22 | 316 | 20186 | 0.000558 | 0.00001039 | 0 | 44.427 |
| 3000 | YES | 22 | 316 | 20186 | 0.000558 | 0.00001039 | 0 | 39.100 |
| 5000 | YES | 22 | 316 | 20186 | 0.000558 | 0.00001039 | 0 | 38.643 |
| 7500 | YES | 22 | 310 | 19554 | 0.002784 | 0.00001190 | 0 | 45.275 |
| 10000 | YES | 22 | 310 | 19554 | 0.002784 | 0.00001190 | 0 | 42.793 |
| 15000 | YES | 22 | 310 | 19554 | 0.002784 | 0.00001190 | 0 | 36.589 |
| 30000 | YES | 22 | 310 | 19554 | 0.002784 | 0.00001190 | 0 | 35.058 |
| 100000 | YES | 22 | 331 | 20389 | 0.001619 | 0.00001166 | 0 | 38.069 |

Weight 10 reuses the existing saved baseline; other rows are fresh CUDA trials. Timings include compilation within each planner call and are not a controlled speed benchmark. Model loading/calibration and post-run collision analysis are excluded.

## Why weight tuning did not eliminate collisions

All tested weights selected a candidate from population 0 (initialization). Every report records 15 CP-passing candidates out of 90 examined. The separate `initial_population_cp.json` check confirms that all 15 initial candidates pass, so none of the 75 updated candidates passed in any tested trial. The iteration-best candidates have bounds mostly 0, far below 21. Thus tuning changed which initial candidate survived final cost ranking; it did not produce an improved CP-accepted updated trajectory.

The best tested weight was chosen by fewest violating pairs, then fewest violating pair-intervals, then greatest minimum separation, then lowest weight. Other objectives can rank these unsafe trajectories differently. No claim is made that zero collisions is impossible, or that this finite weight search found a global optimum. Further work on update behavior or explicit separation constraints is needed before claiming collision-free operation.

## Reproduce or use the best tested trial

From `jax_GNN/`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight \
  --weights 3 30 100 300 1000 2000 3000 5000 7500 10000 15000 30000 100000
```

Existing trial files are reused. Each trial JSON includes its config, selected positions, controls, velocities and metrics. `summary.csv` and `summary.json` aggregate the results.

The explicit-start/goal `best_tested_scenario.json` preserves the experiment geometry:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario reports/trials/11-09/collision_weights_r21/best_tested_scenario.json \
  --robots 50 --particles 15 --steps 320 --iterations 5 \
  --methods gnn_cp_autodiff \
  --output reports/trials/11-09/collision_weights_r21/best_rerun.json
```

The main CLI also accepts `--collision-weight 7500` to override that cost weight. This retains CP acceptance but does not impose a hard collision constraint.
