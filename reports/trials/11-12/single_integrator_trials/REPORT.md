# Single-integrator collision-weight trials — 11/12

Three setups: 50 robots with required r = 21 or 20, and 25 robots with required r = 10. Each uses 50 and 100 particles. The 50-robot coordinates and assigned goals are copied from the saved 11/09 run. The 25-robot case uses its first 25 robot/goal pairs, without resampling. Targets are passed directly as integers to the tuning tool; they are not rounded through r = 2F + 1. The scenario loader’s default F-based behavior remains unchanged.

Fixed: GNN–CP–autodiff, planar single integrator p[k+1] = p[k] + dt*u[k], with velocity command u hard bounded to norm ≤ 3 m/s; position is the state and there is no acceleration bound, 320 future steps, dt = 0.05 s (16 s), five updates, seed 42, velocity noise 0.1 m/s, R = 3 m. Goal arrival remains a soft cost. CP weight = 5, degree weight = 2, goal weight = 10, SVGD step size = 0.01. Only robot count, robustness target, particle count and collision weight vary.

Robot radius = 0.20 m. Physical overlap means center separation < 0.40 m. The hard filter requires ≥ 0.45 m throughout each constant-acceleration interval, including between waypoints. Pair counts are distinct robot pairs that violate the threshold at any time, not numbers of frames. There are 1,225 possible pairs for 50 robots and 300 for 25 robots; raw pair counts across team sizes therefore have different denominators. CP bounds are predicted on hard radius graphs at every sampled future waypoint. This is not an unconditional robustness guarantee under adaptive search.

## Best CP-only weight in each setup

Ranked by physical-overlap pairs, then clearance-violating pairs, then violating pair/control intervals. These are the best tested selections, not global optima.

| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 100000 | 20 | 209 | 257 | 0.5978 | 128.7 |
| 50 | 20 | 100 | 100000 | 22 | 226 | 272 | 0.3048 | 291.8 |
| 50 | 21 | 50 | 30000 | 22 | 181 | 217 | 1.4376 | 143.3 |
| 50 | 21 | 100 | 10000 | 22 | 229 | 274 | 1.0599 | 293.1 |
| 25 | 10 | 50 | 100000 | 11 | 37 | 45 | 1.2950 | 83.1 |
| 25 | 10 | 100 | 10000 | 13 | 37 | 48 | 1.6929 | 52.1 |

## Hard-filter comparison at default and best CP-only weights

CP and joint pass counts include initialization plus all five updated populations. The joint filter checks every CP-passing candidate and chooses the lowest-cost candidate that also meets clearance. A no-feasible result has no selected trajectory; missing metrics below are undefined, not zero.

| Robots | Required r | Particles | Weight | Status | CP passing | Joint accepted | Selected CP bound | Min separation (m) | Goal error (m) |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 10 | no_feasible_trajectory | 300 | 0 | — | — | — |
| 50 | 20 | 50 | 100000 | no_feasible_trajectory | 258 | 0 | — | — | — |
| 50 | 20 | 100 | 10 | no_feasible_trajectory | 600 | 0 | — | — | — |
| 50 | 20 | 100 | 100000 | no_feasible_trajectory | 589 | 0 | — | — | — |
| 50 | 21 | 50 | 10 | no_feasible_trajectory | 300 | 0 | — | — | — |
| 50 | 21 | 50 | 30000 | no_feasible_trajectory | 300 | 0 | — | — | — |
| 50 | 21 | 100 | 10 | no_feasible_trajectory | 600 | 0 | — | — | — |
| 50 | 21 | 100 | 10000 | no_feasible_trajectory | 600 | 0 | — | — | — |
| 25 | 10 | 50 | 10 | no_feasible_trajectory | 300 | 0 | — | — | — |
| 25 | 10 | 50 | 100000 | no_feasible_trajectory | 56 | 0 | — | — | — |
| 25 | 10 | 100 | 10 | no_feasible_trajectory | 600 | 0 | — | — | — |
| 25 | 10 | 100 | 10000 | no_feasible_trajectory | 600 | 0 | — | — | — |

## Every completed CP-only weight trial

| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 10 | 22 | 290 | 325 | 1.4807 | 148.0 |
| 50 | 20 | 50 | 1000 | 22 | 271 | 321 | 1.4804 | 126.6 |
| 50 | 20 | 50 | 10000 | 22 | 221 | 256 | 1.4709 | 130.2 |
| 50 | 20 | 50 | 100000 | 20 | 209 | 257 | 0.5978 | 128.7 |
| 50 | 20 | 100 | 10 | 22 | 284 | 322 | 1.0652 | 290.2 |
| 50 | 20 | 100 | 1000 | 22 | 273 | 319 | 1.0645 | 287.6 |
| 50 | 20 | 100 | 10000 | 22 | 231 | 272 | 1.0598 | 291.6 |
| 50 | 20 | 100 | 100000 | 22 | 226 | 272 | 0.3048 | 291.8 |
| 50 | 21 | 50 | 10 | 22 | 292 | 325 | 1.4809 | 168.3 |
| 50 | 21 | 50 | 1000 | 22 | 273 | 321 | 1.4806 | 159.1 |
| 50 | 21 | 50 | 6900 | 22 | 230 | 281 | 1.4757 | 142.8 |
| 50 | 21 | 50 | 30000 | 22 | 181 | 217 | 1.4376 | 143.3 |
| 50 | 21 | 50 | 100000 | 21 | 202 | 251 | 0.5330 | 140.7 |
| 50 | 21 | 100 | 10 | 22 | 285 | 322 | 1.0646 | 291.9 |
| 50 | 21 | 100 | 1000 | 22 | 272 | 319 | 1.0645 | 287.6 |
| 50 | 21 | 100 | 10000 | 22 | 229 | 274 | 1.0599 | 293.1 |
| 50 | 21 | 100 | 100000 | 22 | 236 | 283 | 0.3332 | 294.3 |
| 25 | 10 | 50 | 10 | 13 | 67 | 79 | 2.3812 | 29.9 |
| 25 | 10 | 50 | 100 | 13 | 67 | 78 | 2.3812 | 27.5 |
| 25 | 10 | 50 | 1000 | 13 | 61 | 73 | 2.3802 | 27.5 |
| 25 | 10 | 50 | 10000 | 13 | 39 | 48 | 2.3090 | 30.9 |
| 25 | 10 | 50 | 100000 | 11 | 37 | 45 | 1.2950 | 83.1 |
| 25 | 10 | 100 | 10 | 13 | 67 | 79 | 1.7536 | 53.5 |
| 25 | 10 | 100 | 100 | 13 | 68 | 79 | 1.7536 | 51.2 |
| 25 | 10 | 100 | 1000 | 13 | 59 | 73 | 1.7526 | 51.2 |
| 25 | 10 | 100 | 10000 | 13 | 37 | 48 | 1.6929 | 52.1 |
| 25 | 10 | 100 | 100000 | 11 | 41 | 52 | 0.8332 | 154.3 |

## Reproduce

Run from `jax_GNN`. The runner resumes existing matching results and runs the six setups sequentially on CUDA. Separate folders prevent mixing targets or populations. Add a new `--root` to recompute instead of resuming.

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.run_single_integrator_trials
```

For one setup, use the exact even-valued target in the trial CLI:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight --dynamics single_integrator \
  --robots 25 --required-robustness 10 --particles 100 \
  --weights 10 100 1000 10000 100000 --cache-hard-batches --progress \
  --output reports/trials/11-12/single_integrator_trials/robots_25_r10/particles_100/joint_filter \
  --compare-cp-only-output reports/trials/11-12/single_integrator_trials/robots_25_r10/particles_100/cp_only
```

CP-only and joint selections share the exact same optimized populations because final acceptance never feeds back into the gradient. Their solve times refer to one measured run, including collision screening. Scorer setup, post-run metrics and file output are excluded. Runs may share compilation caches. Memory batching retains every particle and waypoint.

`comparison.csv` and `comparison.json` include both policies for every completed weight. `best_results.json` points to the full saved trajectories, controls, velocities and reports. For a group with joint successes it also records the accepted result with the smallest goal error; weighted objective totals are not compared across different collision weights. `manifest.json` records commands and completion state. A no-feasible outcome applies only to the tested candidates and budget, not to every possible trajectory.

## Matched double-integrator comparison

The runner repeats the exact weight/population combinations from the prior 11/11 and 11/12 double-integrator sweeps. Matched starts and goals, horizon, dt, updates, seed, R and collision thresholds are preserved. The numeric control noise remains 0.1, but its units change from acceleration to velocity. SI controls have a hard 3 m/s speed limit; DI has a hard 1 m/s² acceleration limit and soft 3 m/s speed cost. The initialization is minimum-energy for each model, targeting arrival and a stopped final interval without fixing optimized endpoints. The cost keeps the goal, path, collision, speed, degree, GNN surrogate, terminal-velocity and control-effort terms; the control-effort units necessarily differ. These differences are part of the model comparison, so results should not be attributed solely to replacing one integration equation.

| Robots | Required r | Particles | DI best weight | DI overlap pairs | SI best weight | SI overlap pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 100000 | 282 | 100000 | 209 |
| 50 | 20 | 100 | 1000 | 262 | 100000 | 226 |
| 50 | 21 | 50 | 100000 | 282 | 30000 | 181 |
| 50 | 21 | 100 | 1000 | 262 | 10000 | 229 |
| 25 | 10 | 50 | 100 | 58 | 100000 | 37 |
| 25 | 10 | 100 | 1000 | 46 | 10000 | 37 |

The comparison CSV contains each matched weight separately. A missing historical hard-filter run is marked `not run`; it is not inferred. For optimized SI paths, collision checks minimize pair separation along each straight segment. DI paths use quadratic constant-acceleration segments. Both include the start state. SI saved controls and velocities are the same held interval commands in m/s; SI reports contain no acceleration array.

## Exact hard-batch prediction reuse

After the first five-weight SI sweep, the remaining runs enable `--cache-hard-batches`. The frozen CP scorer reuses predictions only when the entire ordered binary-graph minibatch is identical, including node count, batch size, CP method and quantile. Minibatches are not reordered or regrouped. The cache holds at most 4,096 batches. Smooth weights, gradients and trajectory updates are never cached. CUDA validation compares all CP outputs with uncached inference on 195 diverse 50-node graphs and 195 diverse 25-node graphs, including partial minibatches; all bounds match exactly. See `hard_batch_cache_validation.json`.

Timing comparisons also reflect this inference optimization and compilation reuse. Historical DI timings and the first SI sweep are uncached; subsequent SI sweeps use exact-batch reuse. These are observed solve times, not a benchmark isolating dynamics. Per-result reports record cache status and cumulative scorer hit/miss counts.

## Saved-result validation

Independent validation completed for 27 configurations and 54 policy results (12,000 candidate evaluations). It checks the saved SI rollout equation, velocity bound, exact robustness targets, soft-goal settings, CP bounds at all waypoints and continuous linear collision metrics. Details are in `validation.json`. Re-run after a completed matrix with:

```bash
.venv/bin/python -m planning.validate_single_integrator_trials
```

The public CLI also completed a CUDA smoke test for all four cost methods (simple, soft_degree, gnn_cp_spsa, gnn_cp_autodiff). This uses two particles, two steps and one update to check model dispatch and output units; it is separate from the full GNN–CP–autodiff weight matrix. See `diagnostics/cli_smoke.json`.
