# Robustness targets and collision weights — 11/12

Two setups: 50 robots with required r = 20, and 25 robots with required r = 10. Each uses 50 and 100 particles. The 50-robot coordinates and assigned goals are copied from the saved 11/09 run. The 25-robot case uses its first 25 robot/goal pairs, without resampling. Targets are passed directly as integers to the tuning tool; they are not rounded through r = 2F + 1. The scenario loader’s default F-based behavior remains unchanged.

Fixed: GNN–CP–autodiff, planar double integrator, 320 future steps, dt = 0.05 s (16 s), five updates, seed 42, acceleration noise 0.1 m/s², R = 3 m, acceleration norm ≤ 1 m/s², soft speed bound 3 m/s. Goal arrival remains a soft cost. CP weight = 5, degree weight = 2, goal weight = 10, SVGD step size = 0.01. Only robot count, robustness target, particle count and collision weight vary.

Robot radius = 0.20 m. Physical overlap means center separation < 0.40 m. The hard filter requires ≥ 0.45 m throughout each constant-acceleration interval, including between waypoints. Pair counts are distinct robot pairs that violate the threshold at any time, not numbers of frames. There are 1,225 possible pairs for 50 robots and 300 for 25 robots; raw pair counts across team sizes therefore have different denominators. CP bounds are predicted on hard radius graphs at every sampled future waypoint. This is not an unconditional robustness guarantee under adaptive search.

## Best CP-only weight in each setup

Ranked by physical-overlap pairs, then clearance-violating pairs, then violating pair/control intervals. These are the best tested selections, not global optima.

| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 100000 | 22 | 282 | 324 | 0.0000 | 115.6 |
| 50 | 20 | 100 | 1000 | 22 | 262 | 298 | 0.5796 | 278.6 |
| 25 | 10 | 50 | 100 | 13 | 58 | 85 | 0.2238 | 91.3 |
| 25 | 10 | 100 | 1000 | 10 | 46 | 61 | 0.8372 | 178.5 |

## Hard-filter comparison at default and best CP-only weights

CP and joint pass counts include initialization plus all five updated populations. The joint filter checks every CP-passing candidate and chooses the lowest-cost candidate that also meets clearance. A no-feasible result has no selected trajectory; missing metrics below are undefined, not zero.

| Robots | Required r | Particles | Weight | Status | CP passing | Joint accepted | Selected CP bound | Min separation (m) | Goal error (m) |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 10 | no_feasible_trajectory | 51 | 0 | — | — | — |
| 50 | 20 | 50 | 100000 | no_feasible_trajectory | 50 | 0 | — | — | — |
| 50 | 20 | 100 | 10 | no_feasible_trajectory | 289 | 0 | — | — | — |
| 50 | 20 | 100 | 1000 | no_feasible_trajectory | 329 | 0 | — | — | — |
| 25 | 10 | 50 | 10 | no_feasible_trajectory | 300 | 0 | — | — | — |
| 25 | 10 | 50 | 100 | no_feasible_trajectory | 247 | 0 | — | — | — |
| 25 | 10 | 100 | 10 | no_feasible_trajectory | 600 | 0 | — | — | — |
| 25 | 10 | 100 | 1000 | no_feasible_trajectory | 347 | 0 | — | — | — |

## Every completed CP-only weight trial

| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 50 | 10 | 22 | 287 | 328 | 0.0000 | 161.8 |
| 50 | 20 | 50 | 1000 | 22 | 289 | 325 | 0.0000 | 136.0 |
| 50 | 20 | 50 | 10000 | 22 | 289 | 325 | 0.0000 | 128.0 |
| 50 | 20 | 50 | 100000 | 22 | 282 | 324 | 0.0000 | 115.6 |
| 50 | 20 | 100 | 10 | 22 | 317 | 372 | 0.3537 | 312.1 |
| 50 | 20 | 100 | 1000 | 22 | 262 | 298 | 0.5796 | 278.6 |
| 50 | 20 | 100 | 10000 | 22 | 290 | 334 | 0.0000 | 254.3 |
| 50 | 20 | 100 | 100000 | 22 | 280 | 327 | 0.0000 | 229.1 |
| 25 | 10 | 50 | 10 | 13 | 71 | 93 | 0.1076 | 93.7 |
| 25 | 10 | 50 | 100 | 13 | 58 | 85 | 0.2238 | 91.3 |
| 25 | 10 | 50 | 1000 | 13 | 64 | 78 | 0.0000 | 92.7 |
| 25 | 10 | 50 | 10000 | 13 | 64 | 78 | 0.0000 | 89.8 |
| 25 | 10 | 50 | 100000 | 13 | 64 | 78 | 0.0000 | 88.1 |
| 25 | 10 | 100 | 10 | 13 | 67 | 87 | 0.0541 | 178.4 |
| 25 | 10 | 100 | 100 | 13 | 70 | 86 | 0.1268 | 178.9 |
| 25 | 10 | 100 | 1000 | 10 | 46 | 61 | 0.8372 | 178.5 |
| 25 | 10 | 100 | 10000 | 13 | 64 | 78 | 0.0000 | 177.9 |
| 25 | 10 | 100 | 100000 | 13 | 64 | 78 | 0.0000 | 175.2 |

## Reproduce

Run from `jax_GNN`. The runner resumes existing matching results and runs the four setups sequentially on CUDA. Separate folders prevent mixing targets or populations. Add a new `--root` to recompute instead of resuming.

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.run_robustness_collision_trials
```

For one setup, use the exact even-valued target in the trial CLI:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight \
  --robots 25 --required-robustness 10 --particles 100 \
  --weights 10 1000 10000 100000 --progress \
  --output reports/trials/11-12/robustness_collision_trials/robots_25_r10/particles_100/joint_filter \
  --compare-cp-only-output reports/trials/11-12/robustness_collision_trials/robots_25_r10/particles_100/cp_only
```

CP-only and joint selections share the exact same optimized populations because final acceptance never feeds back into the gradient. Their solve times refer to one measured run, including collision screening. Scorer setup, post-run metrics and file output are excluded. Runs may share compilation caches. Memory batching retains every particle and waypoint.

`comparison.csv` and `comparison.json` include both policies for every completed weight. `best_results.json` points to the full saved trajectories, controls, velocities and reports. For a group with joint successes it also records the accepted result with the smallest goal error; weighted objective totals are not compared across different collision weights. `manifest.json` records commands and completion state. A no-feasible outcome applies only to the tested candidates and budget, not to every possible trajectory.

## Intermediate-weight follow-up

The 25-robot runs showed a gap between weight 10, whose updated candidates passed CP, and weight 1,000, which lost CP feasibility at 50 particles. Weight 100 was added for both 50 and 100 particles to test this middle range. Its commands and completion state are recorded separately in `adaptive_manifest.json`.

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.run_robustness_collision_trials --cases 25:10:50 25:10:100 --weights 100 --manifest-name adaptive_manifest.json
```

## Memory handling

The first 100-particle run was killed before saving a feasibility result. The SVGD pair-distance calculation materialized a full particle-pair-by-control tensor and its square. It now computes the same direct squared distances one upper-triangle row at a time. Reference tests check the resulting SVGD directions, including nearly identical particles with large offsets.

For 100 particles × 320 steps × 50 robots × 2 controls, the pair-distance temporary estimate drops from 4.77 GiB to 48.3 MiB. A standalone full-size SVGD update completed in 0.287 s with process peak RSS 386.3 MiB. This standalone CPU check excludes the GNN and is not the full planning time. All requested particles and planning settings are retained. The completed 50-particle first sweep precedes this memory change; subsequent sweeps use it. No killed run is counted as a no-feasible trajectory.
