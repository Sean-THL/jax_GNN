# Filtered CUDA results — 2026-10-31

10 robots, 16 particles, 10 future waypoints, 5 iterations, seed 42. Robustness target: 5.

| Method | Success | Solving time (s) | Minimum CP bound | Minimum exact robustness | Passing / evaluated candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| simple | YES | 4.828 | 5 | 5 | 96 / 96 |
| soft_degree | YES | 2.070 | 5 | 5 | 96 / 96 |
| gnn_cp | YES | 8.530 | 5 | 5 | 17 / 96 |

Solving time includes optimization and final CP filtering, including any JIT compilation incurred within the planner call. Scorer loading/calibration and post-run evaluation are excluded. These are single-run timings, not benchmark averages.

The JSON contains selected trajectories, per-step robustness bounds, metrics, and planner history. Candidate counts include initial and intermediate populations; repeated candidates are counted separately.

Run from your terminal:

```bash
cd /home/robin/Seans_Project/DASC/GNN-Based_r-robustness_multi_robot/jax_GNN
mkdir -p reports/filtered_results/10-31
run_stamp=$(date +%Y%m%d_%H%M%S)
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario planning/examples/demo_scenario.json \
  --robots 10 --particles 16 --steps 10 --iterations 5 \
  --connectivity-mode hard \
  --output "reports/filtered_results/10-31/cuda_${run_stamp}.json"
```

Each run writes a separate JSON containing solving times in `results.<method>.wall_seconds`. The terminal table also shows these times in the Seconds column. `summary.csv` describes the saved initial run.

## 50 robots with smooth connectivity

Smooth connectivity is now the planning default. CP acceptance continues to use hard support graphs.

| Method | Success | Solving time (s) | Minimum CP bound |
| --- | --- | ---: | ---: |
| simple | YES | 6.034 | 9 |
| soft_degree | YES | 2.150 | 11 |
| gnn_cp | YES | 6.143 | 11 |

Exact robustness is not computed for 50 robots. Collision and speed limits remain soft penalties; this run has minimum separations below the configured 0.03.

To reproduce, use the terminal command above with `--robots 50 --connectivity-mode smooth`.
