# Smooth connectivity, CP penalties, and final trajectory filtering

Date: October 31, 2026

Naming update (November 1, 2026): the degree-based planner is `soft_degree`.
The temporary name `cp_penalty` has been reverted; historical result files
may still contain it. `gnn_cp` is now `gnn_cp_spsa`. Soft degree uses
`degree_weight` for its optimization penalty, `cp_weight = 0`, and a separate
final CP filter.

This document records the earlier implementation and design discussion.

Implementation update (November 1, 2026): `gnn_cp_autodiff` now implements a
continuous GNN optimization surrogate with direct position gradients and a
separate final hard-graph CP filter. The earlier proposed design below is now
available; the original `gnn_cp_spsa` still uses SPSA. See
[the planner README](../planning/README.md#differentiable-gnn-planner) for the
current formula, limitations, and run commands.

## Does soft degree use the CP penalty?

No. The `soft_degree` comparison explicitly sets `cp_weight = 0`. Its particle updates use the goal, path, collision, speed, and soft-degree costs. CP lower bounds are still calculated for every future waypoint of every evaluated candidate and used by the final acceptance filter.

| Method | Soft-degree penalty | CP penalty guiding updates | Final CP filter |
| --- | --- | --- | --- |
| `simple` | No | No | Yes |
| `soft_degree` | Yes | No | Yes |
| `gnn_cp` | Yes | Yes, using SPSA | Yes |

These configurations are defined in [`comparison_configs()`](../planning/run_experiment.py).

## What the soft-degree penalty measures

Smooth connectivity gives each pair of different robots a continuous edge weight based on distance. The weighted degree of robot i is the sum of its incident communication weights. The planner penalizes the squared shortfall between each robot's weighted degree and the required robustness target, averaged across future waypoints and robots:

```text
soft_degree_penalty = degree_weight
                      * mean(max(required_robustness - weighted_degree, 0)^2)
```

The target is `required_robustness = 2F + 1`, where F is the adversary bound. The degree penalty is a connectivity heuristic. It does not measure r-robustness directly, use a CP lower bound, or establish a robustness guarantee. Even satisfying a minimum binary degree condition is insufficient by itself to establish r-robustness.

Smooth connectivity is now the default planning mode. In the current formula, weights reach zero at the communication radius and remain zero outside it. Therefore, this surrogate provides no attraction gradient for a pair already outside communication range.

## Why smooth connectivity has not removed SPSA

The implementation currently has two separate computational paths:

```text
Differentiable degree path:
positions -> smooth connectivity -> weighted degree -> degree penalty
          -> automatic differentiation -> SVGD update

Learned robustness path:
positions -> hard support graph -> GNN probabilities -> discrete class/set
          -> calibrated CP lower bound -> worst-waypoint penalty
          -> SPSA gradient estimate -> SVGD update
```

A hard support graph has a binary edge whenever the distance between two different robots is at most the communication radius. This matches the binary graph representation used to train the saved GNN and calibrate CP.

Smooth connectivity currently makes the degree path differentiable. It does not make the learned robustness path differentiable. That path still includes binary graph construction, host NumPy/Python graph batching, class `argmax` or set thresholding, and discrete lower-bound calculations. Smooth connectivity alone does not remove these barriers.

The earlier change enabled a differentiable connectivity surrogate; it did not implement a fully differentiable GNN robustness objective.

## What cp_weight does, and why extra GNN evaluations occur

For each trajectory, the code computes a CP lower bound at every future waypoint, then takes the minimum:

```text
trajectory_bound = min(lower_bound_at_each_future_waypoint)
CP_penalty = cp_weight * max(required_robustness - trajectory_bound, 0)^2
```

Multiplication by `cp_weight` is cheap. SVGD also needs the derivative of the penalty with respect to robot positions. A penalty value says how unfavorable a trajectory is, but does not supply a direction for moving it.

The current implementation estimates that direction using SPSA: it perturbs the trajectory positions in positive and negative random directions, evaluates GNN/CP for both, and uses the difference to estimate a gradient of a smoothed objective.

With `cp_directions = 2`, each optimization iteration scores four populations of perturbed graphs for gradient estimation. These probes are now vectorized into one combined scorer submission instead of four separate submissions. All methods also perform one evaluation of the updated population for scoring and final filtering. With 16 particles and 10 waypoints, each population evaluation scores 160 graphs. Each run also evaluates its initial population.

| Setting | Updated-population scoring per iteration | Extra SPSA evaluations | Equivalent graph populations per iteration |
| --- | ---: | ---: | ---: |
| `cp_weight = 0` | 1 | 0 | 1 |
| `cp_weight > 0`, two SPSA directions | 1 | 4 | 5 |

The table counts equivalent graph populations, not scorer submissions or individual neural-network batches. The vectorized implementation submits all positive and negative SPSA probes together; bounded GNN minibatching is still handled internally. It scores the same number of graphs. Graph construction and GNN inference add work. A profiler would be needed to attribute measured runtime precisely.

Simply adding a weighted CP penalty without its gradient would not guide SVGD movement. In addition, every passing trajectory has zero CP shortfall penalty, so that penalty does not change cost ranking among passing trajectories.

## Final acceptance and selection

After optimization finishes, the planner filters all saved candidates, including the initial population and each updated population. It uses their cached CP bounds; it does not need another GNN pass for this final comparison.

A candidate passes if:

1. Its CP lower bound is at least `required_robustness` at every future waypoint.
2. Its total cost is finite.

The planner returns the lowest-cost passing candidate. If none pass, it returns `None` with `status = "no_feasible_trajectory"`.

The CP bounds are computed on hard support graphs in both smooth and hard planning modes. The initial robot configuration and continuous motion between waypoints are not certified by this future-waypoint filter.

The experiment's printed success flag additionally checks theoretical feasibility of the target for the robot count and minimum hard-graph degree. Collision avoidance, speed limits, and goal completion remain soft objectives rather than mandatory success criteria. Exact robustness is reported only for at most 10 robots and does not determine the printed success flag.

## Design needed for differentiable learned guidance

To remove SPSA while retaining learned robustness guidance, a future implementation would need:

```text
Optimization:
positions -> smooth weighted graph -> differentiable GNN output
          -> continuous robustness surrogate -> automatic gradients -> SVGD

Final selection:
candidate positions -> hard support graphs -> calibrated CP lower bounds
                    -> filter every future waypoint -> cheapest passing candidate
```

This requires preserving gradients through graph construction and GNN inference, and replacing discrete class and CP operations in the optimization objective with a continuous surrogate. The exact surrogate remains a design choice. It would guide the search; the separate CP filter would still determine acceptance.

The saved GNN was trained on binary graphs. Applying it to fractional edge weights changes its input distribution and requires validation, potentially retraining. If a model is retrained, the model used for final CP scoring requires corresponding calibration. Reusing a differentiable model architecture alone does not establish that its learned predictions remain valid.

## How to evaluate the CP penalty

Compare `soft_degree` and `gnn_cp` using the same scenarios, initial particles, seeds, iteration counts, and final filter. Record planning success rate across runs, candidate pass rate, selected cost, and solving time. Exact robustness should also be checked wherever practical. A time-budget-matched comparison helps distinguish improved guidance from extra computational effort.

The final filter enforces a predicted-bound requirement. Existing calibration limitations and adaptive trajectory selection mean that passing this filter alone does not establish a statistical guarantee of actual robustness for the selected trajectory.

## Code locations

- [`svgd_cp_planner.py`](../planning/svgd_cp_planner.py): `_smooth_cost_single()`, `cp_cost()`, `cp_spsa_gradient()`, and the final filter in `plan()`.
- [`run_experiment.py`](../planning/run_experiment.py): `comparison_configs()`, trajectory metrics, and solving times.
- [`10-31 results`](filtered_results/10-31/README.md): saved CUDA experiments and timing summaries.

## Probe batching update

`cp_spsa_gradient()` now creates an array with axes `[direction, sign, particle, waypoint, robot, coordinate]`, flattens direction/sign/particle into a single candidate axis, and calls `cp_cost()` once. `cp_cost()` flattens candidates and waypoints before GNN scoring. The costs are reshaped back to compute the same two-sided SPSA gradient. This reduces repeated graph-scoring setup while retaining SPSA and the final hard-graph CP filter.

A CUDA microbenchmark using 50 robots, 16 particles, 10 waypoints, and two SPSA directions verified matching gradients. Across three timed repetitions after warm-up, the median CP-gradient time was 0.892 s with separate submissions and 1.006 s with combined submissions. Timings varied substantially, so this small sample does not establish a reliable speed advantage. The checkpoint uses GNN minibatches of 32; combining submissions does not reduce graph inference work. Raw timings: [cp_probe_batching_benchmark.json](filtered_results/10-31/cp_probe_batching_benchmark.json).
