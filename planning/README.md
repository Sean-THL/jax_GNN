# Experimental SVGD + conformal GATv2 planner

![SVGD, GATv2, and conformal prediction pipeline](pipeline.svg)

`planning.svgd_cp_planner` is an isolated prototype. A particle contains a short
sequence of 2-D waypoints for every robot. Each future waypoint is converted
to a directed communication graph. The default `hard` mode preserves the
trained binary radius graph. The optional `smooth` mode uses equation (7) of
Capelli and Sabattini, with `sigma = R^4 / ln(2)` by default, weighted degrees,
and a fixed dense off-diagonal edge index.

The experiment compares three SVGD objectives using identical initial particles:

| Run | Cost during SVGD | Purpose |
| --- | --- | --- |
| `simple` | Goal, motion, collision, and speed | No robustness objective |
| `soft_degree` | Simple costs plus a smooth degree-shortfall penalty | Cheap connectivity heuristic |
| `gnn_cp` | Soft-degree costs plus a GATv2/CP robustness-shortfall penalty | Test the added learned signal |

The GNN+CP run penalizes a trajectory if the **minimum conformal lower bound over its future
waypoints** is below `required_robustness`. For scenario files, this target is
derived from the user-supplied adversary bound as
`required_robustness = 2 * adversary_bound + 1`. `cp_kind="one_sided"` uses the
integer residual score; `cp_kind="lac"` uses the lower endpoint of the LAC set.
The discrete CP cost has no ordinary path derivative. The prototype estimates
the gradient of a Gaussian-smoothed CP cost with two-sided random probes and
applies an RBF SVGD update to the particles. The soft degree term provides a
differentiable connectivity signal. In smooth mode it uses the equation-(7)
adjacency; hard mode retains the original sigmoid degree heuristic.

In both planning modes, GNN/CP certification uses the binary hard-radius
support graph. The checkpoint and calibration records are binary graphs, and
exact r-robustness is defined on that same support. Smooth weighted adjacency
is used only as a differentiable planning surrogate; feeding it directly to
the binary-trained CP scorer would violate calibration exchangeability.

The GATv2 layer uses a scalar edge value both in its learned attention logit
and as a multiplicative attention prior. Thus zero-weight dense edges cannot
send messages. For binary compact graphs this layer is algebraically identical
to the original `jraphx` layer, so existing checkpoints load unchanged.

## Run the included demonstration

The included scenario reproducibly samples a pool of 50 robots with
**arbitrary coordinate units**. A compact 10-robot seed is generated first;
each additional robot is accepted only when it has at least
`required_robustness` neighbors within 80% of the communication radius. That
20% distance margin protects the required edges from small waypoint noise.
Goals apply one common translation to the starting formation, preserving all
pairwise distances and hard communication edges. `--robots N` selects the
same first N robots in every run. It is only for checking the pipeline, not a
real communication or flight setup.
From the repository root:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario planning/examples/demo_scenario.json \
  --connectivity-mode hard \
  --output /tmp/svgd_results.json
```

Run the weighted mode with the default sigma:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario planning/examples/demo_scenario.json \
  --connectivity-mode smooth \
  --output /tmp/svgd_smooth_results.json
```

Use `--connectivity-sigma VALUE` to override `R^4 / ln(2)`. Sigma must be
positive. These options can instead be stored as `connectivity_mode` and
`connectivity_sigma` in the scenario's `config` object.

The sampling interval is `sampling_time` (also accepted as `--dt`). The motion
limit is computed rather than entered independently:

```text
maximum_step = maximum_speed * sampling_time
```

For example, this runs 20 robots, a 30-step horizon, and `dt=0.1`:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning \
  --scenario planning/examples/demo_scenario.json \
  --robots 20 --steps 30 --dt 0.1 \
  --output reports/robots_20_steps_30_dt_0p1.json
```

Run the supplied sweep (5–50 robots in increments of 5;
10–50 steps in increments of 10; and `dt` 0.05, 0.1, and 0.2) in hard mode with:

```bash
JAX_PLATFORMS=cuda bash planning/run_sweep.sh
```

Run the same grid in smooth mode with:

```bash
CONNECTIVITY_MODE=smooth JAX_PLATFORMS=cuda bash planning/run_sweep.sh
```

Results are stored in `reports/sweep/hard/` or `reports/sweep/smooth/`
according to `CONNECTIVITY_MODE`, so the two sweeps cannot overwrite each
other.

This launches 150 experiments, each containing all three planning methods, so
it can take a long time. Edit the three arrays at the top of `run_sweep.sh` to
reduce or densify the grid. With `F=2`, `r_required=5`; runs with 5–8 robots
are retained as infeasible controls and always report `Success=NO`.

The runner creates 16 initial trajectories around a straight-line route using
one fixed random seed. It gives the **same particles** to all three runs and
uses the same SVGD iteration count. The simple and soft-degree objectives do
not query the GNN during optimization. After optimization, every selected
trajectory is evaluated with the same graph scorer and, for at most 10 robots,
an exhaustive robustness checker. The table reports CP lower bound, exact
robustness, goal error, maximum waypoint step, minimum robot separation, and
cost under the same full objective. Runtime measures SVGD optimization only;
loading/calibrating the scorer and post-run evaluation are excluded. The JSON
output records all three selected trajectories and metrics. Use `--help` to
change particles, horizon, iterations, noise, seed, checkpoint, or calibration
file. Use `JAX_PLATFORMS=cpu` if CUDA is unavailable.

Unless `--noise` is supplied explicitly, initial waypoint noise is set to 10%
of `maximum_speed * sampling_time`. Use `--noise-fraction` to change that
fraction. This keeps random perturbations proportional to how far a robot is
allowed to move in one sampling interval.

```mermaid
flowchart LR
    A[Same starts, goals, seed and initial particles] --> B[Simple SVGD]
    A --> C[SVGD + soft degree]
    A --> D[SVGD + soft degree + GATv2/CP]
    B --> E[Evaluate selected trajectory]
    C --> E
    D --> E
    E --> F[Compare exact robustness, constraints, goal error and runtime]
```

This is an ablation of the added objectives, not a claim that lower training
cost alone produces a better plan. For a useful result, repeat the comparison
across realistic scenarios and seeds. The current one-sided CP quantile is
zero for the saved checkpoint, and the calibration data omit robustness-zero
graphs, so the reported lower bound is not a safety certificate.

To try your own setting, copy `planning/examples/demo_scenario.json`, replace `start`
and `goal` with your robots' 2-D positions (same number and order of robots),
and replace `config.communication_radius`, `minimum_separation`,
`maximum_speed`, `sampling_time`, and `adversary_bound` with your
measured/planned values. For
example, `adversary_bound: 2` produces `required_robustness: 5`.
The radius, separation, maximum step, start, and goal must use the same length
unit. The script will create the initial particles for you. For more than 10
robots, the exact robustness field is `null` because exhaustive verification
becomes expensive.

## Use the planner as a Python module

Example (from the repository root, with the project virtual environment):

```python
from pathlib import Path
import numpy as np
from planning.svgd_cp_planner import ConformalGraphScorer, PlannerConfig, plan

start = np.asarray(...)  # (robots, 2), in meters
goal = np.asarray(...)   # (robots, 2), in meters
particles = np.asarray(...)  # (particles, future_steps, robots, 2)

config = PlannerConfig(
    communication_radius=2.0,   # replace with measured radio/connectivity rule
    minimum_separation=0.5,
    maximum_speed=1.0,
    sampling_time=0.1,          # dt; maximum_step is 1.0 * 0.1 = 0.1
    required_robustness=5,      # 2F+1 for F=2
)
scorer = ConformalGraphScorer(
    Path("checkpoints/full_run_xK0R3I/new_gatv2.msgpack"),
    Path("calib_data.json"),
    config.cp_coverage,
    config.cp_kind,
)
waypoints, report = plan(start, goal, particles, scorer, config, iterations=20)
print(report)
```

This is a planning experiment, not a flight controller. The example distances
are placeholders. The repository has no authoritative dynamics, communication
radius, obstacle map, or planning scenario. The current calibration data have
no robustness-0 graphs, and the selected plan is chosen adaptively from many
GNN queries. Consequently the nominal CP coverage cannot be interpreted as a
guarantee for the selected trajectory. Validate the communication graph rule,
calibration distribution, collision/dynamics constraints, and selected plans
before integrating with a robot system.

## Differentiability and checkpoint limits

The differentiable path in smooth mode is `positions -> equation-(7)
adjacency -> weighted degree -> degree shortfall -> SVGD gradient`.

The learned robustness path is not end-to-end differentiable. It still contains
the hard support condition at `d = R`, class `argmax`, integer conformal
quantile/lower-bound lookup, a minimum over future steps, and particle `argmin`.
It also crosses host NumPy batching. The planner therefore continues to use
two-sided SPSA probes for the GNN/CP penalty. Exact r-robustness evaluation is
combinatorial and is never differentiated.

The gradient audit found these concrete barriers:

| Pipeline operation | Gradient consequence |
| --- | --- |
| Hard `distance_squared <= R^2` and `np.nonzero` | Binary values and a dynamic sparse edge list; no position derivative |
| Host `np.asarray` plus Python graph-record/batch construction | Breaks JAX tracing before GNN inference |
| GNN class `argmax` | Converts continuous probabilities to an integer class |
| LAC comparison and first-true `argmax` | Boolean threshold and discrete set endpoint |
| One-sided CP `ceil`, `clip`, and integer cast | Discrete calibrated lookup/output |
| Minimum lower bound across waypoints | Selects a worst step after the already-discrete lookup |
| Exhaustive r-robustness subset/bit-count logic | Purely combinatorial evaluation |
| Final particle `argmin` | Discrete plan selection, not an optimization gradient |

The hard support test remains in equation (7), but clipping makes the two
pieces meet with zero first derivative at the boundary. It is therefore finite
and differentiable there in JAX while still providing no recovery gradient
from outside the radius.

Equation (7) has zero position gradient at `d = 0`, at `d = R`, and throughout
`d > R`; it cannot attract robots that are already outside communication
range. The branch is evaluated safely by clipping `R^2 - d^2` before `expm1`,
so the inactive exponential cannot overflow under JIT.

The bundled GATv2 checkpoint was trained only on binary edge attributes and
integer degrees. Loading it in smooth mode is structurally compatible, but its
predictions on fractional weights/degrees are out of distribution. Retraining
on weighted graphs, followed by fresh conformal calibration from the same
weighted-graph distribution, is recommended before interpreting those scores.
