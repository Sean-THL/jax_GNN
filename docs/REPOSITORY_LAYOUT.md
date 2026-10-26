# Repository layout

The repository has one maintained classifier package and one maintained planner
package. Historical implementations remain available for reproducibility but
are isolated from current work.

## Maintained code

| Path | Responsibility |
| --- | --- |
| `gnn/` | Graph construction, GAT/GATv2 models, training, evaluation, and the unified CLI |
| `conformal/` | Conformal scores, prediction sets, and lower bounds |
| `planning/` | Hard and smooth connectivity, SVGD planning, and experiment runner |
| `tests/` | Unit and integration tests |
| `scripts/` | Multi-run experiment shell scripts |
| `reports/` | Checked-in reports and representative results |

Use these entry points from the repository root:

```bash
python -m gnn
python -m gnn train --model gatv2
python -m gnn test --model gatv2
python -m planning --scenario planning/examples/demo_scenario.json
```

## Historical code

`legacy/` contains the padded graph representation, historical training
procedure, the prefetch-era implementation, and their original Orbax
checkpoints. Current code must not import from `legacy/` except when the user
explicitly selects `--pipeline historical`.

The fixed older workflow remains runnable as:

```bash
python -m legacy.old_fixed_train
python -m legacy.old_fixed_test
```

## Data and generated artifacts

- `testing_data.json` and `calib_data.json` are the small tracked datasets.
- `training_data.json` is supplied locally and ignored because of its size.
- `checkpoints/` is generated and ignored.
- Local scenarios and generated sweep reports should not be committed unless
  they are intentionally curated as examples or final reports.

Avoid adding new compatibility wrappers at the repository root. Add new
commands to `gnn/cli.py` or the relevant package's `__main__.py` instead.
