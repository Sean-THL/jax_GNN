# Graph Robustness Classifier

This project trains JAX/Flax graph attention networks to classify a graph's
`robustness` value. It preserves Joon's original GAT workflow and adds GATv2,
compact graph batching, independent evaluation data, consistent checkpointing,
and optional conformal uncertainty estimates.

The supplied labels are precomputed integer classes from 0 through 25. This
repository trains and evaluates classifiers; it does not calculate
r-robustness labels from first principles.

## Highlights

- Four supported model configurations: historical and integrated pipelines,
  each with GAT or GATv2 convolution layers.
- Reproducible train/validation splitting with a configurable seed.
- Best-validation checkpointing for the integrated pipelines.
- Independent test-set evaluation for the integrated pipelines.
- Optional conformal prediction sets and one-sided robustness lower bounds.
- JIT-friendly compact batching with disconnected padding and reusable edge
  buckets.

## Pipeline comparison

| Pipeline | Convolution | Graph representation | Data split | Training behavior | Saved checkpoint | Evaluation set |
| --- | --- | --- | --- | --- | --- | --- |
| Historical GAT | `GATConv` | 50 nodes and 2,500 edge slots per graph, including placeholders | 80/20 split of `testing_data.json` | Fixed order, one-hot cross-entropy, up to 500 epochs | Final model and optimizer | The 20 graphs also used for early stopping |
| Historical GATv2 | `GATv2Conv` | 50 nodes and 2,500 edge slots per graph, including placeholders | 80/20 split of `testing_data.json` | Fixed order, one-hot cross-entropy, up to 500 epochs | Final model and optimizer | The 20 graphs also used for early stopping |
| Integrated GAT | `GATConv` | Actual nodes and edges with disconnected batch padding | 90/10 split of `training_data.json` | Shuffled batches, integer cross-entropy, up to 100 epochs | Best validation model | All graphs in the independent `testing_data.json` |
| Integrated GATv2 | `GATv2Conv` | Actual nodes and edges with disconnected batch padding | 90/10 split of `training_data.json` | Shuffled batches, integer cross-entropy, up to 100 epochs | Best validation model | All graphs in the independent `testing_data.json` |

The historical evaluation metrics are validation results because the same 20
graphs control early stopping. They should not be compared directly with the
integrated pipelines' independent test metrics.

Joon's original `main` pipeline introduced actual-graph processing with
`GATConv` and a fixed 70/10/20 split of the 100-graph dataset. The integrated
pipeline retains that actual-node and actual-edge representation, adds GATv2,
uses the larger `training_data.json` for training and validation, and reserves
all of `testing_data.json` for independent evaluation.

## Model architecture

All four configurations use the same high-level classifier design so that the
representation and attention layer can be compared fairly:

- two node features: node degree and a constant value of 1;
- one edge feature with a constant value of 1;
- three graph attention layers with four attention heads;
- 100 hidden features and layer normalization after each attention layer;
- global mean pooling; and
- a three-layer multilayer perceptron with 26 output classes.

The integrated representation operates on each graph's real nodes and directed
edges. Extra batch edges connect only to a dummy node, and extra nodes are
excluded from graph pooling. This keeps padding from affecting attention or
predictions while allowing JAX to reuse compiled functions.

## Setup

Use Python 3.12 and run the following commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH  # useful when a ROS environment is active
python -m pip install -r requirements.txt
```

The requirements install CPU JAX 0.8.0. NVIDIA GPU users should install the
matching CUDA-enabled JAX package by following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Data

| File | Purpose | Tracked by Git |
| --- | --- | --- |
| `training_data.json` | Training and validation for the integrated pipelines | No; place it in the repository root |
| `testing_data.json` | Independent evaluation for the integrated pipelines and the source of the historical 80/20 split | Yes |
| `calib_data.json` | Optional conformal calibration | Yes |

The tracked test file contains 100 graphs with 9–47 actual nodes. The larger
training file is ignored by Git because of its size and must be supplied
separately before an integrated training run.

Each record must provide `degree`, `edge_index`, and `robustness`. Robustness
labels must be integers in `[0, 25]`; values within 0.001 of an integer are
rounded during loading. Input JSON files are never modified.

## Train and evaluate

### Integrated pipelines

Train and test the compact GAT pipeline:

```bash
python jax_train_new.py
python jax_test_new.py
```

Train and test the compact GATv2 pipeline:

```bash
python jax_train_new_gatv2.py
python jax_test_new_gatv2.py
```

The default integrated configuration uses batch size 10, learning rate 0.005,
seed 42, a 90/10 train/validation split, a 100-epoch limit, and early stopping
patience of 15 epochs. It writes the best models to
`checkpoints/new_gat.msgpack` and `checkpoints/new_gatv2.msgpack` with JSON
metadata beside each checkpoint.

For a quick end-to-end check with a subset of the training data:

```bash
python jax_train_new_gatv2.py --epochs 1 --limit 100
python jax_test_new_gatv2.py
```

Common training options include `--epochs`, `--patience`, `--batch-size`,
`--learning-rate`, `--validation-fraction`, `--seed`, `--train-data`,
`--checkpoint`, `--limit`, and `--no-jit`. Test scripts accept `--test-data`,
`--checkpoint`, and `--batch-size`. Run a script with `--help` for the complete
CLI reference.

### Historical baselines

The original padded GAT entry points remain available:

```bash
python jax_train.py
python jax_test.py
```

The corresponding GATv2 baseline changes only the attention convolution:

```bash
python jax_train_gatv2.py
python jax_test_gatv2.py
```

The longer `jax_train_original_*_baseline.py` and
`jax_test_original_*_baseline.py` names are aliases for these historical
workflows. Their checkpoints are `checkpoints/historical_gat.msgpack` and
`checkpoints/historical_gatv2.msgpack`; separate `.optimizer.msgpack` files
contain their final optimizer states.

Joon's fixed original pipeline is also retained. It keeps the original padded
representation and 80/20 experiment design while correcting checkpoint and
test-batching consistency:

```bash
python jax_train_old_fixed.py
python jax_test_old_fixed.py
```

It saves its best validation checkpoint as
`checkpoints/old_fixed_model.msgpack`. Its evaluation subset is also used for
early stopping, so the reported result is not an independent test score.

## Run the complete comparison

Run the four public configurations sequentially:

```bash
mkdir -p checkpoints
bash run_all_variants.sh
```

The script creates a unique directory under `checkpoints/full_run_*` and saves
checkpoints and logs for every configuration. It defaults to CUDA; on a
CPU-only system, run:

```bash
JAX_PLATFORMS=cpu bash run_all_variants.sh
```

Environment variables can override the shared settings:

```bash
BATCH_SIZE=32 BASELINE_EPOCHS=200 EPOCHS=200 bash run_all_variants.sh
```

For a controlled four-way experiment in which every model uses the integrated
training engine and the same dataset split, run:

```bash
bash run_shared_variants.sh
```

## Conformal evaluation

The integrated test scripts can report calibrated uncertainty through LAC
prediction sets and one-sided lower bounds. For 90% desired marginal coverage:

```bash
python jax_test_new.py --conformal-coverage 0.9
python jax_test_new_gatv2.py --conformal-coverage 0.9
```

Use `--calibration-data PATH` to select a different independent calibration
file. The report includes calibration and test coverage, mean prediction-set
size, one-sided lower-bound coverage, and any examples missed by the prediction
sets. The coverage interpretation requires calibration and future graphs to be
exchangeable.

## Reported metrics

Evaluation scripts report:

- exact robustness accuracy;
- the fraction of predictions less than or equal to the true robustness;
- mean signed deviation (`actual - predicted`); and
- accuracy within one and two robustness classes.

The observed lower-bound rate is an empirical metric, not a mathematical
guarantee. Use the conformal option when calibrated uncertainty bounds are
required.

## Repository structure

| Path | Description |
| --- | --- |
| `jax_variants.py` | Integrated training, evaluation, batching, checkpointing, and conformal integration |
| `jax_historical_baseline.py` | Shared implementation of the historical GAT and GATv2 baselines |
| `jax_gnn.py`, `jax_gnn_gatv2.py` | Historical padded GAT and GATv2 models |
| `jax_gnn_new.py`, `jax_gnn_new_gatv2.py` | Compact integrated GAT and GATv2 models |
| `jax_dataset.py`, `jax_dataset_new.py` | Historical and compact graph construction helpers |
| `jax_conformal.py` | Conformal prediction sets and one-sided lower bounds |
| `run_all_variants.sh` | End-to-end comparison using the public entry points |
| `run_shared_variants.sh` | Matched-settings comparison using the integrated engine |

## Notes on performance

Integrated records are converted once to compact NumPy arrays in host memory.
A background thread prepares the next batch while the current batch runs on the
device, and only complete batches are transferred to the device. Power-of-two
edge capacities reduce the number of JAX compilations, although a new edge
bucket, validation shape, or partial batch can still trigger compilation.

The initial JSON parse loads the complete source file into host memory. Use
`--limit` for smoke tests and `--no-jit` only for debugging. Check the printed
`Devices:` line at startup to confirm whether JAX selected CPU or GPU.
