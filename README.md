# Graph robustness classification: four entry points

Run commands from `jax_GNN`. The original GAT and GATv2 entry points follow the
historical 80/20 split of `testing_data.json`. The new GAT and GATv2 entry
points use `training_data.json` (90/10 train/validation split, seed 42) and
evaluate on all graphs in the separate `testing_data.json`. `calib_data.json`
is unused. The target is `robustness`, with integer classes 0–25. The new
loader rounds labels within 0.001 of an integer (for example, 20.999977
becomes 21) and rejects labels outside this tolerance. Source JSON files are
not modified.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
python -m pip install -r requirements.txt
```

These requirements install CPU JAX. For GPU setup, see
https://docs.jax.dev/en/latest/installation.html and retain JAX version 0.8.0.

## Train, then test

| Representation | Layer | Training | Testing | Checkpoint under `checkpoints/` |
| --- | --- | --- | --- | --- |
| Original padded, historical pipeline | GATConv | `python jax_train.py` | `python jax_test.py` | `historical_gat.msgpack` |
| Original padded, historical pipeline | GATv2Conv | `python jax_train_gatv2.py` | `python jax_test_gatv2.py` | `historical_gatv2.msgpack` |
| New actual graphs | GATConv | `python jax_train_new.py` | `python jax_test_new.py` | `new_gat.msgpack` |
| New actual graphs | GATv2Conv | `python jax_train_new_gatv2.py` | `python jax_test_new_gatv2.py` | `new_gatv2.msgpack` |

The historical original runs save final weights and optimizer state after
training ends. Their test scripts report results for the 20 graphs used for
early stopping. The new runs save their best validation model as training
progresses; their test scripts use the separate 100-graph test file. Each run
has a JSON metadata file. Previously saved `model.msgpack` and Orbax
checkpoints are not used by these commands.

The original representation retains 50 nodes and 2,500 edge slots per graph,
including placeholder nodes and edges. The new representation computes each
graph using its actual nodes and edges; batch padding is disconnected from real
graphs and excluded from mean pooling. Both use three attention layers, four attention heads, 100
hidden features, layer normalization, mean pooling, and a three-layer classifier.
Historical training and evaluation are in `jax_historical_baseline.py` and
construct graphs using `jax_dataset.py` and `Batch.from_data_list()`. The new
training and evaluation paths are in `jax_variants.py`; they construct compact
NumPy batches. Models are imported from `jax_gnn.py` (original GAT),
`jax_gnn_gatv2.py` (original GATv2), `jax_gnn_new.py` (new GAT), and
`jax_gnn_new_gatv2.py` (new GATv2).

Original variants default to 500 epochs with patience 50, batch size 10,
fixed training order, and one-hot cross-entropy. New variants default to
100 epochs with patience 15, batch size 10, shuffled training, and integer
cross-entropy. All use Adam with learning rate 0.005.

## Historical original baselines

The original entry points follow the original training procedure. Both read
`testing_data.json`, split
it 80/20 with seed 42, construct graphs through `jax_dataset.py`, and use
`Batch.from_data_list`, batch size 10, one-hot cross-entropy, fixed training
order, learning rate 0.005, and up to 500 epochs with patience 50. GATv2
changes only the convolution model. Final weights and optimizer state are
saved, without restoring the best validation weights.

```bash
python jax_train_original_baseline.py
python jax_test_original_baseline.py
python jax_train_original_gatv2_baseline.py
python jax_test_original_gatv2_baseline.py
```

The checkpoints are `checkpoints/historical_gat.msgpack` and
`checkpoints/historical_gatv2.msgpack`; separate `.optimizer.msgpack` files
hold final optimizer states. The test entry points evaluate the 20 graphs used
for validation and early stopping, so those metrics are not independent test
results. The longer `jax_train_original_baseline.py` and
`jax_train_original_gatv2_baseline.py` names are aliases for the same
historical training pipeline.

## Quick check and options

To run all four public entry points sequentially:

```bash
bash run_all_variants.sh
```

This runs sequentially on CUDA and creates a new `checkpoints/full_run_*`
directory. Original models follow the historical 80/20 split, batch size 32,
500-epoch limit and patience 50; new models follow the full training dataset,
batch size 32, 100-epoch limit and patience 15. Set `BATCH_SIZE` to change the
batch size for all four. The original validation metrics
and new independent test metrics are not directly comparable. Ctrl+C stops the
sequence; restarting creates a fresh run directory. Example overrides:

```bash
BASELINE_EPOCHS=200 EPOCHS=200 bash run_all_variants.sh
```

To reproduce the earlier matched-settings four-way comparison within the
shared pipeline, use `bash run_shared_variants.sh`. It invokes
`jax_variants.py` for all four representations without changing the public
original entry points.

Example using 100 training records, followed by evaluation on the separate test file:

```bash
python jax_train_new.py --epochs 1 --limit 100
python jax_test_new.py
```

The new entry points accept the shared-pipeline options:

```bash
python jax_train_new_gatv2.py --epochs 100 --patience 15 --batch-size 10 --learning-rate 0.005
```

Other new-variant options: `--validation-fraction`, `--seed`, `--train-data`,
`--checkpoint`, and `--no-jit`. Their test scripts accept `--test-data`,
`--checkpoint`, and `--batch-size`. Historical original train scripts accept
`--epochs`, `--patience`, `--batch-size`, and `--checkpoint`; their test scripts
accept `--checkpoint` and `--batch-size` (which defaults to the saved value).
Paths supplied through flags are relative to your current working directory;
default data and checkpoint paths are relative to the scripts.

For new variants, graphs are converted once to compact NumPy arrays in host memory. Each batch
is assembled on the CPU, and a background thread prepares the next batch while
training runs. Only batches are transferred to the device. Loss values are
collected at the end of each training/validation phase rather than copied to
Python after every step. Both phases show progress and elapsed time. Initial
JSON parsing still loads the full file into host memory. New graph batches use
a fixed node capacity (batch size times the largest graph in the split, plus a
dummy node) and power-of-two edge capacities, starting at 256. Extra edges connect
only to the dummy node, and extra nodes have a graph ID outside the pooling range.
This preserves real graph attention, pooling, and labels while allowing JAX to
reuse a small set of compiled steps across shuffled batches. Each new edge bucket
still incurs an initial compilation; validation and partial batches compile
separately. `--no-jit` is for debugging and usually slows training. Check the
printed `Devices:` line to confirm whether JAX is using CPU or GPU.

Testing reports exact accuracy, the fraction of predictions <= actual labels,
mean signed deviation (actual minus prediction), and accuracy within ±1 and ±2.
The observed lower-bound rate is not a mathematical guarantee. Independent files
should also be checked for duplicate graphs when interpreting generalization.
