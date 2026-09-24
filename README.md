# Graph robustness classifier

This project trains a JAX/Flax graph attention network to classify the
`robustness` value in `testing_data.json`. The file contains 100 graphs with
9–47 nodes. The split is fixed at 70 training, 10 validation, and 20 held-out
test graphs.

## Set up

Use Python 3.12 and run these commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH  # useful if a ROS environment is active
python -m pip install -r requirements.txt
```

The requirements install CPU JAX. NVIDIA GPU users should follow the current
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
their driver and CUDA version.

## Run the pipeline

```bash
python jax_train_new.py --epochs 1  # quick end-to-end check
python jax_test_new.py
python jax_train_new.py             # full training, up to 100 epochs
python jax_test_new.py              # final held-out evaluation
```

Training stops after 15 epochs without a validation-loss improvement. It saves
the best model to `checkpoints/model.msgpack`; the test script loads that file.
The 1-epoch run writes a temporary model for the quick check, and the full run
replaces it.

Each graph uses its actual nodes and directed edges. Node features are degree
and a constant 1; edge features are a constant 1. `Batch.from_data_list`
combines graphs for training. The model has three `GATConv` layers, mean
pooling, and a multilayer classifier with output classes 0–25.

The reported test accuracy comes from only 20 graphs and may vary substantially
with a different dataset or split. The supplied graph labels were precomputed;
this repository does not implement the r-robustness calculation itself.

The original `jax_dataset.py`, `jax_gnn.py`, `jax_train.py`, and `jax_test.py`
remain available unchanged. The `*_new.py` files contain the updated pipeline.

## Run the fixed original pipeline

This variant retains the original 50-node/2,500-edge padding, 80/20 split,
three-layer GAT, and 500-epoch limit. It fixes checkpoint consistency and test
batching, and saves the best validation checkpoint separately:

```bash
python jax_train_old_fixed.py
python jax_test_old_fixed.py
```

Its 20 evaluation graphs are also used for early stopping, matching the
original experiment design. Its reported metrics therefore are not from an
independent held-out test set. The checkpoint is
`checkpoints/old_fixed_model.msgpack`.
