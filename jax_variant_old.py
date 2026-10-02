"""Preserved shared pipeline before CPU batching and prefetch optimizations.

The standard training/testing entry points still use jax_variants.py.
This module retains the label-rounding fix and four model-module choices.
"""

import flax.nnx as nnx
from flax import serialization
import os
from pathlib import Path
from importlib import import_module

import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orjson
from jraphx.data import Batch
from jax_dataset import create_graph as padded_graph
from jax_dataset_new import create_graph as actual_graph
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent


def model_class(representation, convolution):
    modules = {
        ("original", "gat"): "jax_gnn",
        ("original", "gatv2"): "jax_gnn_gatv2",
        ("new", "gat"): "jax_gnn_new",
        ("new", "gatv2"): "jax_gnn_new_gatv2",
    }
    return import_module(modules[representation, convolution]).GNN


def checkpoint_path(representation, convolution):
    return PROJECT / "checkpoints" / f"{representation}_{convolution}.msgpack"


def save_checkpoint(model, path, metadata):
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(nnx.to_pure_dict(nnx.state(model))))
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def read_records(path, limit=None):
    records = orjson.loads(path.read_bytes())
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No records in {path}")
    normalized = 0
    for i, record in enumerate(records):
        label = float(record["robustness"])
        if not np.isfinite(label):
            raise ValueError(f"Record {i}: robustness must be finite")
        integer_label = round(label)
        if abs(label - integer_label) > 0.001 or not 0 <= integer_label < 26:
            raise ValueError(
                f"Record {i}: robustness {label} must be within 0.001 of an integer from 0 to 25"
            )
        normalized += label != integer_label
        record["robustness"] = integer_label
        n = len(record["degree"])
        edges = np.asarray(record["edge_index"])
        if not 1 <= n <= 50 or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f"Record {i}: invalid graph shape")
        if edges.size and (edges.min() < 0 or edges.max() >= n):
            raise ValueError(f"Record {i}: edge index outside node range")
    if normalized:
        print(f"Rounded {normalized} near-integer robustness labels in {path.name}", flush=True)
    return records


def batches(records, batch_size, representation, order=None):
    create = padded_graph if representation == "original" else actual_graph
    if order is None:
        order = range(len(records))
    for start in range(0, len(records), batch_size):
        selected = [records[int(i)] for i in order[start:start + batch_size]]
        graphs = [create(r["degree"], r["edge_index"], int(r["robustness"])) for r in selected]
        yield Batch.from_data_list(graphs), len(selected)


def batch_arrays(batch):
    return batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y


def loss_fn(model, x, edge_index, edge_attr, batch_index, y):
    logits = model(x, edge_index, edge_attr, batch_index)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y.reshape(-1)).mean()


@nnx.jit
def train_step(model, optimizer, *arrays):
    model.train()
    loss, grads = nnx.value_and_grad(loss_fn)(model, *arrays)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, *arrays):
    model.eval()
    return loss_fn(model, *arrays)


def train_main(representation, convolution):
    parser = argparse.ArgumentParser(description=f"Train {representation} {convolution}")
    parser.add_argument("--epochs", type=int, default=500 if representation == "original" else 100)
    parser.add_argument("--patience", type=int, default=50 if representation == "original" else 15)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-data", type=Path, default=PROJECT / "training_data.json")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_path(representation, convolution))
    parser.add_argument("--limit", type=int, help="Use the first N training records for a quick check")
    parser.add_argument("--no-jit", action="store_true", help="Avoid whole-step compilation for variable graph shapes")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or not 0 < args.validation_fraction < 1:
        parser.error("epochs, batch-size, patience must be positive; validation-fraction must be between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2")
    records = read_records(args.train_data, args.limit)
    train, validation = train_test_split(records, test_size=args.validation_fraction, random_state=args.seed)
    del records
    print(f"{representation} / {convolution}: {len(train)} training, {len(validation)} validation graphs", flush=True)
    print(f"Devices: {jax.devices()}; checkpoint: {args.checkpoint}", flush=True)
    model = model_class(representation, convolution)(2, 100, 26, nnx.Rngs(args.seed), args.batch_size)
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.Param)
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    stale = 0
    metadata = dict(representation=representation, convolution=convolution, seed=args.seed,
                    training_data=str(args.train_data.resolve()), training_graphs=len(train),
                    validation_graphs=len(validation), validation_fraction=args.validation_fraction,
                    batch_size=args.batch_size, learning_rate=args.learning_rate)
    with jax.disable_jit(args.no_jit):
        for epoch in range(args.epochs):
            total = 0.0
            for batch, count in tqdm(batches(train, args.batch_size, representation, rng.permutation(len(train))),
                                     total=(len(train)+args.batch_size-1)//args.batch_size, desc=f"Epoch {epoch+1}"):
                model.batchsize = count
                total += float(train_step(model, optimizer, *batch_arrays(batch))) * count
            validation_total = 0.0
            for batch, count in batches(validation, args.batch_size, representation):
                model.batchsize = count
                validation_total += float(eval_step(model, *batch_arrays(batch))) * count
            value = validation_total / len(validation)
            print(f"Epoch {epoch+1}: train loss {total/len(train):.4f}, validation loss {value:.4f}", flush=True)
            if value < best:
                best, stale = value, 0
                metadata.update(best_epoch=epoch+1, validation_loss=value)
                save_checkpoint(model, args.checkpoint, metadata)
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping")
                    break
    print(f"Saved best model: {args.checkpoint}")


def test_main(representation, convolution):
    parser = argparse.ArgumentParser(description=f"Test {representation} {convolution}")
    parser.add_argument("--test-data", type=Path, default=PROJECT / "testing_data.json")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_path(representation, convolution))
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.checkpoint.exists():
        parser.error(f"No checkpoint at {args.checkpoint}; run the corresponding training script first")
    meta_path = args.checkpoint.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    if (metadata["representation"], metadata["convolution"]) != (representation, convolution):
        parser.error("Checkpoint belongs to a different model variant")
    records = read_records(args.test_data)
    model = model_class(representation, convolution)(2, 100, 26, nnx.Rngs(42), args.batch_size)
    state = nnx.state(model)
    restored = serialization.from_bytes(nnx.to_pure_dict(state), args.checkpoint.read_bytes())
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    model.eval()
    predictions, targets = [], []
    for batch, count in batches(records, args.batch_size, representation):
        model.batchsize = count
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predictions.extend(np.asarray(jnp.argmax(logits, axis=-1)).tolist())
        targets.extend(np.asarray(batch.y).reshape(-1).tolist())
    predicted, actual = np.asarray(predictions), np.asarray(targets)
    error = actual - predicted
    print(f"Variant: {representation} / {convolution}; test graphs: {len(records)}")
    print("Exact robustness accuracy:", float(np.mean(predicted == actual)))
    print("Lower-bound accuracy:", float(np.mean(predicted <= actual)))
    print("Average deviation (actual - prediction):", float(np.mean(error)))
    print("Accuracy within ±1:", float(np.mean(np.abs(error) <= 1)))
    print("Accuracy within ±2:", float(np.mean(np.abs(error) <= 2)))
