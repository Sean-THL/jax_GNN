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
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
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


def prepare_records(records):
    """Keep compact graph arrays on the CPU, never pad the entire dataset."""
    for record in records:
        degree = np.asarray(record["degree"], dtype=np.float32)
        record["features"] = np.stack((degree, np.ones_like(degree)), axis=1)
        record["edge_index"] = np.asarray(record["edge_index"], dtype=np.int32)
        # Discard unused fields and large Python lists after conversion.
        for key in list(record):
            if key not in ("features", "edge_index", "robustness"):
                del record[key]
    return records


def batches(records, batch_size, representation, order=None, *, pad_new=True):
    """Assemble host batches, bounding JIT shapes for the new representation.

    Padding edges connect only to a disconnected dummy node. Padding nodes use
    a graph ID outside the pooling range, so they never enter graph means.
    """
    if order is None:
        order = range(len(records))
    padded = representation == "new" and pad_new
    # One node capacity per dataset/batch size, with space for a dummy node.
    node_capacity = batch_size * max((len(r["features"]) for r in records), default=0) + 1 if padded else None
    for start in range(0, len(records), batch_size):
        selected = [records[int(i)] for i in order[start:start + batch_size]]
        counts = [50 if representation == "original" else len(r["features"]) for r in selected]
        edges = [2500 if representation == "original" else r["edge_index"].shape[1] for r in selected]
        node_count_total, edge_count_total = sum(counts), sum(edges)
        # Power-of-two edge buckets reuse executables without padding every
        # sparse batch to the densest possible graph.
        edge_capacity = edge_count_total
        if padded:
            edge_capacity = max(256, 1 << (max(1, edge_count_total) - 1).bit_length())
        x = np.zeros((node_capacity if padded else node_count_total, 2), dtype=np.float32)
        edge_index = np.full((2, edge_capacity), node_count_total if padded else 0, dtype=np.int32)
        edge_attr = np.zeros((edge_capacity, 1), dtype=np.float32)
        batch_index = np.full(len(x), len(selected), dtype=np.int32)
        batch_index[:node_count_total] = np.repeat(np.arange(len(selected), dtype=np.int32), counts)
        y = np.asarray([r["robustness"] for r in selected], dtype=np.int32)
        node_offset = edge_offset = 0
        for record, node_count, edge_count in zip(selected, counts, edges):
            n, e = len(record["features"]), record["edge_index"].shape[1]
            x[node_offset:node_offset+n] = record["features"]
            # Preserve original placeholder edges at local node zero.
            edge_index[:, edge_offset:edge_offset+edge_count] = node_offset
            edge_index[:, edge_offset:edge_offset+e] = record["edge_index"] + node_offset
            edge_attr[edge_offset:edge_offset+e] = 1
            node_offset += node_count
            edge_offset += edge_count
        yield SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr,
                              batch=batch_index, y=y), len(selected)


def device_batches(records, batch_size, representation, order=None):
    """Build the next host batch while the current batch runs on the device."""
    iterator = iter(batches(records, batch_size, representation, order))
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(next, iterator, None)
        while True:
            item = pending.result()
            if item is None:
                break
            pending = executor.submit(next, iterator, None)
            batch, count = item
            arrays = jax.device_put(batch_arrays(batch))
            yield SimpleNamespace(**dict(zip(("x", "edge_index", "edge_attr", "batch", "y"), arrays))), count


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
    parser.add_argument("--no-jit", action="store_true", help="Disable JIT for debugging (usually slower)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or not 0 < args.validation_fraction < 1:
        parser.error("epochs, batch-size, patience must be positive; validation-fraction must be between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2")
    records = read_records(args.train_data, args.limit)
    records = prepare_records(records)
    train, validation = train_test_split(records, test_size=args.validation_fraction, random_state=args.seed)
    del records
    print(f"{representation} / {convolution}: {len(train)} training, {len(validation)} validation graphs", flush=True)
    print(f"Devices: {jax.devices()}; checkpoint: {args.checkpoint}", flush=True)
    if representation == "new":
        print("Using disconnected batch padding and power-of-two edge buckets to reuse JIT compilations", flush=True)
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
            epoch_start = perf_counter()
            total = jnp.asarray(0.0)
            for batch, count in tqdm(device_batches(train, args.batch_size, representation, rng.permutation(len(train))),
                                     total=(len(train)+args.batch_size-1)//args.batch_size, desc=f"Epoch {epoch+1}"):
                model.batchsize = count
                total += train_step(model, optimizer, *batch_arrays(batch)) * count
            train_value = float(total) / len(train)
            train_seconds = perf_counter() - epoch_start
            validation_start = perf_counter()
            validation_total = jnp.asarray(0.0)
            for batch, count in tqdm(device_batches(validation, args.batch_size, representation),
                                     total=(len(validation)+args.batch_size-1)//args.batch_size, desc="Validation"):
                model.batchsize = count
                validation_total += eval_step(model, *batch_arrays(batch)) * count
            value = float(validation_total) / len(validation)
            validation_seconds = perf_counter() - validation_start
            print(f"Epoch {epoch+1}: train loss {train_value:.4f}, validation loss {value:.4f}; "
                  f"training {train_seconds:.1f}s, validation {validation_seconds:.1f}s", flush=True)
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
    records = prepare_records(read_records(args.test_data))
    model = model_class(representation, convolution)(2, 100, 26, nnx.Rngs(42), args.batch_size)
    state = nnx.state(model)
    restored = serialization.from_bytes(nnx.to_pure_dict(state), args.checkpoint.read_bytes())
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    model.eval()
    predictions, targets = [], []
    for batch, count in device_batches(records, args.batch_size, representation):
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
