"""The original 80/20 training procedure, shared by GAT and GATv2.

This intentionally keeps the pasted script's data source, graph construction,
batching, training order, loss, early stopping, and final-weight saving. It is
separate from the four modern variants in jax_variants.py.
"""

import argparse
import json
import os
from importlib import import_module
from pathlib import Path

import flax.nnx as nnx
from flax import serialization
import jax
import jax.numpy as jnp
import optax
import orjson
from jraphx.data import Batch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from legacy.dataset import create_graph


PROJECT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 10
NUM_CLASSES = 26


def model_class(convolution):
    module = "legacy.models.gat" if convolution == "gat" else "legacy.models.gatv2"
    return import_module(module).GNN


def checkpoint_path(convolution):
    return PROJECT / "checkpoints" / f"historical_{convolution}.msgpack"


def graph_from_record(record):
    return create_graph(
        jnp.array(record["degree"]),
        jnp.array(record["edge_index"]),
        jnp.array(record["robustness"]),
    )


@nnx.jit
def train_step(model, optimizer, x, edge_index, edge_attr, batch_index, y):
    model.train()

    def loss_fn(model):
        logits = model(x, edge_index, edge_attr, batch_index)
        targets = jax.nn.one_hot(y.flatten(), NUM_CLASSES)
        return optax.softmax_cross_entropy(logits, targets).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, edge_index, edge_attr, batch_index, y):
    model.eval()
    logits = model(x, edge_index, edge_attr, batch_index)
    targets = jax.nn.one_hot(y.flatten(), NUM_CLASSES)
    return optax.softmax_cross_entropy(logits, targets).mean()


def save_final(model, optimizer, checkpoint, convolution, completed_epochs, batch_size):
    """Save final model and optimizer separately, without touching old paths."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    optimizer_path = checkpoint.with_name(checkpoint.stem + ".optimizer.msgpack")
    model_tmp = checkpoint.with_suffix(".tmp")
    optimizer_tmp = optimizer_path.with_suffix(".tmp")
    model_tmp.write_bytes(serialization.to_bytes(nnx.to_pure_dict(nnx.state(model))))
    optimizer_tmp.write_bytes(serialization.to_bytes(nnx.to_pure_dict(nnx.state(optimizer))))
    os.replace(model_tmp, checkpoint)
    os.replace(optimizer_tmp, optimizer_path)
    checkpoint.with_suffix(".json").write_text(json.dumps({
        "pipeline": "historical", "convolution": convolution,
        "data": str((PROJECT / "testing_data.json").resolve()),
        "train_graphs": 80, "validation_graphs": 20,
        "batch_size": batch_size, "seed": 42, "learning_rate": 0.005,
        "saved_weights": "final", "completed_epochs": completed_epochs,
    }, indent=2))


def train_main(convolution):
    parser = argparse.ArgumentParser(description=f"Historical original {convolution} training")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_path(convolution))
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive; patience cannot be negative")

    raw_data = orjson.loads((PROJECT / "testing_data.json").read_bytes())
    train_data, validation_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    train_graphs = [graph_from_record(record) for record in train_data]
    validation_graphs = [graph_from_record(record) for record in validation_data]
    print(f"Historical original / {convolution}: {len(train_graphs)} training, "
          f"{len(validation_graphs)} validation graphs", flush=True)
    print(f"Devices: {jax.devices()}; final checkpoint: {args.checkpoint}", flush=True)

    model = model_class(convolution)(2, 100, NUM_CLASSES, nnx.Rngs(42), args.batch_size)
    optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
    best_eval_loss = float("inf")
    counter = 0
    completed_epochs = 0

    for epoch in tqdm(range(args.epochs)):
        batch_losses = []
        for i in range(0, len(train_graphs), args.batch_size):
            selected = train_graphs[i:i + args.batch_size]
            batch = Batch.from_data_list(selected)
            model.batchsize = len(selected)
            batch_losses.append(train_step(
                model, optimizer, batch.x, batch.edge_index,
                batch.edge_attr, batch.batch, batch.y,
            ))
        train_loss = jnp.mean(jnp.stack(batch_losses))

        eval_batch_losses = []
        for i in range(0, len(validation_graphs), args.batch_size):
            selected = validation_graphs[i:i + args.batch_size]
            batch = Batch.from_data_list(selected)
            model.batchsize = len(selected)
            eval_batch_losses.append(eval_step(
                model, batch.x, batch.edge_index,
                batch.edge_attr, batch.batch, batch.y,
            ))
        eval_loss = jnp.mean(jnp.stack(eval_batch_losses))
        completed_epochs = epoch + 1
        print(f"Epoch {completed_epochs}: train loss {float(train_loss):.4f}, "
              f"validation loss {float(eval_loss):.4f}", flush=True)

        if float(eval_loss) < best_eval_loss:
            best_eval_loss = float(eval_loss)
            counter = 0
        else:
            counter += 1
            if counter > args.patience:
                print("Early stopping", flush=True)
                break

    # The pasted baseline saves the final model, not the best validation epoch.
    save_final(model, optimizer, args.checkpoint, convolution, completed_epochs, args.batch_size)
    print(f"Saved final model and optimizer: {args.checkpoint}", flush=True)


def test_main(convolution):
    parser = argparse.ArgumentParser(description=f"Evaluate historical original {convolution}")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_path(convolution))
    parser.add_argument("--batch-size", type=int, help="Defaults to the checkpoint's training batch size")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error(f"No checkpoint at {args.checkpoint}")
    metadata_path = args.checkpoint.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    batch_size = args.batch_size if args.batch_size is not None else metadata.get("batch_size", BATCH_SIZE)
    if batch_size < 1:
        parser.error("batch-size must be positive")

    raw_data = orjson.loads((PROJECT / "testing_data.json").read_bytes())
    _, validation_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    graphs = [graph_from_record(record) for record in validation_data]
    model = model_class(convolution)(2, 100, NUM_CLASSES, nnx.Rngs(42), batch_size)
    state = nnx.state(model)
    restored = serialization.from_bytes(nnx.to_pure_dict(state), args.checkpoint.read_bytes())
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    model.eval()

    predictions, targets = [], []
    for i in range(0, len(graphs), batch_size):
        selected = graphs[i:i + batch_size]
        batch = Batch.from_data_list(selected)
        model.batchsize = len(selected)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predictions.extend(jnp.argmax(logits, axis=-1).tolist())
        targets.extend(batch.y.reshape(-1).tolist())
    predicted, actual = jnp.asarray(predictions), jnp.asarray(targets)
    error = actual - predicted
    print(f"Historical original / {convolution}; validation graphs: {len(graphs)}")
    print("These graphs were used for early stopping; this is not independent test accuracy.")
    print("Exact robustness accuracy:", float(jnp.mean(predicted == actual)))
    print("Lower-bound accuracy:", float(jnp.mean(predicted <= actual)))
    print("Average deviation (actual - prediction):", float(jnp.mean(error)))
    print("Accuracy within ±1:", float(jnp.mean(jnp.abs(error) <= 1)))
    print("Accuracy within ±2:", float(jnp.mean(jnp.abs(error) <= 2)))
