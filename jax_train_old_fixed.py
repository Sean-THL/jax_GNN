"""Train the original padded-graph model with reliable checkpointing."""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orjson
from flax import nnx, serialization
from jraphx.data import Batch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from jax_dataset import create_graph
from jax_gnn import GNN


BATCH_SIZE = 10
NUM_CLASSES = 26
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "old_fixed_model.msgpack"


@nnx.jit
def train_step(model, optimizer, x, edge_index, edge_attr, batch_index, y):
    model.train()

    def loss_fn(model):
        logits = model(x, edge_index, edge_attr, batch_index)
        targets = jax.nn.one_hot(y.reshape(-1), NUM_CLASSES)
        return optax.softmax_cross_entropy(logits, targets).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, edge_index, edge_attr, batch_index, y):
    model.eval()
    logits = model(x, edge_index, edge_attr, batch_index)
    targets = jax.nn.one_hot(y.reshape(-1), NUM_CLASSES)
    return optax.softmax_cross_entropy(logits, targets).mean()


def graph_from_record(record):
    return create_graph(record["degree"], record["edge_index"], record["robustness"])


def save_model(model):
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    model_bytes = serialization.to_bytes(nnx.to_pure_dict(nnx.state(model)))
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary_path.write_bytes(model_bytes)
    os.replace(temporary_path, CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    records = orjson.loads((PROJECT_DIR / "testing_data.json").read_bytes())
    train_records, evaluation_records = train_test_split(
        records, test_size=0.2, random_state=42
    )
    train_graphs = [graph_from_record(record) for record in train_records]
    evaluation_graphs = [graph_from_record(record) for record in evaluation_records]
    print("Using original padding: 50 nodes and 2,500 edge slots per graph")
    print("Training on 80 graphs; validating on 20 graphs")

    model = GNN(2, 100, NUM_CLASSES, nnx.Rngs(42), batchsize=BATCH_SIZE)
    optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    progress = tqdm(range(args.epochs))

    for epoch in progress:
        train_losses = []
        for i in range(0, len(train_graphs), BATCH_SIZE):
            batch = Batch.from_data_list(train_graphs[i:i + BATCH_SIZE])
            train_losses.append(train_step(
                model, optimizer, batch.x, batch.edge_index,
                batch.edge_attr, batch.batch, batch.y
            ))

        evaluation_losses = []
        for i in range(0, len(evaluation_graphs), BATCH_SIZE):
            batch = Batch.from_data_list(evaluation_graphs[i:i + BATCH_SIZE])
            evaluation_losses.append(eval_step(
                model, batch.x, batch.edge_index,
                batch.edge_attr, batch.batch, batch.y
            ))

        train_loss = float(jnp.mean(jnp.stack(train_losses)))
        evaluation_loss = float(jnp.mean(jnp.stack(evaluation_losses)))
        progress.set_description(
            f"Train Loss: {train_loss:.4f} | Val Loss: {evaluation_loss:.4f}"
        )

        if evaluation_loss < best_loss:
            best_loss = evaluation_loss
            best_state = nnx.to_pure_dict(nnx.state(model))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > args.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    if best_state is None:
        raise ValueError("No training epochs ran; pass --epochs 1 or more")

    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, best_state)
    nnx.update(model, state)
    save_model(model)
    print(f"Saved best model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
