"""Train a graph classifier and save the best validation checkpoint."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orjson
from flax import nnx
from jraphx.data import Batch
from tqdm import trange

from jax_dataset_new import records_to_graphs, split_records
from jax_gnn_new import GNN, save_model


BATCH_SIZE = 10
NUM_CLASSES = 26
DATA_PATH = Path(__file__).resolve().parent / "testing_data.json"


@nnx.jit
def train_step(model, optimizer, x, edge_index, edge_attr, batch_index, y):
    model.train()

    def loss_fn(model):
        logits = model(x, edge_index, edge_attr, batch_index)
        labels = jax.nn.one_hot(y.reshape(-1), NUM_CLASSES)
        return optax.softmax_cross_entropy(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, edge_index, edge_attr, batch_index, y):
    model.eval()
    logits = model(x, edge_index, edge_attr, batch_index)
    labels = jax.nn.one_hot(y.reshape(-1), NUM_CLASSES)
    return optax.softmax_cross_entropy(logits, labels).mean()


def make_batches(graphs):
    if len(graphs) % BATCH_SIZE:
        raise ValueError("Every split must have a whole number of batches")
    return [Batch.from_data_list(graphs[i:i + BATCH_SIZE])
            for i in range(0, len(graphs), BATCH_SIZE)]


def batch_arrays(batch):
    return batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    records = orjson.loads(DATA_PATH.read_bytes())
    train_records, validation_records, _ = split_records(records)
    train_batches = make_batches(records_to_graphs(train_records))
    validation_batches = make_batches(records_to_graphs(validation_records))
    print(f"Training on {len(train_records)} graphs; validating on {len(validation_records)}")

    model = GNN(2, 100, NUM_CLASSES, nnx.Rngs(42), batchsize=BATCH_SIZE)
    optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in trange(args.epochs):
        train_loss = float(jnp.mean(jnp.stack([
            train_step(model, optimizer, *batch_arrays(batch)) for batch in train_batches
        ])))
        validation_loss = float(jnp.mean(jnp.stack([
            eval_step(model, *batch_arrays(batch)) for batch in validation_batches
        ])))
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, validation loss {validation_loss:.4f}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = nnx.to_pure_dict(nnx.state(model))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping")
                break

    if best_state is None:
        raise ValueError("No training epochs ran; pass --epochs 1 or more")
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, best_state)
    nnx.update(model, state)
    save_model(model)
    print("Saved the model with the lowest validation loss")


if __name__ == "__main__":
    main()
