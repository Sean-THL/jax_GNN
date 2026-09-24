"""Evaluate the trained model on the held-out 20-graph test split."""

from pathlib import Path

import jax.numpy as jnp
import orjson
from jraphx.data import Batch

from jax_dataset_new import records_to_graphs, split_records
from jax_gnn_new import load_model


def main():
    records = orjson.loads((Path(__file__).resolve().parent / "testing_data.json").read_bytes())
    _, _, test_records = split_records(records)
    graphs = records_to_graphs(test_records)
    batch = Batch.from_data_list(graphs)
    model = load_model(batchsize=len(graphs))

    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    predicted = jnp.argmax(logits, axis=-1)
    actual = batch.y.reshape(-1)
    error = actual - predicted

    print(f"Held-out graphs: {len(graphs)}")
    print("Exact robustness accuracy:", float(jnp.mean(predicted == actual)))
    print("Lower-bound accuracy:", float(jnp.mean(predicted <= actual)))
    print("Average deviation (actual - prediction):", float(jnp.mean(error)))
    print("Accuracy within ±1:", float(jnp.mean(jnp.abs(error) <= 1)))
    print("Accuracy within ±2:", float(jnp.mean(jnp.abs(error) <= 2)))


if __name__ == "__main__":
    main()
