"""Evaluate the fixed original pipeline on its 20-graph evaluation split."""

from pathlib import Path

import jax.numpy as jnp
import orjson
from flax import nnx, serialization
from jraphx.data import Batch
from sklearn.model_selection import train_test_split

from jax_dataset import create_graph
from jax_gnn import GNN


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "old_fixed_model.msgpack"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH}; run jax_train_old_fixed.py first"
        )

    records = orjson.loads((PROJECT_DIR / "testing_data.json").read_bytes())
    _, evaluation_records = train_test_split(records, test_size=0.2, random_state=42)
    graphs = [
        create_graph(record["degree"], record["edge_index"], record["robustness"])
        for record in evaluation_records
    ]
    batch = Batch.from_data_list(graphs)

    model = GNN(2, 100, 26, nnx.Rngs(42), batchsize=len(graphs))
    state = nnx.state(model)
    restored = serialization.from_bytes(
        nnx.to_pure_dict(state), CHECKPOINT_PATH.read_bytes()
    )
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    model.eval()

    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    predicted = jnp.argmax(logits, axis=-1)
    actual = batch.y.reshape(-1)
    error = actual - predicted

    print("Evaluation graphs: 20 (these were also used for early stopping)")
    print("Exact robustness accuracy:", float(jnp.mean(predicted == actual)))
    print("Lower-bound accuracy:", float(jnp.mean(predicted <= actual)))
    print("Average deviation (actual - prediction):", float(jnp.mean(error)))
    print("Accuracy within ±1:", float(jnp.mean(jnp.abs(error) <= 1)))
    print("Accuracy within ±2:", float(jnp.mean(jnp.abs(error) <= 2)))


if __name__ == "__main__":
    main()
