"""Load graph records without introducing placeholder nodes or edges."""

import jax.numpy as jnp
from jraphx.data import Data
from sklearn.model_selection import train_test_split


def split_records(records):
    """Return reproducible 70/10/20 train, validation, and test splits."""
    train_val, test = train_test_split(records, test_size=0.2, random_state=42)
    train, validation = train_test_split(train_val, test_size=0.125, random_state=42)
    return train, validation, test


def create_graph(degree, edge_list, robustness):
    """Build a graph from its actual nodes and directed edges."""
    degree = jnp.asarray(degree, dtype=jnp.float32)
    x = jnp.stack((degree, jnp.ones_like(degree)), axis=1)
    edge_index = jnp.asarray(edge_list, dtype=jnp.int32).reshape(2, -1)
    edge_attr = jnp.ones((edge_index.shape[1], 1), dtype=jnp.float32)
    y = jnp.asarray([robustness], dtype=jnp.int32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def records_to_graphs(records):
    return [create_graph(d["degree"], d["edge_index"], d["robustness"]) for d in records]
