import flax.nnx as nnx
from flax import serialization
from jraphx.nn.conv import GATConv
from jraphx.nn.pool import global_mean_pool
import os
from pathlib import Path

## GNN model 
class GNN(nnx.Module):
    def __init__(self, in_features, hidden_features, num_classes, rngs, batchsize, heads=4):
        self.conv1 = GATConv(in_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)
        self.conv2 = GATConv(hidden_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)
        self.conv3 = GATConv(hidden_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)

        self.norm1 = nnx.LayerNorm(hidden_features, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_features, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_features, rngs=rngs)

        # Final MLP head
        self.linear1 = nnx.Linear(hidden_features, hidden_features // 2, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_features // 2, hidden_features // 2, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_features // 2, num_classes, rngs=rngs)
        self.batchsize = batchsize



    
    def __call__(self, x, edge_index, edge_attr, batch):
        """
        This model takes four inputs for prediction:

        1. x:
        - x[:, 0]: Node degree.
        - x[:, 1]: Constant 1 for each real node.

        2. edge_index: 
        - Contains the graph's actual directed edges.

        3. edge_attr: 
        - Constant 1 for each actual edge.

        4. batch:
        - Batch assignment vector used for mini-batching.

        """
        x = self.norm1(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.norm2(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = self.norm3(self.conv3(x, edge_index, edge_attr=edge_attr))

        x = global_mean_pool(x, batch,size=self.batchsize)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)

        return x


CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "model.msgpack"


def save_model(model):
    """Save model state for later evaluation."""
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    state_bytes = serialization.to_bytes(nnx.to_pure_dict(nnx.state(model)))
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary_path.write_bytes(state_bytes)
    os.replace(temporary_path, CHECKPOINT_PATH)


def load_model(batchsize):
    """Restore the model parameters saved by :func:`save_model`."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No trained model at {CHECKPOINT_PATH}; run jax_train_new.py first")
    model = GNN(2, 100, 26, nnx.Rngs(42), batchsize=batchsize)
    state = nnx.state(model)
    restored = serialization.from_bytes(nnx.to_pure_dict(state), CHECKPOINT_PATH.read_bytes())
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(model, state)
    model.eval()
    return model
