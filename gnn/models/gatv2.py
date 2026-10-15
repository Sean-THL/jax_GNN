"""GATv2Conv model; checkpoints are managed by jax_variants.py."""
import flax.nnx as nnx
from jraphx.nn.conv import GATv2Conv
from jraphx.nn.pool import global_mean_pool

## GNN model 
class GNN(nnx.Module):
    def __init__(self, in_features, hidden_features, num_classes, rngs, batchsize, heads=4):
        self.conv1 = GATv2Conv(in_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)
        self.conv2 = GATv2Conv(hidden_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)
        self.conv3 = GATv2Conv(hidden_features, hidden_features // heads, heads=heads, edge_dim=1, rngs=rngs)

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

