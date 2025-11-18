import jax
import flax.nnx as nnx
from jraphx.nn.conv import GATConv
from jraphx.nn.pool import global_mean_pool
import orbax.checkpoint as ocp
import os
import optax

## GNN model 
class GNN(nnx.Module):
    def __init__(self, in_features, hidden_features, num_classes, rngs, batchsize, heads=4):
        self.conv1 = GATConv(in_features, hidden_features // heads, heads=4, edge_dim =1, rngs=rngs)
        self.conv2 = GATConv(hidden_features, hidden_features // heads, heads=4,edge_dim =1, rngs=rngs)
        self.conv3 = GATConv(hidden_features, hidden_features // heads, heads=4, edge_dim =1,rngs=rngs)

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
        - x[:, 0]: Node degree. A value of 0 indicates a fake node.
        - x[:, 1]: Node validity flag. 1 for a real node, 0 for a fake node.

        2. edge_index: 
        - Contains all possible directed edges between nodes.

        3. edge_attr: 
        - Edge existence indicator: 1 if the edge exists, 0 otherwise.

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


def save_model(model, optimizer):
    """
    Save the model and optimizer states using Orbax's StandardCheckpointer.

    This function extracts the internal state of a Flax/NNX model and optimizer,
    converts them into pure Python dictionaries, and writes them to disk using
    Orbax checkpointing. Two checkpoint directories are created:

        - 'pure_dict_state'      : contains the model parameters/state
        - 'pure_dict_optimizer'  : contains the optimizer state

    Notes
    -----
    - If a directory with the same checkpoint name already exists, Orbax will
      raise an error unless `force=True` is explicitly provided.
    - The caller is responsible for ensuring that checkpoint directory names
      do not conflict with existing ones.

    Parameters
    ----------
    model : nnx.Module
        The model whose parameters and state should be saved.

    optimizer : nnx.Optimizer or optax.GradientTransformation
        The optimizer whose state should be saved.

    """
    with ocp.StandardCheckpointer() as ckptr:
        _, state = nnx.split(model)
        pure_dict_state = nnx.to_pure_dict(state)

        # saving the model and optimizer parameters
        ckptr.save(os.path.abspath('pure_dict_state2'), pure_dict_state)
        ckptr.save(os.path.abspath('pure_dict_optimzier2'), nnx.state(optimizer), force=True)

def load_model(batchsize):
    """
    Load a previously checkpointed GNN model and optimizer state.

    This function reconstructs the model and optimizer from the pure-dictionary
    checkpoints saved by `save_model()`. Because NNX separates the static
    computation graph from the mutable state, this function uses abstract
    shapes (via `nnx.eval_shape`) to rebuild the correct state structure before
    loading the serialized checkpoint dictionaries.

    Workflow
    --------
    1. Build an abstract model (with no actual parameter values).
    2. Load the checkpointed model state (`pure_dict_state`) and insert it
       into the abstract state tree.
    3. Reconstruct the full GNN model using the stored parameters.
    4. Build an abstract optimizer to obtain the correct optimizer state tree.
    5. Load the checkpointed optimizer state (`pure_dict_optimizer`).
    6. Initialize a real optimizer instance and update it with the restored state.

    Parameters
    ----------
    batchsize : int
        The batch size used when constructing the GNN model during training
        (must match the saved model's configuration).

    Returns
    -------
    model : nnx.Module
        The reconstructed GNN model containing the restored parameters.
    optimizer : nnx.Module
        The reconstructed optimizer containing the restored parameters.
    """

    def build_model():
        return GNN(
            in_features=2,
            hidden_features=100,
            num_classes=26,
            rngs=nnx.Rngs(42),  
            batchsize=batchsize
        )

    # load the params
    # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
    checkpointer = ocp.StandardCheckpointer()
    restored_pure_dict = checkpointer.restore(os.path.abspath('pure_dict_state'))
    abstract_model = nnx.eval_shape(build_model)
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    model = nnx.merge(graphdef, abstract_state)

    # Build an abstract optimizer to generate an abstract state tree
    abstract_optimizer = nnx.eval_shape(
        lambda: nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
    )
    abstractOptStateTree = nnx.state(abstract_optimizer)

    # Restore optimizer state
    optimizerState = checkpointer.restore(
        os.path.abspath('pure_dict_optimzier'),
        abstractOptStateTree
    )

    # Build a real optimizer instance and load its state
    optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
    nnx.update(optimizer, optimizerState)
    return model, optimizer