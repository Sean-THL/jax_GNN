import jax
import jax.numpy as jnp
from jraphx.data import Data
from flax.struct import dataclass

N_MAX = 50  # maximum nodes per graph


@dataclass
class GraphData(Data):
    x: jnp.ndarray          # Node features, shape = (N_MAX, num_node_features)
    edge_index: jnp.ndarray # Edge indices, shape = (2, N_MAX * N_MAX)
    edge_attr: jnp.ndarray  # Edge attributes, shape = (N_MAX * N_MAX, num_edge_features)
    y: jnp.ndarray          # Labels, shape = (1,)
    batch: jnp.ndarray      # Batch vector, shape = (N_MAX,)


# Precompute full edge_index once
DEGREE_PAD = jnp.zeros((N_MAX, 2), dtype=jnp.float32)
E_MAX = N_MAX * N_MAX
EDGE_PAD = jnp.zeros((2, E_MAX), dtype=jnp.int32)



def create_graph(degree, edge_list, robustness):
    """
    This function takes 
    ############################################
    1. degree: degrees of nodes in a graph 

    2. edge_list: directed edges of nodes in a graph

    3. robustness: robustness of a graph
    ############################################
    
    and creates a graph dataset with:

    1. x: tensor of shape (N_MAX, 2)
    - x[:, 0]: Node degree. A value of 0 indicates a fake node.
    - x[:, 1]: Node validity flag. 1 for a real node, 0 for a fake node.

    2. edge_index: tensor of shape (2, E_MAX)
    - Contains all possible directed edges between nodes.

    3. edge_attr: tensor of shape (E_MAX, 1)
    - Edge existence indicator: 1 if the edge exists, 0 otherwise.

    4. y: tensor of shape (1,)
    - Contains the robustness of the given graph 

    """
    n = len(degree)
    len_edges = len(edge_list[0])
    edge_list = jnp.array(edge_list, dtype=jnp.int32)

    # --- Node features: degree + binary flag ---
    x = DEGREE_PAD
    x = x.at[:n, 0].set(jnp.array(degree, dtype=jnp.float32))
    x = x.at[:n, 1].set(1.0)  # real node flag

    # --- Labels ---
    y = jnp.array([robustness], dtype=jnp.int32)

    edge_index = EDGE_PAD
    edge_index = edge_index.at[:, :len_edges].set(edge_list)

    edge_attr = jnp.zeros((E_MAX,), dtype=jnp.float32)

    # scatter 1s into correct edge positions
    edge_attr = edge_attr.at[:len_edges].set(1.0)

    # reshape into (num_edges, 1)
    edge_attr = edge_attr.reshape(-1, 1)

    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y    )