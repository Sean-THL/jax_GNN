import jax.numpy as jnp
import flax.nnx as nnx
from jax_dataset import create_graph
from jax_gnn import *
import orjson



# Open the dataset of interest
with open("testing_data.json", "r") as f:
    test_data = orjson.loads(f.read()) 


# Parameters
BATHCSIZE = 100
num_classes = 26


# Define the model init function
def build_model():
    return GNN(
        in_features=2,
        hidden_features=100,
        num_classes=26,
        rngs=nnx.Rngs(42),  
        batchsize=BATHCSIZE
    )

# Process the dataset for jraphx
test_graphs = [create_graph(jnp.array(d["degree"]),jnp.array(d["edge_index"]), jnp.array(d["robustness"])) for d in test_data]

# Loading the saved model
model, _ = load_model(BATHCSIZE)


# -------------------- 
# Checking the accuracy 
# -------------------- 

total = len(test_graphs)
correct = 0.0
lower_bound = 0.0
deviation = 0.0
correct_by_pm1 = 0.0
correct_by_pm2 = 0.0

for batch in test_graphs:
    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    actual = batch.y
    deviation += (actual - jnp.argmax(pred, axis=-1) ).sum().item()

    correct += jnp.sum(jnp.argmax(pred, axis=-1) == actual)
    lower_bound += jnp.sum(jnp.argmax(pred, axis=-1) <= actual)
    correct_by_pm1 += jnp.sum((jnp.abs(jnp.argmax(pred, axis=-1) - actual)) <= 1)
    correct_by_pm2 += jnp.sum((jnp.abs(jnp.argmax(pred, axis=-1) - actual)) <= 2)

print("(1) Exact Robustness Accuracy:", correct/total)
print("(2) Lower Bounding Accuracy:", lower_bound/total)
print("(3) Average Deviation (actual - prediction):", deviation/total)
print("(4) Accuracy within ±1:", correct_by_pm1/total)
print("(5) Accuracy within ±2:", correct_by_pm2/total)