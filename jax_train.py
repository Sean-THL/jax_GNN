import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from jax_dataset import create_graph
from jax_gnn import *
import orjson
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from jraphx.data import Batch

# -----------------------------
#  Model setup
# -----------------------------
num_classes = 26
BATCHSIZE= 10
model = GNN(in_features=2, hidden_features=100, num_classes=num_classes, rngs=nnx.Rngs(42), batchsize=BATCHSIZE)


# -----------------------------
#  Training utilities
# -----------------------------

def cross_entropy_loss(logits, labels):
    return (optax.softmax_cross_entropy(logits=logits, labels=labels)).mean()


optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)


## Train Step
@nnx.jit
def train_step(model, optimizer, x, edge_index, edge_attr, batch, y):
    model.train() 
    """JIT-compiled training step."""
    def loss_fn(model):
        train_predictions = model(x, edge_index, edge_attr, batch)
        train_targets = jax.nn.one_hot(y.flatten(), 26) 
        return cross_entropy_loss( train_predictions, train_targets )

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# Eval Step
@nnx.jit
def eval_step(model, x, edge_index, edge_attr, batch, y):
    model.eval() 
    predictions = model(x, edge_index, edge_attr, batch)
    train_targets = jax.nn.one_hot(y.flatten(), 26)
    loss = cross_entropy_loss( predictions, train_targets )
    return loss


# -----------------------------
# Data preparation
# -----------------------------

with open("testing_data.json", "r") as f:
    raw_data = orjson.loads(f.read()) 

print("Opening Done")

# Split the dataset into train and test dataset
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42) 

# Process the data to the form for jraphx
train_graphs = [create_graph(jnp.array(d["degree"]),jnp.array(d["edge_index"]), jnp.array(d["robustness"])) for d in train_data] 
test_graphs = [create_graph(jnp.array(d["degree"]),jnp.array(d["edge_index"]), jnp.array(d["robustness"])) for d in test_data]

print("Dataset Prep Done")


# -------------------- 
# Training loop  
# -------------------- 
NUM_EPOCHS = 500 
best_eval_loss = float("inf") 
patience = 0 
max_patience = 50 
counter = 0
train_losses = [] 
eval_losses = [] 
pt = tqdm(range(NUM_EPOCHS)) 

train_len = len(train_graphs)
test_len = len(test_graphs)

for epoch in pt: 

    # ---- Training ---- 
    batch_losses = [] 
    for i in range(0,train_len, BATCHSIZE):
        # Mini-batching
        batch = Batch.from_data_list(train_graphs[i:i+BATCHSIZE])
        loss = train_step(model, optimizer, batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y)
        batch_losses.append(loss)
    train_loss = jnp.mean(jnp.stack(batch_losses)) 
        

    # ---- Evaluation ---- 
    eval_batch_losses = []
    for i in range(0,test_len, BATCHSIZE):
        # Mini-batching
        batch = Batch.from_data_list(test_graphs[i:i+BATCHSIZE])
        loss = eval_step(model, batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y)
        eval_batch_losses.append(loss) 
    eval_loss = jnp.mean(jnp.stack(eval_batch_losses)) 
    train_losses.append(float(train_loss)) 
    eval_losses.append(float(eval_loss)) 
    pt.set_description(f"Train Loss: {train_loss:.4f} | Val Loss: {eval_loss:.4f}")

   # Early stopping 
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        counter = 0
    else:
        counter += 1
        if counter > max_patience:
            print("Early stopping")
            break

# checkpoint or save the model
save_model(model, optimizer)

# -----------------------------
# Plot curves
# -----------------------------
# plt.plot(train_losses, label="Train Loss")
# plt.plot(eval_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Cross Entropy Loss")
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.show()
