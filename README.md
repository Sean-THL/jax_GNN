# jax_GNN

A clean and modular **Graph Neural Network (GNN)** training framework built using **JAX**, **Flax/NNX**, and **jraphx**.  
This project includes a working implementation of **GATv2** based on the official `jraphx` library by [DBraun](https://github.com/DBraun/jraphx), and supports fast JIT-compiled training pipelines for graph classification tasks.

---

## Features

- **GATv2 layer** (attention-based GNN) from [this work](https://arxiv.org/abs/2105.14491) implemented via `jraphx.nn.GATv2Conv`
- Fully JAX/NNX-compatible model class with modular design
- Efficient graph batching using `jraphx.data.Batch`
- JIT-compiled training & evaluation steps
- Early stopping with validation monitoring
- Checkpointing through Orbax (`StandardCheckpointer`)
- Clean separation of dataset processing, model definition, and training script
- Contains a dataset of 100 randomly generated graphs of sizes from 1 to 50 nodes with r-robustness (defined in [here](https://ieeexplore.ieee.org/document/6481629)) 

---

## Requirements
Your environment must use:
```
Python 3.12.3
```

Install dependencies:

```
pip install -r requirements.txt
```

## Project Structure
```
jax_GNN/
├── jax_dataset.py       # Data loading and graph preprocessing
├── jax_gnn.py           # GATv2-style GNN implementation
├── jax_train.py         # Training script
├── jax_test.py          # Inference / testing script
├── testing_data.json    # raw dataset that contains 100 graphs with computed r-robustness
├── pure_dict_state      # saved model parameters
├── pure_dict_optimizer  # saved optimizer parameters 
├── requirements.txt
└── README.md
```

## Example Dataset

The repository includes testing_data.json, which contains 100 randomly generated graphs. Each graph entry includes:

- Node counts: ranging from 1 to 50 
- Degree sequence: the degrees of all nodes in the graph
- Directed edges: list of edges representing the graph connectivity
- r-robustness: precomputed robustness value for the graph
  
This dataset is used to demonstrate training and evaluation of the GNN.
