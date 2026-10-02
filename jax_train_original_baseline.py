"""Train the historical padded original GAT baseline."""
from jax_historical_baseline import train_main

if __name__ == "__main__":
    train_main("gat")
