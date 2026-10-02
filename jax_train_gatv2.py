"""Train original GATv2 with the historical baseline procedure."""
from jax_historical_baseline import train_main

if __name__ == "__main__":
    train_main("gatv2")
