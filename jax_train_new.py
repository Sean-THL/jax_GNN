"""Train the new graph representation with gat."""
from jax_variants import train_main

if __name__ == "__main__":
    train_main("new", "gat")
