"""Evaluate original GAT on the historical validation split."""
from jax_historical_baseline import test_main

if __name__ == "__main__":
    test_main("gat")
