"""Communication adjacency functions shared by planning and tests."""

from __future__ import annotations

import jax.numpy as jnp


def smooth_adjacency(positions, radius, sigma=None):
    """Return equation-(7) edge weights for one graph or a batch of graphs.

    ``positions`` has shape ``(..., robots, dimensions)`` and the result has
    shape ``(..., robots, robots)``.  The default ``sigma = R**4 / log(2)``
    makes every off-diagonal, zero-distance edge have weight one.

    The positive part is applied before ``expm1``.  Consequently, distances
    outside the communication radius never evaluate the potentially large
    inactive exponential branch, including under JIT tracing.
    """
    positions = jnp.asarray(positions)
    radius = jnp.asarray(radius, dtype=positions.dtype)
    if sigma is None:
        sigma = radius**4 / jnp.log(jnp.asarray(2.0, dtype=positions.dtype))
    sigma = jnp.asarray(sigma, dtype=positions.dtype)

    displacement = positions[..., :, None, :] - positions[..., None, :, :]
    distance_squared = jnp.sum(displacement**2, axis=-1)
    radius_squared = radius**2
    supported_gap = jnp.maximum(radius_squared - distance_squared, 0.0)
    weights = jnp.expm1(supported_gap**2 / sigma)
    diagonal = jnp.eye(positions.shape[-2], dtype=bool)
    return jnp.where(diagonal, jnp.zeros((), dtype=weights.dtype), weights)


def hard_adjacency(positions, radius):
    """Return the existing binary radius graph with a zero diagonal."""
    positions = jnp.asarray(positions)
    displacement = positions[..., :, None, :] - positions[..., None, :, :]
    distance_squared = jnp.sum(displacement**2, axis=-1)
    diagonal = jnp.eye(positions.shape[-2], dtype=bool)
    return ((distance_squared <= radius**2) & ~diagonal).astype(positions.dtype)


def adjacency(positions, radius, mode="hard", sigma=None):
    """Select the hard baseline or equation-(7) communication weights."""
    if mode == "hard":
        return hard_adjacency(positions, radius)
    if mode == "smooth":
        return smooth_adjacency(positions, radius, sigma)
    raise ValueError("connectivity mode must be 'hard' or 'smooth'")
