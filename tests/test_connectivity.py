import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jraphx.nn.conv import GATv2Conv

from gnn.models.weighted_gatv2 import WeightedGATv2Conv, weighted_attention_softmax
from planning.connectivity import smooth_adjacency
from planning.svgd_cp_planner import PlannerConfig, _smooth_cost_single


class SmoothAdjacencyTests(unittest.TestCase):
    def test_equation_seven_values_diagonal_and_symmetry(self):
        radius = 1.0
        epsilon = 1e-4
        positions = jnp.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [radius - epsilon, 0.0],
            [radius, 0.0],
            [radius + epsilon, 0.0],
        ])
        weights = np.asarray(smooth_adjacency(positions, radius))
        self.assertAlmostEqual(weights[0, 1], 1.0, places=6)
        self.assertGreater(weights[0, 2], 0.0)
        self.assertAlmostEqual(weights[0, 3], 0.0, places=7)
        self.assertEqual(weights[0, 4], 0.0)
        np.testing.assert_array_equal(np.diag(weights), np.zeros(5))
        np.testing.assert_allclose(weights, weights.T, rtol=0, atol=0)

    def test_batched_jit_gradients_are_finite(self):
        positions = jnp.asarray([
            [[0.0, 0.0], [0.4, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [1.0e6, 0.0]],
        ])
        objective = jax.jit(lambda value: smooth_adjacency(value, 1.0).sum())
        gradient = jax.jit(jax.grad(objective))(positions)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        np.testing.assert_allclose(np.asarray(gradient)[1, 2], 0.0, atol=0.0)

    def test_gradient_matches_finite_difference_away_from_boundary(self):
        def weight_at(distance):
            points = jnp.asarray([[0.0, 0.0], [distance, 0.0]])
            return smooth_adjacency(points, 1.0)[0, 1]

        distance = 0.4
        automatic = float(jax.grad(weight_at)(distance))
        step = 1e-3
        finite_difference = float((weight_at(distance + step) - weight_at(distance - step)) / (2 * step))
        self.assertAlmostEqual(automatic, finite_difference, places=3)

    def test_gradient_is_zero_at_zero_boundary_and_outside(self):
        def weight_at(distance):
            points = jnp.asarray([[0.0, 0.0], [distance, 0.0]])
            return smooth_adjacency(points, 1.0)[0, 1]

        for distance in (0.0, 1.0, 1.1):
            with self.subTest(distance=distance):
                self.assertEqual(float(jax.grad(weight_at)(distance)), 0.0)

    def test_gradient_reaches_smooth_planning_objective(self):
        start = jnp.asarray([[0.0, 0.0], [0.5, 0.0]])
        waypoints = start[None]
        config = PlannerConfig(
            communication_radius=1.0,
            minimum_separation=0.1,
            maximum_speed=1.0,
            sampling_time=1.0,
            required_robustness=1,
            connectivity_mode="smooth",
            goal_weight=0.0,
            path_weight=0.0,
            collision_weight=0.0,
            speed_weight=0.0,
            degree_weight=1.0,
            cp_weight=0.0,
        )
        objective = jax.jit(lambda value: _smooth_cost_single(value, start, start, config))
        gradient = np.asarray(jax.grad(objective)(waypoints))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(np.linalg.norm(gradient), 0.0)


class WeightedGATv2Tests(unittest.TestCase):
    def test_zero_weight_large_logits_do_not_underflow_active_attention(self):
        logits = jnp.asarray([[10000., 10000.], [0., 0.]])
        targets = jnp.asarray([0, 0])
        weights = jnp.asarray([0., 1.])
        value = weighted_attention_softmax(logits, targets, weights, 1)
        np.testing.assert_allclose(value, [[0., 0.], [0.5, 0.5]])
        gradient = jax.grad(lambda w: weighted_attention_softmax(
            logits, targets, w, 1)[1, 0])(weights)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())

    def test_binary_edges_match_upstream_layer(self):
        kwargs = dict(in_features=2, out_features=2, heads=2, edge_dim=1)
        original = GATv2Conv(**kwargs, rngs=nnx.Rngs(7))
        weighted = WeightedGATv2Conv(**kwargs, rngs=nnx.Rngs(7))
        x = jnp.asarray([[1.0, 0.2], [0.3, 1.1], [0.7, -0.4]])
        edges = jnp.asarray([[0, 1, 1, 2], [1, 0, 2, 1]])
        attributes = jnp.ones((4, 1))
        np.testing.assert_allclose(
            np.asarray(weighted(x, edges, attributes)),
            np.asarray(original(x, edges, attributes)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_continuous_weights_change_messages_and_are_differentiable(self):
        layer = WeightedGATv2Conv(
            2, 2, heads=2, edge_dim=1, rngs=nnx.Rngs(11)
        )
        x = jnp.asarray([[1.0, 0.2], [0.3, 1.1], [0.7, -0.4]])
        edges = jnp.asarray([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])

        def output_sum(attributes):
            return layer(x, edges, attributes).sum()

        ones = jnp.ones((6, 1))
        varied = ones.at[0, 0].set(0.25)
        self.assertFalse(np.allclose(np.asarray(layer(x, edges, ones)),
                                     np.asarray(layer(x, edges, varied))))
        gradient = np.asarray(jax.grad(output_sum)(varied))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(np.linalg.norm(gradient), 0.0)


if __name__ == "__main__":
    unittest.main()
