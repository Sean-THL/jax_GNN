import unittest
from unittest.mock import patch
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from gnn.models.gatv2 import GNN
from planning.svgd_cp_planner import PlannerConfig, communication_records, make_smooth_cp_value_and_grad, plan


class AutodiffPlannerTests(unittest.TestCase):
    def test_full_gnn_position_gradient_matches_finite_difference(self):
        model = GNN(2, 8, 26, nnx.Rngs(7), batchsize=3)
        model.eval()
        config = PlannerConfig(1., 0.1, 1., 1., 20, cp_gradient="autodiff")
        evaluate = make_smooth_cp_value_and_grad(model, 0., config)
        particles = jnp.asarray([[[[0., 0.], [0.4, 0.1], [0.2, 0.6]]]])
        cost, gradient = evaluate(particles)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-5)
        index = np.unravel_index(np.argmax(np.abs(np.asarray(gradient))), gradient.shape)
        step = 1e-3
        plus = evaluate(particles.at[index].add(step))[0].sum()
        minus = evaluate(particles.at[index].add(-step))[0].sum()
        numerical = float((plus - minus) / (2 * step))
        np.testing.assert_allclose(float(gradient[index]), numerical, rtol=0.03, atol=0.08)
        self.assertEqual(model.batchsize, 3)
        double_cost, _ = make_smooth_cp_value_and_grad(
            model, 0., PlannerConfig(1., 0.1, 1., 1., 20, cp_weight=10., cp_gradient="autodiff")
        )(particles)
        np.testing.assert_allclose(double_cost, 2 * cost, rtol=1e-5)

    def test_autodiff_still_filters_hard_bounds_without_spsa(self):
        class Scorer:
            def __init__(self):
                self.model = GNN(2, 8, 26, nnx.Rngs(11), batchsize=2)
                self.model.eval()
                self.quantile = 20.
                self.calls = []
            def lower_bounds(self, positions, radius, mode, sigma):
                self.calls.append((mode, sigma))
                return np.asarray([int(r["features"][:, 0].min()) for r in
                                   communication_records(positions, radius, mode, sigma)])
        scorer = Scorer()
        start = np.asarray([[0., 0.], [0.4, 0.1], [0.2, 0.6]])
        particles = np.stack((start[None], (start + 0.01)[None]))
        config = PlannerConfig(1., 0.1, 1., 1., 1, cp_gradient="autodiff", svgd_step_size=0.001)
        with patch("planning.svgd_cp_planner.cp_spsa_gradient") as spsa:
            waypoints, report = plan(start, start, particles, scorer, config, iterations=1)
            spsa.assert_not_called()
        self.assertIsNotNone(waypoints)
        self.assertGreaterEqual(report["minimum_lower_bound"], 1)
        self.assertEqual(scorer.calls, [("hard", None), ("hard", None)])
        self.assertEqual(report["optimization_cp_cost"], "continuous_expected_class_surrogate")

    def test_unsupported_calibration_is_explicit(self):
        model = GNN(2, 8, 26, nnx.Rngs(1), batchsize=1)
        for kind, quantile in (("lac", 0.5), ("one_sided", float("inf"))):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "one_sided CP with a finite quantile"):
                    make_smooth_cp_value_and_grad(
                        model, quantile, PlannerConfig(1., 0.1, 1., 1., 1, cp_kind=kind))
