import unittest

import numpy as np

from planning.svgd_cp_planner import (
    PlannerConfig,
    communication_records,
    cp_cost,
    plan,
    svgd_direction,
)


class _DegreeScorer:
    def lower_bounds(self, positions, radius):
        records = communication_records(positions, radius)
        return np.asarray([int(record["features"][:, 0].min()) for record in records])


class PlannerTests(unittest.TestCase):
    def test_graph_representation_matches_directed_degree_features(self):
        positions = np.asarray([[[0., 0.], [0.5, 0.], [2., 0.]]])
        graph = communication_records(positions, radius=1.0)[0]
        np.testing.assert_array_equal(graph["features"][:, 0], [1, 1, 0])
        np.testing.assert_array_equal(graph["edge_index"], [[0, 1], [1, 0]])

    def test_svgd_repels_particles_when_cost_gradient_is_zero(self):
        particles = np.asarray([[-1.0], [1.0]])
        direction = svgd_direction(particles, np.zeros_like(particles))
        self.assertLess(direction[0, 0], 0)
        self.assertGreater(direction[1, 0], 0)

    def test_cp_cost_uses_worst_future_step(self):
        particles = np.asarray([
            [[[0., 0.], [0.5, 0.]], [[0., 0.], [2., 0.]]],
            [[[0., 0.], [0.5, 0.]], [[0., 0.], [0.5, 0.]]],
        ])
        config = PlannerConfig(communication_radius=1., minimum_separation=0.1,
                               maximum_step=2., required_robustness=1, cp_weight=5.)
        cost, lower = cp_cost(particles, _DegreeScorer(), config)
        np.testing.assert_array_equal(lower, [0, 1])
        np.testing.assert_array_equal(cost, [5., 0.])

    def test_baseline_plans_without_graph_scorer(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        goal = np.asarray([[0.1, 0.], [0.6, 0.]])
        particles = np.broadcast_to(goal, (2, 1, 2, 2)).copy()
        particles[1] += 0.01
        config = PlannerConfig(communication_radius=1., minimum_separation=0.1,
                               maximum_step=1., required_robustness=1,
                               degree_weight=0., cp_weight=0.)
        waypoints, report = plan(start, goal, particles, None, config, iterations=1)
        self.assertEqual(waypoints.shape, (1, 2, 2))
        self.assertIsNone(report["minimum_lower_bound"])


if __name__ == "__main__":
    unittest.main()
