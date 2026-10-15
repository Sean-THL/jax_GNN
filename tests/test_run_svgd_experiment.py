import unittest

import numpy as np

from planning.run_experiment import comparison_configs, exact_robustness, initial_trajectories
from planning.svgd_cp_planner import PlannerConfig, communication_records


class ExperimentTests(unittest.TestCase):
    def test_exact_robustness_on_known_graphs(self):
        positions = np.asarray([[[0., 0.], [0.1, 0.], [0.2, 0.], [0.3, 0.]]])
        for radius, expected in ((0.001, 0), (0.11, 1), (1.0, 2)):
            with self.subTest(radius=radius):
                record = communication_records(positions, radius)[0]
                self.assertEqual(exact_robustness(record), expected)

    def test_straight_line_particles_have_requested_shape_and_goal(self):
        start = np.asarray([[0., 0.], [1., 1.]])
        goal = np.asarray([[1., 0.], [2., 1.]])
        particles = initial_trajectories(start, goal, 4, 3, 0., 42)
        self.assertEqual(particles.shape, (4, 3, 2, 2))
        np.testing.assert_array_equal(particles[:, -1], np.broadcast_to(goal, (4, 2, 2)))

    def test_comparison_changes_only_connectivity_weights(self):
        config = PlannerConfig(communication_radius=1., minimum_separation=0.1,
                               maximum_step=1., required_robustness=1)
        simple, degree, learned = comparison_configs(config)
        self.assertEqual([simple[0], degree[0], learned[0]],
                         ["simple", "soft_degree", "gnn_cp"])
        self.assertEqual((simple[1].degree_weight, simple[1].cp_weight), (0., 0.))
        self.assertEqual((degree[1].degree_weight, degree[1].cp_weight),
                         (config.degree_weight, 0.))
        self.assertEqual(learned[1], config)
        self.assertEqual(simple[1].goal_weight, config.goal_weight)


if __name__ == "__main__":
    unittest.main()
