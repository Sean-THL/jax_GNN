import unittest
from dataclasses import replace

import numpy as np

from planning.run_experiment import (
    comparison_configs,
    trajectory_summary,
    config_from_scenario,
    exact_robustness,
    initial_trajectories,
    scenario_positions,
)
from planning.svgd_cp_planner import PlannerConfig, communication_records


class ExperimentTests(unittest.TestCase):
    def test_independent_goals_are_repeatable_and_not_a_translation(self):
        scenario = {
            "random_positions": {"robot_count": 10, "seed": 42,
                                 "lower": [2., 2.], "upper": [10., 10.], "robust_seed_size": 10},
            "random_goals": {"lower": [70., 2.], "upper": [78., 10.], "seed": 137},
            "config": {"minimum_separation": 0.3, "communication_radius": 3.},
        }
        start, goal = scenario_positions(scenario, 10, 5, 3.)
        start2, goal2 = scenario_positions(scenario, 10, 5, 3.)
        np.testing.assert_array_equal(start, start2)
        np.testing.assert_array_equal(goal, goal2)
        self.assertGreater(np.ptp(goal-start, axis=0).max(), 0.5)
        self.assertTrue((np.linalg.norm(goal-start, axis=1) > 60).all())
        for positions in (start, goal):
            distances = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            self.assertGreaterEqual(distances.min(), 0.3-1e-6)
            self.assertGreaterEqual((distances <= 3.).sum(axis=1).min(), 5)

    def test_summary_handles_no_feasible_trajectory(self):
        summary = trajectory_summary(None, None, None, None, None, 0.25)
        self.assertIsNone(summary["waypoints"])
        self.assertFalse(summary["meets_predicted_target"])
        self.assertEqual(summary["status"], "no_feasible_trajectory")

    def test_required_robustness_is_derived_from_adversary_bound(self):
        config, adversary_bound = config_from_scenario({
            "communication_radius": 1.0,
            "minimum_separation": 0.1,
            "maximum_speed": 1.0,
            "sampling_time": 0.2,
            "adversary_bound": 2,
        })
        self.assertEqual(adversary_bound, 2)
        self.assertEqual(config.required_robustness, 5)
        self.assertEqual(config.connectivity_mode, "smooth")
        self.assertAlmostEqual(config.maximum_step, 0.2)

    def test_scenario_cannot_override_derived_robustness(self):
        with self.assertRaisesRegex(ValueError, "not required_robustness"):
            config_from_scenario({
                "communication_radius": 1.0,
                "minimum_separation": 0.1,
                "maximum_speed": 1.0,
                "sampling_time": 0.2,
                "adversary_bound": 2,
                "required_robustness": 2,
            })

    def test_random_robot_pool_is_reproducible_and_prefix_stable(self):
        scenario = {
            "random_positions": {
                "robot_count": 12,
                "seed": 7,
                "lower": [0.0, 0.0],
                "upper": [0.75, 0.75],
                "robust_seed_size": 10,
                "goal_translation": [0.1, 0.05],
            },
            "config": {"minimum_separation": 0.03, "communication_radius": 0.3},
        }
        start_5, goal_5 = scenario_positions(scenario, 5, 5, 0.3)
        start_12, goal_12 = scenario_positions(scenario, 12, 5, 0.3)
        np.testing.assert_array_equal(start_5, start_12[:5])
        np.testing.assert_array_equal(goal_5, goal_12[:5])
        np.testing.assert_allclose(
            goal_12 - start_12, np.broadcast_to([0.1, 0.05], (12, 2)), atol=1e-7
        )
        self.assertEqual(start_12.shape, (12, 2))
        self.assertEqual(goal_12.shape, (12, 2))
        distances = np.linalg.norm(start_12[:, None] - start_12[None, :], axis=-1)
        degree = ((distances <= 0.3) & ~np.eye(12, dtype=bool)).sum(axis=1)
        self.assertGreaterEqual(degree.min(), 5)

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
                               maximum_speed=1., sampling_time=1.,
                               required_robustness=1)
        simple, degree, learned, automatic = comparison_configs(config)
        self.assertEqual([simple[0], degree[0], learned[0], automatic[0]],
                         ["simple", "soft_degree", "gnn_cp_spsa", "gnn_cp_autodiff"])
        self.assertEqual((simple[1].degree_weight, simple[1].cp_weight), (0., 0.))
        self.assertEqual((degree[1].degree_weight, degree[1].cp_weight),
                         (config.degree_weight, 0.))
        self.assertEqual(learned[1], replace(config, cp_gradient="spsa"))
        self.assertEqual(automatic[1].cp_gradient, "autodiff")
        self.assertEqual(automatic[1].connectivity_mode, "smooth")
        self.assertEqual(simple[1].goal_weight, config.goal_weight)


if __name__ == "__main__":
    unittest.main()
