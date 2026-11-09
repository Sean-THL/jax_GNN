import unittest

import numpy as np

from planning.svgd_cp_planner import (
    PlannerConfig,
    communication_records,
    cp_cost,
    cp_spsa_gradient,
    plan,
    svgd_direction,
)


class _DegreeScorer:
    def __init__(self):
        self.calls = []

    def lower_bounds(self, positions, radius, mode="hard", sigma=None):
        self.calls.append((mode, sigma))
        records = communication_records(positions, radius, mode, sigma)
        return np.asarray([int(record["features"][:, 0].min()) for record in records])


class PlannerTests(unittest.TestCase):
    def test_fixed_goal_is_preserved_in_every_population(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        goal = start + 0.1
        particles = np.broadcast_to(start, (2, 3, 2, 2)).copy()
        particles[1] += 0.02
        config = PlannerConfig(1., 0.1, 1., 1., 1, cp_weight=0., fixed_goal=True)
        selected, report = plan(start, goal, particles, _DegreeScorer(), config,
                                iterations=2, record_particles=True)
        np.testing.assert_allclose(selected[-1], goal, atol=1e-7)
        endpoints = report["particle_snapshots"][:, :, -1]
        np.testing.assert_allclose(endpoints, np.broadcast_to(goal, endpoints.shape), atol=1e-7)

    def test_fixed_infeasible_goal_returns_no_trajectory(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        goal = np.asarray([[0., 0.], [2., 0.]])
        particles = np.broadcast_to(start, (2, 2, 2, 2)).copy()
        config = PlannerConfig(1., 0.1, 1., 1., 1, cp_weight=0., fixed_goal=True)
        selected, report = plan(start, goal, particles, _DegreeScorer(), config, iterations=1)
        self.assertIsNone(selected)
        self.assertEqual(report["status"], "no_feasible_trajectory")

    def test_batched_spsa_matches_separate_probes(self):
        particles = np.random.default_rng(7).normal(size=(3, 2, 4, 2))
        for directions in (1, 2, 3):
            with self.subTest(directions=directions):
                config = PlannerConfig(1., 0.1, 1., 1., 2,
                                       cp_directions=directions)
                rng = np.random.default_rng(42)
                reference = np.zeros_like(particles)
                scorer = _DegreeScorer()
                for _ in range(directions):
                    noise = rng.normal(size=particles.shape)
                    positive, _ = cp_cost(particles + config.cp_smoothing * noise,
                                          scorer, config)
                    negative, _ = cp_cost(particles - config.cp_smoothing * noise,
                                          scorer, config)
                    reference += ((positive - negative) / (2 * config.cp_smoothing))[:, None, None, None] * noise
                scorer = _DegreeScorer()
                gradient = cp_spsa_gradient(particles, scorer, config,
                                            np.random.default_rng(42))
                np.testing.assert_allclose(gradient, reference / directions)
                self.assertEqual(len(scorer.calls), 1)

    def test_graph_representation_matches_directed_degree_features(self):
        positions = np.asarray([[[0., 0.], [0.5, 0.], [2., 0.]]])
        graph = communication_records(positions, radius=1.0)[0]
        np.testing.assert_array_equal(graph["features"][:, 0], [1, 1, 0])
        np.testing.assert_array_equal(graph["edge_index"], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(graph["edge_attr"], [[1.], [1.]])

    def test_smooth_graph_keeps_fixed_edges_and_weighted_degrees(self):
        positions = np.asarray([[[0., 0.], [0.5, 0.], [2., 0.]]])
        graph = communication_records(positions, radius=1.0, mode="smooth")[0]
        self.assertEqual(graph["edge_index"].shape, (2, 6))
        np.testing.assert_allclose(
            graph["features"][:, 0],
            [graph["edge_attr"][0, 0], graph["edge_attr"][2, 0], 0.0],
            rtol=1e-6,
            atol=1e-6,
        )

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
                               maximum_speed=2., sampling_time=1.,
                               required_robustness=1, cp_weight=5.)
        scorer = _DegreeScorer()
        cost, lower = cp_cost(particles, scorer, config)
        np.testing.assert_array_equal(lower, [0, 1])
        np.testing.assert_array_equal(cost, [5., 0.])
        self.assertEqual(scorer.calls, [("hard", None)])

    def test_cp_uses_hard_support_when_planning_mode_is_smooth(self):
        particles = np.asarray([
            [[[0., 0.], [0.5, 0.]]],
            [[[0., 0.], [0.6, 0.]]],
        ])
        config = PlannerConfig(communication_radius=1., minimum_separation=0.1,
                               maximum_speed=1., sampling_time=1.,
                               required_robustness=1, connectivity_mode="smooth")
        scorer = _DegreeScorer()
        cp_cost(particles, scorer, config)
        self.assertEqual(scorer.calls, [("hard", None)])

    def test_baseline_still_applies_final_filter(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        goal = np.asarray([[0.1, 0.], [0.6, 0.]])
        particles = np.broadcast_to(goal, (2, 1, 2, 2)).copy()
        particles[1] += 0.01
        config = PlannerConfig(communication_radius=1., minimum_separation=0.1,
                               maximum_speed=1., sampling_time=1.,
                               required_robustness=1,
                               degree_weight=0., cp_weight=0.)
        waypoints, report = plan(start, goal, particles, _DegreeScorer(), config, iterations=1)
        self.assertEqual(waypoints.shape, (1, 2, 2))
        self.assertEqual(report["minimum_lower_bound"], 1)
        self.assertEqual(report["status"], "feasible")

    def test_filter_rejects_cheaper_candidate_and_checks_every_step(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        goal = np.asarray([[0., 0.], [2., 0.]])
        # Candidate 0 reaches the goal but fails at its final step.
        # Candidates 1 and 2 pass; candidate 1 is cheaper.
        particles = np.asarray([
            [start, goal],
            [start, [[0., 0.], [0.8, 0.]]],
            [start, start],
        ])
        for weight in (0., 0.001):
            with self.subTest(cp_weight=weight):
                config = PlannerConfig(
                    communication_radius=1., minimum_separation=0.1,
                    maximum_speed=10., sampling_time=1., required_robustness=1,
                    cp_weight=weight, degree_weight=0., path_weight=0., cp_gradient="spsa",
                    svgd_step_size=0.,
                )
                waypoints, report = plan(start, goal, particles, _DegreeScorer(),
                                         config, iterations=1)
                np.testing.assert_array_equal(waypoints, particles[1])
                self.assertEqual(report["index"], 1)
                self.assertEqual(report["passing_candidates"], 4)

    def test_no_passing_candidate_returns_no_trajectory(self):
        start = np.asarray([[0., 0.], [2., 0.]])
        particles = np.broadcast_to(start, (2, 1, 2, 2)).copy()
        config = PlannerConfig(
            communication_radius=1., minimum_separation=0.1,
            maximum_speed=1., sampling_time=1., required_robustness=1,
            cp_weight=0., svgd_step_size=0.,
        )
        waypoints, report = plan(start, start, particles, _DegreeScorer(),
                                 config, iterations=1)
        self.assertIsNone(waypoints)
        self.assertEqual(report["status"], "no_feasible_trajectory")
        self.assertIsNone(report["cost"])
        self.assertIsNone(report["index"])

    def test_final_filter_requires_scorer_even_without_cp_penalty(self):
        start = np.asarray([[0., 0.], [0.5, 0.]])
        particles = np.broadcast_to(start, (2, 1, 2, 2)).copy()
        config = PlannerConfig(1., 0.1, 1., 1., 1, cp_weight=0.)
        with self.assertRaisesRegex(ValueError, "final CP filter"):
            plan(start, start, particles, None, config, iterations=1)


if __name__ == "__main__":
    unittest.main()
