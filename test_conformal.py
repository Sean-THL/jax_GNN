import unittest

import numpy as np

from jax_conformal import (
    calibrate_underbound,
    conformal_quantile,
    coverage,
    fit_lac,
    lac_prediction_sets,
    predict_underbounds,
)


class ConformalTests(unittest.TestCase):
    def test_finite_sample_quantile(self):
        # ceil((5 + 1) * 0.8) selects the fifth order statistic.
        self.assertEqual(conformal_quantile([0.1, 0.5, 0.2, 0.4, 0.3], 0.2), 0.5)
        self.assertTrue(np.isinf(conformal_quantile([0.1, 0.2], 0.1)))

    def test_lac_sets_and_coverage(self):
        probabilities = np.array([[0.8, 0.15, 0.05], [0.1, 0.6, 0.3]])
        labels = np.array([0, 2])
        threshold = fit_lac(probabilities, labels, alpha=0.5)
        self.assertAlmostEqual(threshold, 0.7)
        sets = lac_prediction_sets(probabilities, threshold)
        np.testing.assert_array_equal(sets, [[True, False, False], [False, True, True]])
        np.testing.assert_array_equal(coverage(sets, labels), [True, True])

    def test_empty_lac_set_is_enlarged_with_argmax(self):
        sets = lac_prediction_sets(np.array([[0.4, 0.35, 0.25]]), threshold=0.1)
        np.testing.assert_array_equal(sets, [[True, False, False]])

    def test_one_sided_lower_bounds(self):
        predictions = np.array([4, 7, 2, 9])
        actual = np.array([4, 6, 3, 7])
        quantile = calibrate_underbound(predictions, actual, desired_coverage=0.75)
        self.assertEqual(quantile, 2)
        np.testing.assert_array_equal(
            predict_underbounds(np.array([0, 5, 25]), quantile), [0, 3, 23]
        )

    def test_one_sided_score_is_not_absolute(self):
        predictions = np.array([2, 8, 4])
        actual = np.array([5, 7, 4])
        # Scores are [-3, 1, 0], so the median correction is 0 rather than 1.
        self.assertEqual(calibrate_underbound(predictions, actual, 0.5), 0)


if __name__ == "__main__":
    unittest.main()
