import math
import numpy as np


def conformal_quantile(scores, alpha):
    """Return the split-conformal finite-sample quantile."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    # No finite order statistic can provide the requested coverage when this
    # rank is n + 1. Infinity produces the conservative full prediction set.
    if rank > len(scores):
        return float("inf")
    return np.sort(scores)[rank - 1]


def fit_lac(probabilities, labels, alpha):
    probabilities, labels = _validate_predictions(probabilities, labels)
    scores = 1.0 - probabilities[np.arange(len(labels)), labels]
    return float(conformal_quantile(scores, alpha))


def lac_prediction_sets(probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must be a nonempty two-dimensional array")
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    sets = (1.0 - probabilities) <= float(threshold)

    # Adding argmax only enlarges the conformal set, so coverage is preserved.
    empty = ~sets.any(axis=1)
    predicted = probabilities.argmax(axis=1)
    sets[empty, predicted[empty]] = True
    return sets


def calibrate_underbound(predictions, actual, desired_coverage):
    """Calibrate a one-sided correction from prediction - actual scores.

    The returned quantile is computed only from labeled calibration data. A
    future lower bound is ``prediction - quantile`` and has marginal coverage
    ``desired_coverage`` when calibration and future examples are exchangeable.
    """
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)
    if predictions.ndim != 1 or actual.ndim != 1 or predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be equally sized one-dimensional arrays")
    if not 0 < desired_coverage < 1:
        raise ValueError("desired_coverage must be between zero and one")
    scores = predictions - actual
    return float(conformal_quantile(scores, 1.0 - desired_coverage))


def predict_underbounds(predictions, quantile, num_classes=26):
    """Apply a calibrated one-sided quantile without using actual labels."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    if math.isinf(quantile):
        return np.zeros_like(predictions, dtype=np.int64)
    return np.clip(np.ceil(predictions - quantile), 0, num_classes - 1).astype(np.int64)


def coverage(prediction_sets, labels):
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    labels = np.asarray(labels)
    if prediction_sets.ndim != 2 or labels.shape != (len(prediction_sets),):
        raise ValueError("prediction sets and labels have incompatible shapes")
    return prediction_sets[np.arange(len(labels)), labels]


def set_members(prediction_set):
    """Convert one boolean class mask to its integer class members."""
    return np.flatnonzero(prediction_set).tolist()


def _validate_predictions(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must be a nonempty two-dimensional array")
    if labels.shape != (len(probabilities),):
        raise ValueError("labels must have one entry per probability row")
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must index probability columns")
    return probabilities, labels
