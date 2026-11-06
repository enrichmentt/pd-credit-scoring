"""Тесты метрик классификации и PSI."""

import numpy as np

from src.models.train import compute_metrics
from src.monitoring.drift import psi


def test_compute_metrics_keys_and_bounds():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.7, 0.6, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    for key in ["roc_auc", "precision", "recall", "f1", "accuracy"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.05, 0.95, 0.1, 0.9])
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_psi_identical_distributions():
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 0.1, 5000)
    same = rng.normal(0.3, 0.1, 5000)
    assert psi(base, same) < 0.05


def test_psi_detects_drift():
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 0.1, 5000)
    shifted = rng.normal(0.6, 0.1, 5000)
    assert psi(base, shifted) > 0.25
