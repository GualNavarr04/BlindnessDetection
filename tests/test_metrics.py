"""Pruebas de métricas y selección de umbrales (contrato 4.6)."""

import numpy as np

from src.data.metrics import choose_thresholds, compute_metrics


def one_hot_probs(labels, num_classes=5, confidence=0.9):
    """Probabilidades donde la clase correcta domina con `confidence`, el resto se reparte parejo."""
    labels = np.asarray(labels)
    other = (1 - confidence) / (num_classes - 1)
    probs = np.full((len(labels), num_classes), other)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    metrics = compute_metrics(y_true, one_hot_probs(y_true))
    assert metrics["qwk"] > 0.95
    assert metrics["sensitivity_referable"] == 1.0
    assert metrics["specificity_referable"] == 1.0
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (5, 5)
    assert cm.sum() == len(y_true)


def test_compute_metrics_no_referable_cases_is_well_defined():
    # Sin ningún caso referible real, sensibilidad no puede dividir entre cero: debe dar 0.0, no tronar.
    y_true = np.zeros(10, dtype=int)
    metrics = compute_metrics(y_true, one_hot_probs(y_true))
    assert metrics["sensitivity_referable"] == 0.0
    assert metrics["specificity_referable"] == 1.0


def test_compute_metrics_referable_threshold_changes_sensitivity():
    y_true = np.array([2, 2, 2, 0, 0, 0])
    probs = np.array(
        [
            [0.30, 0.25, 0.45, 0.0, 0.0],
            [0.30, 0.25, 0.45, 0.0, 0.0],
            [0.30, 0.25, 0.45, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0],
        ]
    )
    loose = compute_metrics(y_true, probs, referable_threshold=0.4)
    strict = compute_metrics(y_true, probs, referable_threshold=0.5)
    assert loose["sensitivity_referable"] == 1.0
    assert strict["sensitivity_referable"] == 0.0


def test_choose_thresholds_meets_min_sensitivity():
    # 3 casos referibles (índices 2,3,4) con P(clase>=2) = 0.55, 0.45 y 0.95; 2 sanos con 0.05 y 0.30.
    y_true = np.array([0, 1, 2, 2, 4])
    referable_mass = [0.05, 0.30, 0.55, 0.45, 0.95]
    probs = np.array([[1 - m, 0.0, m, 0.0, 0.0] for m in referable_mass])

    thresholds = choose_thresholds(y_true, probs, min_sensitivity=0.9)
    assert thresholds["referable_threshold"] <= 0.45

    metrics = compute_metrics(y_true, probs, thresholds["referable_threshold"])
    assert metrics["sensitivity_referable"] >= 0.9


def test_choose_thresholds_confidence_excludes_low_confidence_error():
    # Único error: índice 3 (predicha clase 1, real clase 2), y es el de menor probabilidad máxima.
    y_true = np.array([0, 1, 2, 2, 4])
    probs = np.array(
        [
            [0.90, 0.04, 0.03, 0.02, 0.01],
            [0.05, 0.55, 0.20, 0.10, 0.10],
            [0.10, 0.10, 0.60, 0.10, 0.10],
            [0.20, 0.45, 0.20, 0.10, 0.05],
            [0.01, 0.01, 0.01, 0.02, 0.95],
        ]
    )
    thresholds = choose_thresholds(y_true, probs, min_confidence_accuracy=0.9)
    assert thresholds["confidence_threshold"] > probs.max(axis=1)[3]
