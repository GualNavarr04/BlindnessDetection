"""Métricas de evaluación y selección de umbrales (contrato 4.6 de docs/plan.md).

Una clase es "referible" si es moderada o más (clase >= REFERABLE_CLASS): ahí es donde un falso
negativo (mandar a casa a alguien que sí necesita oftalmólogo) es el error caro que la herramienta
de tamizaje debe evitar.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REFERABLE_CLASS = 2


def _referable_probability(probs: np.ndarray) -> np.ndarray:
    return probs[:, REFERABLE_CLASS:].sum(axis=1)


def _sensitivity_specificity(true_referable: np.ndarray, pred_referable: np.ndarray) -> tuple[float, float]:
    tp = np.sum(true_referable & pred_referable)
    fn = np.sum(true_referable & ~pred_referable)
    tn = np.sum(~true_referable & ~pred_referable)
    fp = np.sum(~true_referable & pred_referable)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sensitivity), float(specificity)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, referable_threshold: float = 0.5) -> dict:
    """y_true: (N,) clases 0-4; probs: (N, 5) softmax.
    Regresa {"qwk", "sensitivity_referable", "specificity_referable", "confusion_matrix"}."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    predicted_class = probs.argmax(axis=1)

    true_referable = y_true >= REFERABLE_CLASS
    pred_referable = _referable_probability(probs) >= referable_threshold
    sensitivity, specificity = _sensitivity_specificity(true_referable, pred_referable)

    return {
        "qwk": float(cohen_kappa_score(y_true, predicted_class, weights="quadratic")),
        "sensitivity_referable": sensitivity,
        "specificity_referable": specificity,
        "confusion_matrix": confusion_matrix(
            y_true, predicted_class, labels=list(range(probs.shape[1]))
        ).tolist(),
    }


def _best_referable_threshold(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray, min_sensitivity: float) -> float:
    true_referable = y_true >= REFERABLE_CLASS
    referable_prob = _referable_probability(probs)

    candidates = []
    for t in grid:
        sensitivity, specificity = _sensitivity_specificity(true_referable, referable_prob >= t)
        candidates.append((t, sensitivity, specificity))

    feasible = [(t, spec) for t, sens, spec in candidates if sens >= min_sensitivity]
    if feasible:
        return max(feasible, key=lambda ts: ts[1])[0]
    # Ningún umbral alcanza min_sensitivity (dataset chico o modelo muy inseguro):
    # usar el que da la mayor sensibilidad posible.
    return max(candidates, key=lambda c: c[1])[0]


def _best_confidence_threshold(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray, min_accuracy: float) -> float:
    predicted_class = probs.argmax(axis=1)
    correct = predicted_class == y_true
    max_prob = probs.max(axis=1)

    candidates = []
    for t in grid:
        mask = max_prob >= t
        if mask.any():
            candidates.append((t, correct[mask].mean()))

    feasible = [t for t, accuracy in candidates if accuracy >= min_accuracy]
    if feasible:
        # el umbral más bajo que ya cumple: marca como "baja confianza" al menor número de casos posible
        return min(feasible)
    if candidates:
        return max(candidates, key=lambda c: c[1])[0]
    return 0.5


def choose_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_sensitivity: float = 0.9,
    min_confidence_accuracy: float = 0.9,
) -> dict:
    """y_true: (N,) clases 0-4; probs: (N, 5) softmax.
    Barre umbrales sobre P(clase >= 2) y elige el de mayor especificidad que cumpla
    min_sensitivity. Regresa {"referable_threshold", "confidence_threshold"}."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    grid = np.linspace(0.01, 0.99, 99)

    return {
        "referable_threshold": _best_referable_threshold(y_true, probs, grid, min_sensitivity),
        "confidence_threshold": _best_confidence_threshold(y_true, probs, grid, min_confidence_accuracy),
    }
