"""Pruebas de rethreshold.py (umbrales nuevos sin reentrenar), sin llamar a AWS."""

import numpy as np
import pandas as pd

from src.training import rethreshold as rt


def fake_predictions(seed=0, n=200):
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("val", "test"):
        labels = rng.integers(0, 5, n)
        logits = rng.normal(0, 1, (n, 5))
        logits[np.arange(n), labels] += 2.0  # el modelo acierta casi siempre, pero no siempre
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        rows.append(pd.DataFrame(probs, columns=[f"p{c}" for c in range(5)]).assign(split=split, diagnosis=labels))
    return pd.concat(rows, ignore_index=True)


METADATA = {
    "run_id": "r1",
    "architecture": "x",
    "decision": {"referable_threshold": 0.57, "confidence_threshold": 0.87,
                 "traffic_light": {"verde": [0, 1], "amarillo": [2], "rojo": [3, 4]}},
    "metrics": {"qwk": 0.0, "sensitivity_referable": 0.0, "specificity_referable": 0.0},
    "validation_metrics": {"qwk": 0.0, "sensitivity_referable": 0.0, "specificity_referable": 0.0},
}


def test_higher_target_gives_more_sensitivity_on_validation():
    preds = fake_predictions()
    low = rt.updated_metadata(METADATA, preds, 0.80)
    high = rt.updated_metadata(METADATA, preds, 0.97)
    assert high["validation_metrics"]["sensitivity_referable"] >= 0.97
    assert high["decision"]["referable_threshold"] <= low["decision"]["referable_threshold"]
    assert high["validation_metrics"]["sensitivity_referable"] >= low["validation_metrics"]["sensitivity_referable"]


def test_keeps_contract_fields_and_records_policy():
    new = rt.updated_metadata(METADATA, fake_predictions(), 0.95)
    assert set(new["decision"]) == {"referable_threshold", "confidence_threshold", "traffic_light"}
    assert new["decision"]["traffic_light"] == METADATA["decision"]["traffic_light"]
    assert new["run_id"] == "r1" and new["architecture"] == "x"
    assert new["threshold_selection"]["min_sensitivity"] == 0.95 and new["threshold_selection"]["chosen_on"] == "val"
    assert new["metrics"]["n"] == 200 and 0 <= new["metrics"]["accuracy"] <= 1
    assert METADATA["decision"]["referable_threshold"] == 0.57  # no modifica el original


def test_metrics_are_measured_on_test_not_validation():
    preds = fake_predictions()
    new = rt.updated_metadata(METADATA, preds, 0.95)
    test = preds[preds.split == "test"]
    accuracy = (test[[f"p{c}" for c in range(5)]].to_numpy().argmax(axis=1) == test.diagnosis.to_numpy()).mean()
    assert np.isclose(new["metrics"]["accuracy"], accuracy)
