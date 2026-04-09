"""
drift – data-drift detection using Evidently AI + automatic retraining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from app.config import (
    DRIFT_REPORT_PATH,
    DRIFT_THRESHOLD,
    FEATURE_NAMES,
    ROOT_DIR,
    logger,
)
from app.model import build_model, clear_cache, save_model


def _feature_columns() -> list[str]:
    return FEATURE_NAMES


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: Optional[str] = None,
) -> Tuple[bool, float, dict]:
    """
    Compare reference (training) vs current (production) data.

    Returns:
        (is_drifted, drift_share, full_results_dict)
    """
    cols = _feature_columns()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])

    # Save HTML report
    out = Path(report_path or (ROOT_DIR / DRIFT_REPORT_PATH))
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    logger.info("Drift report saved → %s", out)

    # Parse results
    result = report.as_dict()
    metrics = result["metrics"]

    # Find the DataDriftTable metric result
    drift_share = 0.0
    dataset_drift = False
    for m in metrics:
        res = m.get("result", {})
        if "share_of_drifted_columns" in res:
            drift_share = res["share_of_drifted_columns"]
            dataset_drift = res.get("dataset_drift", False)
            break

    is_drifted = drift_share >= DRIFT_THRESHOLD or dataset_drift

    logger.info(
        "Drift analysis | share=%.2f | threshold=%.2f | drifted=%s",
        drift_share,
        DRIFT_THRESHOLD,
        is_drifted,
    )

    return is_drifted, drift_share, result


def retrain_if_needed(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_path: Optional[str] = None,
) -> dict:
    """
    Run drift detection; if drift exceeds threshold, retrain on combined data
    AND update the reference file (train.csv) with the new data distribution.
    """
    is_drifted, share, _ = detect_drift(reference, current)

    if not is_drifted:
        return {
            "retrained": False,
            "drift_share": round(share, 4),
            "message": "No significant drift detected – model unchanged.",
        }

    logger.warning("Drift detected (%.2f) — retraining model …", share)

    cols = _feature_columns()
    combined = pd.concat([reference[cols], current[cols]], ignore_index=True)

    # 1. Update the ML model binary
    model = build_model()
    model.fit(combined)
    save_model(model)
    clear_cache()

    # 2. Update the reference CSV so NEXT time drift detection is based on THIS state
    # We replace the baseline with the current production data to "reset" the drift
    if reference_path:
        save_path = ROOT_DIR / reference_path
        current[cols].to_csv(save_path, index=False)
        logger.info("Reference data reset to current production state → %s", save_path)

    return {
        "retrained": True,
        "drift_share": round(share, 4),
        "message": "Model retrained and reference baseline updated.",
    }
