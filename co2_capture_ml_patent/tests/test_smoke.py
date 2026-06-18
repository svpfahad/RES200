from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from co2patent.pipeline import infer_group, infer_target, load_table, screen_candidates, train_best_model


def test_training_and_screening_smoke() -> None:
    data = load_table(ROOT / "data" / "sample" / "demo_co2_solubility.csv")
    target = infer_target(data, "co2_solubility_mol_frac")
    group = infer_group(data, "solvent_id")
    result, leaderboard, predictions = train_best_model(
        data,
        target=target,
        group=group,
        test_size=0.25,
        calibration_size=0.25,
        alpha=0.10,
        random_state=7,
    )

    assert result.model_name in set(leaderboard["model"])
    assert result.calibration_qhat >= 0
    assert {"prediction", "lower", "upper", "abs_error"}.issubset(predictions.columns)

    bundle = {
        "model_name": result.model_name,
        "model": result.model,
        "metrics": result.metrics,
        "calibration_qhat": result.calibration_qhat,
        "feature_columns": result.feature_columns,
        "target_column": result.target_column,
        "group_column": result.group_column,
        "train_profile": result.train_profile,
        "alpha": result.alpha,
    }
    candidates = load_table(ROOT / "data" / "sample" / "candidate_screen.csv")
    ranked = screen_candidates(bundle, candidates)

    assert {"rank_score", "confidence_grade", "domain_status", "decision"}.issubset(ranked.columns)
    assert len(ranked) == len(candidates)
    assert "needs_data" in set(ranked["decision"])

