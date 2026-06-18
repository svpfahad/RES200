import math
from pathlib import Path

import pandas as pd


def test_normalize_res200_rows_maps_schema_and_tasks():
    from sota_pka.data import normalize_res200_rows

    raw = pd.DataFrame(
        {
            "OriginalSmiles": ["CC(=O)O", "CN"],
            "InChI Key_QSARr": ["QTBSBXVTEAMEQO-UHFFFAOYSA-N", "BAVYZALUXZFZLV-UHFFFAOYSA-N"],
            "pKa": ["4.76", "10.6"],
            "basicOrAcidic": ["acidic", "basic"],
            "temp": ["25", ""],
            "method": ["literature", "potentiometric"],
        }
    )

    normalized = normalize_res200_rows(raw, source="unit", split="train")

    assert list(normalized.columns) == [
        "smiles",
        "canonical_smiles",
        "inchi_key",
        "pka",
        "task",
        "source",
        "temperature",
        "method",
        "split",
        "notes",
    ]
    assert normalized["task"].tolist() == ["acidic", "basic"]
    assert normalized["pka"].tolist() == [4.76, 10.6]
    assert normalized["source"].unique().tolist() == ["unit"]


def test_normalize_iupac_rows_maps_acidity_labels():
    from sota_pka.data import normalize_iupac_rows

    raw = pd.DataFrame(
        {
            "SMILES": ["CN", "CC(=O)O"],
            "pka_value": ["10.65", "4.76"],
            "acidity_label": ["AH", "HA"],
            "T": ["25", "25"],
            "method": ["E1b", "O2"],
            "assessment": ["Reliable", "Reliable"],
        }
    )

    normalized = normalize_iupac_rows(raw, source="iupac", split="external")

    assert normalized["task"].tolist() == ["basic", "acidic"]
    assert normalized["split"].unique().tolist() == ["external"]
    assert normalized["notes"].tolist() == ["assessment=Reliable", "assessment=Reliable"]


def test_deduplicate_prefers_earlier_sources_and_rejects_split_leakage():
    from sota_pka.data import assert_no_split_leakage, deduplicate_measurements, purge_split_leakage

    rows = pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCN"],
            "canonical_smiles": ["CCO", "CCO", "CCN"],
            "inchi_key": ["A", "A", "B"],
            "pka": [15.9, 16.0, 10.7],
            "task": ["acidic", "acidic", "basic"],
            "source": ["primary", "secondary", "primary"],
            "temperature": [25.0, 25.0, 25.0],
            "method": ["m1", "m2", "m1"],
            "split": ["train", "train", "test"],
            "notes": ["", "", ""],
        }
    )

    deduped = deduplicate_measurements(rows)

    assert len(deduped) == 2
    assert deduped.loc[deduped["inchi_key"] == "A", "pka"].iloc[0] == 15.9
    assert_no_split_leakage(deduped, train_splits={"train"}, test_splits={"test"})

    leaked = deduped.copy()
    leaked.loc[leaked["inchi_key"] == "A", "split"] = "test"
    leaked = pd.concat([deduped, leaked[leaked["inchi_key"] == "A"]], ignore_index=True)
    try:
        assert_no_split_leakage(leaked, train_splits={"train"}, test_splits={"test"})
    except ValueError as exc:
        assert "split leakage" in str(exc).lower()
    else:
        raise AssertionError("Expected leakage check to fail")

    purged = purge_split_leakage(leaked, train_splits={"train"}, test_splits={"test"})
    assert_no_split_leakage(purged, train_splits={"train"}, test_splits={"test"})
    assert "test" not in purged.loc[purged["inchi_key"] == "A", "split"].tolist()


def test_metrics_are_recomputed_from_prediction_file(tmp_path: Path):
    from sota_pka.evaluate import regression_metrics, summarize_prediction_file

    predictions = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.5, 2.5]})
    pred_path = tmp_path / "predictions.csv"
    predictions.to_csv(pred_path, index=False)

    metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
    summary = summarize_prediction_file(pred_path)

    assert math.isclose(metrics["rmse"], 0.40824829046, rel_tol=1e-6)
    assert math.isclose(metrics["mae"], 1.0 / 3.0, rel_tol=1e-6)
    assert summary["n"] == 3
    assert summary["rmse"] == metrics["rmse"]


def test_train_sklearn_baselines_writes_metrics_and_predictions(tmp_path: Path):
    from sota_pka.train_baselines import train_sklearn_baselines

    train = pd.DataFrame(
        {
            "pKa": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    test = pd.DataFrame({"pKa": [7.0, 8.0], "f1": [7.0, 8.0], "f2": [4.0, 4.0]})

    result = train_sklearn_baselines(
        train,
        test,
        output_dir=tmp_path,
        models=["elasticnet", "random_forest"],
        seed=0,
        random_forest_estimators=8,
    )

    metrics_path = tmp_path / "metrics.csv"
    assert metrics_path.exists()
    assert {"elasticnet", "random_forest"} == set(result["model"])
    assert (tmp_path / "predictions_elasticnet.csv").exists()
    assert (tmp_path / "predictions_random_forest.csv").exists()


def test_descriptor_alignment_retains_train_columns_missing_from_test():
    from sota_pka.train_baselines import align_numeric_features

    train = pd.DataFrame({"pKa": [1.0, 2.0], "shared": [1.0, 2.0], "train_only": [3.0, 4.0]})
    test = pd.DataFrame({"pKa": [3.0], "shared": [3.0], "extra_test": [99.0]})

    x_train, x_test, _, _ = align_numeric_features(train, test, "pKa")

    assert x_train.columns.tolist() == ["shared", "train_only"]
    assert x_test.columns.tolist() == ["shared", "train_only"]
    assert x_test["train_only"].tolist() == [0.0]


def test_y_randomization_control_writes_shuffled_target_run(tmp_path: Path):
    from sota_pka.train_baselines import train_y_randomization_control

    train = pd.DataFrame(
        {
            "pKa": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    test = pd.DataFrame({"pKa": [7.0, 8.0], "f1": [7.0, 8.0], "f2": [4.0, 4.0]})

    result = train_y_randomization_control(
        train,
        test,
        output_dir=tmp_path,
        model="random_forest",
        seed=123,
        random_forest_estimators=8,
    )

    assert result.loc[0, "control"] == "y_randomization"
    assert (tmp_path / "predictions_y_randomization_random_forest.csv").exists()
