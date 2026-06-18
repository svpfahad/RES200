from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_CANDIDATES = (
    "co2_solubility_mol_frac",
    "co2_solubility",
    "solubility",
    "x_co2",
    "loading",
    "henry_constant",
    "henry_law_constant",
)

TARGET_LEAK_COLUMNS = set(TARGET_CANDIDATES) | {
    "ln_co2_solubility_exp",
    "ln_co2_solubility_experimental",
    "co2_solubility_mol_frac",
}

SMILES_TOKENS = ("Cl", "Br", "C", "N", "O", "S", "P", "F", "I", "B")

METADATA_COLUMNS = {
    "candidate_id",
    "solvent_id",
    "ionic_liquid",
    "il",
    "il_no",
    "ref",
    "reference",
    "source_sheet",
    "source_file",
}


@dataclass
class TrainResult:
    model_name: str
    model: Pipeline
    metrics: dict[str, float]
    calibration_qhat: float
    feature_columns: list[str]
    target_column: str
    group_column: str | None
    train_profile: dict[str, Any]
    alpha: float


def load_table(path: str | Path, sheet: str | int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def smiles_descriptor_frame(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = series.fillna("").astype(str)
    rows: dict[str, pd.Series] = {
        f"{prefix}_smiles_len": values.str.len(),
        f"{prefix}_branches": values.str.count(r"\(") + values.str.count(r"\)"),
        f"{prefix}_ring_digits": values.str.count(r"[0-9]"),
        f"{prefix}_positive_charge": values.str.count(r"\+"),
        f"{prefix}_negative_charge": values.str.count(r"\-"),
        f"{prefix}_aromatic_atoms": values.str.count(r"[cnops]"),
        f"{prefix}_double_bonds": values.str.count("="),
        f"{prefix}_triple_bonds": values.str.count("#"),
    }
    for token in SMILES_TOKENS:
        safe = token.lower()
        rows[f"{prefix}_atom_{safe}"] = values.str.count(token)
    descriptor_frame = pd.DataFrame(rows)
    hetero_cols = [
        f"{prefix}_atom_n",
        f"{prefix}_atom_o",
        f"{prefix}_atom_s",
        f"{prefix}_atom_p",
        f"{prefix}_atom_f",
        f"{prefix}_atom_cl",
        f"{prefix}_atom_br",
        f"{prefix}_atom_i",
        f"{prefix}_atom_b",
    ]
    descriptor_frame[f"{prefix}_hetero_atoms"] = descriptor_frame[
        [c for c in hetero_cols if c in descriptor_frame]
    ].sum(axis=1)
    descriptor_frame[f"{prefix}_formal_charge_proxy"] = (
        descriptor_frame[f"{prefix}_positive_charge"]
        - descriptor_frame[f"{prefix}_negative_charge"]
    )
    return descriptor_frame


def augment_smiles_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    augmented = df.copy()
    smiles_columns = [
        c for c in augmented.columns if "smiles" in c.lower() and pd.api.types.is_object_dtype(augmented[c])
    ]
    for column in smiles_columns:
        prefix = column.lower().replace("smiles", "").strip("_") or "mol"
        descriptors = smiles_descriptor_frame(augmented[column], prefix)
        for desc_column in descriptors.columns:
            augmented[desc_column] = descriptors[desc_column]
    return augmented


def infer_target(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Target column not found: {explicit}")
        return explicit
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(
        "Could not infer target column. Pass --target explicitly. "
        f"Tried: {', '.join(TARGET_CANDIDATES)}"
    )


def infer_group(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit:
        if explicit.strip().lower() in {"none", "random", "no_group", "nogroup"}:
            return None
        if explicit not in df.columns:
            raise ValueError(f"Group column not found: {explicit}")
        return explicit
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in ("solvent_id", "ionic_liquid", "il"):
        if name in lower_to_original:
            return lower_to_original[name]
    if "cation" in lower_to_original and "anion" in lower_to_original:
        cation = lower_to_original["cation"]
        anion = lower_to_original["anion"]
        df["solvent_id"] = df[cation].astype(str) + "|" + df[anion].astype(str)
        return "solvent_id"
    return None


def select_feature_columns(
    df: pd.DataFrame,
    target: str,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    exclude = set(exclude_columns or [])
    features: list[str] = []
    for column in df.columns:
        if column == target:
            continue
        if column.lower() in TARGET_LEAK_COLUMNS:
            continue
        if column in exclude:
            continue
        lower = column.lower()
        if lower in METADATA_COLUMNS:
            continue
        if "smiles" in lower:
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        features.append(column)
    if not features:
        raise ValueError("No usable feature columns found.")
    return features


def clean_training_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[target] = pd.to_numeric(cleaned[target], errors="coerce")
    cleaned = cleaned.dropna(subset=[target]).reset_index(drop=True)
    if len(cleaned) < 10:
        raise ValueError("Need at least 10 rows with a numeric target for training.")
    return cleaned


def split_data(
    df: pd.DataFrame,
    target: str,
    group: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group and group in df.columns and df[group].nunique(dropna=True) >= 4:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(df, df[target], groups=df[group].astype(str)))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric_features = [
        c for c in features if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = [c for c in features if c not in numeric_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def model_candidates(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=2,
        )
    except Exception:
        pass
    return models


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def conformal_qhat(y_true: pd.Series, y_pred: np.ndarray, alpha: float) -> float:
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if len(residuals) == 0:
        return float("nan")
    n = len(residuals)
    quantile = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, quantile, method="higher"))


def build_domain_profile(train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for column in features:
        if pd.api.types.is_numeric_dtype(train[column]):
            series = pd.to_numeric(train[column], errors="coerce")
            profile["numeric"][column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        else:
            values = sorted(str(v) for v in train[column].dropna().unique())
            profile["categorical"][column] = values
    return profile


def assess_domain(row: pd.Series, profile: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    numeric_violations = 0
    unseen_categories = 0

    for column, stats in profile.get("numeric", {}).items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        low = stats["min"]
        high = stats["max"]
        width = max(high - low, 1e-12)
        margin = 0.05 * width
        if value < low - margin or value > high + margin:
            numeric_violations += 1
            reasons.append(f"{column}=outside_training_range")
        elif value < low or value > high:
            reasons.append(f"{column}=edge_training_range")

    for column, seen in profile.get("categorical", {}).items():
        value = row.get(column)
        if pd.isna(value):
            continue
        if str(value) not in seen:
            unseen_categories += 1
            reasons.append(f"{column}=unseen")

    if unseen_categories > 0 or numeric_violations >= 2:
        return "out_of_domain", reasons
    if numeric_violations == 1 or reasons:
        return "edge", reasons
    return "in_domain", ["within_training_domain"]


def confidence_grade(interval_width: float, target_iqr: float, domain_status: str) -> str:
    if not np.isfinite(interval_width):
        return "D"
    normalized = interval_width / max(target_iqr, 1e-12)
    if domain_status == "out_of_domain":
        return "D"
    if domain_status == "edge":
        return "C" if normalized <= 1.25 else "D"
    if normalized <= 0.75:
        return "A"
    if normalized <= 1.25:
        return "B"
    return "C"


def decision_label(prediction: float, rank_score: float, grade: str, domain_status: str) -> str:
    if domain_status == "out_of_domain":
        return "needs_data"
    if grade in {"A", "B"} and rank_score > 0:
        return "test"
    if grade == "C" and prediction > 0:
        return "watch"
    return "reject"


def assign_decisions(rows: pd.DataFrame) -> list[str]:
    usable = rows[rows["domain_status"] != "out_of_domain"]["rank_score"].astype(float)
    if usable.empty:
        return ["needs_data" for _ in range(len(rows))]
    high_cut = float(usable.quantile(0.67))
    low_cut = float(usable.quantile(0.33))
    decisions: list[str] = []
    for _, row in rows.iterrows():
        status = str(row["domain_status"])
        grade = str(row["confidence_grade"])
        score = float(row["rank_score"])
        if status == "out_of_domain" or grade == "D":
            decisions.append("needs_data")
        elif grade in {"A", "B"} and score >= high_cut:
            decisions.append("test")
        elif grade in {"A", "B", "C"} and score >= low_cut:
            decisions.append("watch")
        else:
            decisions.append("reject")
    return decisions


def train_best_model(
    df: pd.DataFrame,
    target: str,
    group: str | None,
    exclude_columns: list[str] | None = None,
    test_size: float = 0.25,
    calibration_size: float = 0.25,
    alpha: float = 0.10,
    random_state: int = 42,
) -> tuple[TrainResult, pd.DataFrame, pd.DataFrame]:
    df = augment_smiles_descriptors(df)
    df = clean_training_frame(df, target)
    excluded = list(exclude_columns or [])
    if group:
        excluded.append(group)
    features = select_feature_columns(df, target, excluded)
    train_full, test = split_data(df, target, group, test_size, random_state)

    calibration_group = group if group and group in train_full.columns else None
    train_fit, calibration = split_data(
        train_full,
        target,
        calibration_group,
        calibration_size,
        random_state + 1,
    )
    if len(calibration) < 3:
        train_fit, calibration = train_test_split(
            train_full,
            test_size=min(0.4, max(0.2, 3 / len(train_full))),
            random_state=random_state + 1,
        )
        train_fit = train_fit.reset_index(drop=True)
        calibration = calibration.reset_index(drop=True)

    preprocessor = build_preprocessor(train_fit, features)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in model_candidates(random_state).items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(train_fit[features], train_fit[target])
        predictions = pipe.predict(test[features])
        metrics = regression_metrics(test[target], predictions)
        rows.append({"model": name, **metrics})
        fitted[name] = pipe

    leaderboard = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=True)
    best_name = str(leaderboard.iloc[0]["model"])
    best_model = fitted[best_name]

    calibration_predictions = best_model.predict(calibration[features])
    qhat = conformal_qhat(calibration[target], calibration_predictions, alpha)
    test_predictions = best_model.predict(test[features])
    metrics = regression_metrics(test[target], test_predictions)

    profile = build_domain_profile(train_fit, features)
    profile["target_iqr"] = float(
        np.subtract(*np.percentile(train_fit[target].astype(float), [75, 25]))
    )
    profile["train_rows"] = int(len(train_fit))
    profile["calibration_rows"] = int(len(calibration))
    profile["test_rows"] = int(len(test))
    profile["leaderboard"] = leaderboard.to_dict(orient="records")

    result = TrainResult(
        model_name=best_name,
        model=best_model,
        metrics=metrics,
        calibration_qhat=qhat,
        feature_columns=features,
        target_column=target,
        group_column=group,
        train_profile=profile,
        alpha=alpha,
    )

    prediction_frame = test.copy()
    prediction_frame["prediction"] = test_predictions
    prediction_frame["lower"] = prediction_frame["prediction"] - qhat
    prediction_frame["upper"] = prediction_frame["prediction"] + qhat
    prediction_frame["abs_error"] = (prediction_frame[target] - prediction_frame["prediction"]).abs()
    coverage = (
        (prediction_frame[target] >= prediction_frame["lower"])
        & (prediction_frame[target] <= prediction_frame["upper"])
    ).mean()
    result.metrics["interval_coverage"] = float(coverage)
    result.metrics["mean_interval_width"] = float((prediction_frame["upper"] - prediction_frame["lower"]).mean())
    result.metrics["test_rows"] = float(len(prediction_frame))

    return result, leaderboard, prediction_frame


def save_train_result(result: TrainResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    joblib.dump(bundle, out / "model.joblib")
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def screen_candidates(bundle: dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = augment_smiles_descriptors(candidates)
    features = bundle["feature_columns"]
    missing = [c for c in features if c not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate file is missing model feature columns: {missing}")

    model = bundle["model"]
    qhat = float(bundle["calibration_qhat"])
    target_iqr = float(bundle["train_profile"].get("target_iqr", qhat * 2))
    predictions = model.predict(candidates[features])

    rows = candidates.copy()
    rows["prediction"] = predictions
    rows["lower"] = rows["prediction"] - qhat
    rows["upper"] = rows["prediction"] + qhat
    rows["interval_width"] = rows["upper"] - rows["lower"]

    statuses: list[str] = []
    reasons: list[str] = []
    grades: list[str] = []
    scores: list[float] = []
    for _, row in rows.iterrows():
        status, reason_list = assess_domain(row, bundle["train_profile"])
        grade = confidence_grade(float(row["interval_width"]), target_iqr, status)
        if status == "out_of_domain":
            domain_penalty = max(abs(float(row["prediction"])), 2.0 * target_iqr)
        elif status == "edge":
            domain_penalty = 0.5 * target_iqr
        else:
            domain_penalty = 0.0
        uncertainty_penalty = 0.5 * float(row["interval_width"])
        score = float(row["prediction"]) - uncertainty_penalty - domain_penalty
        statuses.append(status)
        reasons.append(";".join(reason_list))
        grades.append(grade)
        scores.append(score)

    rows["domain_status"] = statuses
    rows["confidence_grade"] = grades
    rows["rank_score"] = scores
    rows["reason_codes"] = reasons
    decisions = assign_decisions(rows)
    rows["decision"] = decisions
    rows["decision_priority"] = rows["decision"].map(
        {"test": 0, "watch": 1, "needs_data": 2, "reject": 3}
    )
    return rows.sort_values(
        ["decision_priority", "rank_score", "prediction"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
