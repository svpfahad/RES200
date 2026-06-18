from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import regression_metrics, write_predictions


def align_numeric_features(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y_train = pd.to_numeric(train[target], errors="coerce")
    y_test = pd.to_numeric(test[target], errors="coerce")
    x_train = train.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    x_test = test.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    x_train = x_train.loc[:, ~x_train.columns.duplicated()]
    x_test = x_test.loc[:, ~x_test.columns.duplicated()]
    x_train = x_train.dropna(axis=1, how="all")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
    finite_columns = np.isfinite(x_train.replace([np.inf, -np.inf], np.nan)).any(axis=0)
    x_train = x_train.loc[:, finite_columns]
    x_test = x_test[x_train.columns]
    return x_train, x_test, y_train, y_test


def _model_factory(name: str, seed: int, random_forest_estimators: int):
    if name == "elasticnet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.05, max_iter=20000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=random_forest_estimators,
                        random_state=seed,
                        n_jobs=-1,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=seed, n_jobs=-1)),
            ]
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, random_seed=seed, verbose=False),
                ),
            ]
        )
    raise ValueError(f"Unknown baseline model: {name}")


def train_sklearn_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    target: str = "pKa",
    models: list[str] | None = None,
    seed: int = 42,
    random_forest_estimators: int = 300,
) -> pd.DataFrame:
    models = models or ["elasticnet", "random_forest", "xgboost", "lightgbm", "catboost"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    x_train, x_test, y_train, y_test = align_numeric_features(train, test, target)
    if x_train.empty:
        raise ValueError("No numeric features available after train/test alignment")

    metric_rows: list[dict[str, object]] = []
    for model_name in models:
        try:
            estimator = _model_factory(model_name, seed=seed, random_forest_estimators=random_forest_estimators)
            estimator.fit(x_train, y_train)
            train_pred = estimator.predict(x_train)
            test_pred = estimator.predict(x_test)
        except ImportError as exc:
            metric_rows.append({"model": model_name, "status": f"missing dependency: {exc.name}"})
            continue

        train_metrics = regression_metrics(y_train, train_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        row = {
            "model": model_name,
            "status": "ok",
            "n_features": int(x_train.shape[1]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        metric_rows.append(row)
        write_predictions(output_dir / f"predictions_{model_name}.csv", y_test, test_pred, {"model": model_name})
        with (output_dir / "models" / f"{model_name}.pkl").open("wb") as handle:
            pickle.dump(estimator, handle)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    return metrics


def train_y_randomization_control(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    model: str = "xgboost",
    target: str = "pKa",
    seed: int = 42,
    random_forest_estimators: int = 300,
) -> pd.DataFrame:
    shuffled = train.copy()
    rng = np.random.default_rng(seed)
    shuffled[target] = rng.permutation(pd.to_numeric(shuffled[target], errors="coerce").to_numpy())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = align_numeric_features(shuffled, test, target)
    estimator = _model_factory(model, seed=seed, random_forest_estimators=random_forest_estimators)
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)
    metrics = regression_metrics(y_test, pred)
    row = {"control": "y_randomization", "model": model, "seed": seed, **metrics}
    write_predictions(output_dir / f"predictions_y_randomization_{model}.csv", y_test, pred, row)
    result = pd.DataFrame([row])
    result.to_csv(output_dir / f"metrics_y_randomization_{model}.csv", index=False)
    return result
