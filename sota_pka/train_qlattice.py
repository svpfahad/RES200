from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import regression_metrics, write_predictions


def run_qlattice_grid(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    target: str = "pKa",
    top_k_values: list[int] | None = None,
    criteria: list[str] | None = None,
    complexities: list[int] | None = None,
    epochs: list[int] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run a Feyn QLattice search on numeric top-K features.

    If Feyn is not installed, a status file is written and the caller can still
    complete the benchmark without pretending QLattice was run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import feyn
    except Exception as exc:
        status = {"status": "skipped", "reason": f"Feyn/QLattice unavailable: {exc}"}
        (output_dir / "qlattice_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return pd.DataFrame([status])

    top_k_values = top_k_values or [10, 20, 40, 80, 150]
    criteria = criteria or ["bic", "aic"]
    complexities = complexities or [20, 50, 100, 200, 300]
    epochs = epochs or [100, 200, 500]
    seeds = seeds or [0, 1, 2, 3, 4]

    numeric_train = train.select_dtypes(include="number").dropna(axis=1, how="any")
    numeric_test = test[numeric_train.columns.intersection(test.columns)].select_dtypes(include="number").dropna(axis=1, how="any")
    common = [column for column in numeric_train.columns if column in numeric_test.columns and column != target]
    selector = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    selector.fit(numeric_train[common], train[target])
    importances = pd.Series(selector.feature_importances_, index=common).sort_values(ascending=False)

    results: list[dict[str, object]] = []
    for top_k, criterion, complexity, n_epochs, seed in product(top_k_values, criteria, complexities, epochs, seeds):
        features = importances.head(top_k).index.tolist()
        q_train = train[[target] + features].dropna().copy()
        q_test = test[[target] + features].dropna().copy()
        ql = feyn.QLattice(random_seed=seed) if "random_seed" in feyn.QLattice.__init__.__code__.co_varnames else feyn.QLattice()
        models = ql.auto_run(
            q_train,
            output_name=target,
            kind="regression",
            n_epochs=n_epochs,
            max_complexity=complexity,
            criterion=criterion,
            threads=16,
        )
        if not models:
            continue
        model = models[0]
        pred = model.predict(q_test)
        metrics = regression_metrics(q_test[target], pred)
        result = {
            "status": "ok",
            "top_k": top_k,
            "criterion": criterion,
            "max_complexity": complexity,
            "n_epochs": n_epochs,
            "seed": seed,
            **metrics,
        }
        try:
            result["formula"] = str(model.sympify(signif=5).as_expr())
        except Exception:
            result["formula"] = str(model)
        results.append(result)
        write_predictions(
            output_dir / f"predictions_qlattice_k{top_k}_{criterion}_c{complexity}_e{n_epochs}_s{seed}.csv",
            q_test[target],
            pred,
            result,
        )
    results_df = pd.DataFrame(results).sort_values("rmse") if results else pd.DataFrame()
    results_df.to_csv(output_dir / "qlattice_metrics.csv", index=False)
    return results_df
