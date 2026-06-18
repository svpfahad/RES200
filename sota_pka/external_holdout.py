"""E6 — External-holdout generalization test for the OP2-trained pKa models.

Scientific question (H3): do the OP2-trained leakage-safe symbolic pKa formulas
(and the LightGBM accuracy ceiling) generalize to *external* blind benchmarks,
or is their skill specific to the OP2 train/test split?

External benchmarks (curated, single-value, already acid/base split) come from the
SAT4pKa test suite shipped in ``data/raw_external/SAT4pKa/data/test/``:
  * Novartis      acidic (112) / basic (168)   — standard medicinal-chemistry set
  * AvLiLuMoVe123 acidic (26)  / basic (97)     — classic literature benchmark
  * SAMPL7        acidic (20)                   — blind challenge sulfonamides

Validity control that matters here: OP2 was featurized from ``OriginalSmiles`` —
the *neutral drawn structure* (acids protonated -COOH/-SO3H, amines neutral). The
SAT4pKa SMILES are the *ionized conjugate* forms (-COO-, -NH+). Feeding ionized
structures to a model trained on neutral ones would conflate "fails to generalize"
with "wrong protonation-state representation". We therefore **neutralize** every
external molecule (RDKit Uncharger + largest-fragment parent) so the representation
matches the training convention, then featurize through the *identical*
``featurize_op_split`` path. A raw (ionized) variant is also produced as a
representation-sensitivity check.

Protocol (leakage-safe, honest):
  * The symbolic model uses the *exact recorded selected features* (read from the
    stored summary JSON) and is re-fit on OP2-train only with deterministic seeds;
    we report the reproduced OP2-test R^2 next to the recorded value as a fidelity
    anchor. The generalization GAP is measured within that one model
    (R2_OP2test - R2_external), so it is internally valid regardless of exact
    feyn reproduction.
  * The LightGBM ceiling + ElasticNet/RF baselines are fit on OP2-train and scored
    once on each external set.
  * A y-randomization control (OP2 labels shuffled) is scored on every external set;
    R^2 must collapse to ~0, proving external skill is not an artefact of the
    descriptor space.

Run:
    .venv_mac/bin/python -m sota_pka.external_holdout            # full study
    .venv_mac/bin/python -m sota_pka.external_holdout --sets AvLiLuMoVe_123_acidic --reps neutral   # smoke
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import regression_metrics
from .qlattice_search import (
    _safe_names,
    _train_bic,
    fit_qlattice,
    model_formula,
)
from .featurize import featurize_op_split

TARGET = "pKa"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _op2_dir() -> Path:
    return _repo_root() / "RES 200-20260312T035531Z-1-001" / "RES 200"


def _ext_root() -> Path:
    return _repo_root() / "sota_pka" / "data" / "raw_external" / "SAT4pKa" / "data" / "test"


# OP2 descriptor matrices + recorded run dirs, per task.
TASKS = {
    "acidic": {
        "train_desc": _op2_dir() / "train_descriptors_op2.csv",
        "test_desc": _op2_dir() / "test_descriptors_op2.csv",
        "summary": _repo_root() / "sota_pka" / "runs" / "qlattice_acidic_op2" / "summary_acidic_direct.json",
    },
    "basic": {
        "train_desc": _op2_dir() / "train_descriptors_basic_op2.csv",
        "test_desc": _op2_dir() / "test_descriptors_basic_op2.csv",
        "summary": _repo_root() / "sota_pka" / "runs" / "qlattice_basic_op2" / "summary_basic_direct.json",
    },
}

# External benchmark sets (SAT4pKa). smiles col is "smiles", target "pKa".
EXTERNAL_SETS = [
    {"name": "novartis_acidic", "task": "acidic"},
    {"name": "AvLiLuMoVe_123_acidic", "task": "acidic"},
    {"name": "SAMPL7_acidic", "task": "acidic"},
    {"name": "novartis_basic", "task": "basic"},
    {"name": "AvLiLuMoVe_123_basic", "task": "basic"},
]


# --------------------------------------------------------------------------- #
# Neutralization
# --------------------------------------------------------------------------- #
def _standardizers():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.Uncharger(), rdMolStandardize.LargestFragmentChooser()


def neutralize(smiles: object, uncharger, fragment_chooser) -> str | None:
    """Largest-fragment parent + formal-charge neutralization -> neutral SMILES.

    Matches the OP2 ``OriginalSmiles`` (neutral drawn structure) convention so the
    external molecules are presented to the model in the same representation as
    training. Zwitterions / quaternary centers that cannot be neutralized are left
    as-is (correct behavior)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = fragment_chooser.choose(mol)
    except Exception:
        pass
    try:
        mol = uncharger.uncharge(mol)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# External featurization (cached)
# --------------------------------------------------------------------------- #
def featurize_external(name: str, rep: str, force: bool = False) -> Path:
    """Featurize one external set under representation ``rep`` in {neutral, raw}.

    Writes a temp SMILES+pKa CSV (column ``OriginalSmiles``) then runs the exact
    OP2 featurizer. Cached to data/processed/external/<name>__<rep>.csv."""
    out_dir = _repo_root() / "sota_pka" / "data" / "processed" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}__{rep}.csv"
    if out_path.exists() and not force:
        return out_path

    raw = pd.read_csv(_ext_root() / f"{name}.csv")
    smi = raw["smiles"].astype(str)
    pka = pd.to_numeric(raw["pKa"], errors="coerce")

    if rep == "neutral":
        uncharger, fragment_chooser = _standardizers()
        smi = smi.map(lambda s: neutralize(s, uncharger, fragment_chooser))
    elif rep != "raw":
        raise ValueError(f"unknown rep: {rep}")

    tmp = pd.DataFrame({"OriginalSmiles": smi, "pKa": pka})
    n_before = len(tmp)
    tmp = tmp[tmp["OriginalSmiles"].notna() & tmp["pKa"].notna()].reset_index(drop=True)
    tmp_path = out_dir / f"_tmp_{name}__{rep}.csv"
    tmp.to_csv(tmp_path, index=False)
    featurize_op_split(tmp_path, out_path, smiles_col="OriginalSmiles", target="pKa", strip_salts=True)
    tmp_path.unlink(missing_ok=True)
    out = pd.read_csv(out_path)
    print(f"[featurize] {name} ({rep}): {n_before} raw -> {out.shape[0]} featurized x {out.shape[1]} cols")
    return out_path


# --------------------------------------------------------------------------- #
# QLattice: re-fit recorded model on OP2-train, predict on arbitrary sets
# --------------------------------------------------------------------------- #
def _numeric_train(train: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean OP2-train numeric block; return (x_train, y_train, train_medians)."""
    y = pd.to_numeric(train[target], errors="coerce")
    x = train.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    x = x.loc[:, ~x.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True)
    mask = y.notna().to_numpy()
    return x.loc[mask].reset_index(drop=True), y[mask].reset_index(drop=True), medians


def fit_qlattice_recorded(
    train: pd.DataFrame,
    feats: list[str],
    target: str = TARGET,
    max_complexity: int = 30,
    criterion: str = "bic",
    refine_epochs: int = 120,
    refit_seeds: tuple[int, ...] = (0, 1, 2),
    threads: int = 8,
) -> dict:
    """Re-fit the symbolic model on OP2-train using the *exact recorded features*.

    Selection variance is eliminated (features are fixed); only the deterministic
    seeded feyn refit + BIC pick remains."""
    x_train, y_train, medians = _numeric_train(train, target)
    missing = [f for f in feats if f not in x_train.columns]
    if missing:
        raise ValueError(f"OP2-train is missing recorded features: {missing}")
    feat_medians = medians[feats]
    xf = x_train[feats].fillna(feat_medians).fillna(0.0)
    safe, inverse = _safe_names(feats)
    full = xf.rename(columns=safe).copy()
    full[target] = y_train.to_numpy(dtype=float)

    candidates = []
    for seed in refit_seeds:
        models = fit_qlattice(
            full, target, seed=seed, epochs=refine_epochs,
            max_complexity=max_complexity, criterion=criterion, threads=threads,
        )
        candidates.extend(models[:3])
    final = min(candidates, key=lambda m: _train_bic(m, full, y_train.to_numpy(dtype=float)))
    formula, ops = model_formula(final, inverse)
    return {
        "final": final,
        "feats": feats,
        "safe": safe,
        "inverse": inverse,
        "feat_medians": feat_medians,
        "formula": formula,
        "ops": ops,
        "edge_count": int(getattr(final, "edge_count", 0) or 0),
    }


def qlattice_predict(fit: dict, frame: pd.DataFrame, target: str = TARGET) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score a fitted symbolic model on any featurized frame (OP2-test or external)."""
    feats, safe, feat_medians = fit["feats"], fit["safe"], fit["feat_medians"]
    y = pd.to_numeric(frame[target], errors="coerce")
    x = frame.select_dtypes(include=[np.number]).copy().replace([np.inf, -np.inf], np.nan)
    cols = {}
    for f in feats:
        col = x[f] if f in x.columns else pd.Series(np.nan, index=x.index)
        cols[f] = col.fillna(feat_medians[f])
    xf = pd.DataFrame(cols).fillna(feat_medians).fillna(0.0)
    mask = y.notna().to_numpy()
    xf, y = xf.loc[mask], y[mask]
    pred = np.asarray(fit["final"].predict(xf.rename(columns=safe)), dtype=float)
    return regression_metrics(y.to_numpy(), pred), y.to_numpy(), pred


# --------------------------------------------------------------------------- #
# Baselines (LightGBM ceiling + ElasticNet/RF) — fit once, predict many
# --------------------------------------------------------------------------- #
def _full_numeric(train: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(train[target], errors="coerce")
    x = train.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    x = x.loc[:, ~x.columns.duplicated()].dropna(axis=1, how="all")
    finite = np.isfinite(x.replace([np.inf, -np.inf], np.nan)).any(axis=0)
    x = x.loc[:, finite]
    mask = y.notna().to_numpy()
    return x.loc[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def _align_ext(cols: list[str], frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(frame[target], errors="coerce")
    x = frame.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    x = x.loc[:, ~x.columns.duplicated()].reindex(columns=cols, fill_value=0.0)
    mask = y.notna().to_numpy()
    return x.loc[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def fit_baselines(train: pd.DataFrame, models: list[str], target: str = TARGET, seed: int = 42) -> dict:
    from .train_baselines import _model_factory

    x_train, y_train = _full_numeric(train, target)
    fitted = {}
    for name in models:
        est = _model_factory(name, seed=seed, random_forest_estimators=300)
        est.fit(x_train, y_train)
        fitted[name] = est
    return {"models": fitted, "cols": list(x_train.columns), "x_train": x_train, "y_train": y_train}


def baseline_predict(bundle: dict, name: str, frame: pd.DataFrame, target: str = TARGET):
    xe, ye = _align_ext(bundle["cols"], frame, target)
    pred = np.asarray(bundle["models"][name].predict(xe), dtype=float)
    return regression_metrics(ye.to_numpy(), pred), ye.to_numpy(), pred


def fit_yrandom(train: pd.DataFrame, target: str = TARGET, seed: int = 42):
    shuffled = train.copy()
    rng = np.random.default_rng(seed)
    shuffled[target] = rng.permutation(pd.to_numeric(shuffled[target], errors="coerce").to_numpy())
    return fit_baselines(shuffled, ["lightgbm"], target=target, seed=seed)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(prog="external_holdout")
    ap.add_argument("--sets", nargs="*", default=[s["name"] for s in EXTERNAL_SETS])
    ap.add_argument("--reps", nargs="*", default=["neutral", "raw"], choices=["neutral", "raw"])
    ap.add_argument("--baselines", nargs="*", default=["lightgbm", "elasticnet", "random_forest"])
    ap.add_argument("--refine-epochs", type=int, default=120)
    ap.add_argument("--skip-qlattice", action="store_true")
    ap.add_argument("--force-featurize", action="store_true")
    ap.add_argument("--out", default=str(_repo_root() / "sota_pka" / "runs" / "external_holdout"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = [s for s in EXTERNAL_SETS if s["name"] in args.sets]
    tasks_used = sorted({s["task"] for s in sets})

    # ---- Fit OP2-train models once per task ----
    op2 = {}
    for task in tasks_used:
        cfg = TASKS[task]
        train_df = pd.read_csv(cfg["train_desc"])
        test_df = pd.read_csv(cfg["test_desc"])
        recorded = json.loads(Path(cfg["summary"]).read_text())
        feats = recorded["selected_features"]
        rec_r2 = recorded["test_metrics"]["r2"]
        print(f"\n=== TASK {task}: OP2-train n={len(train_df)} feats={len(feats)} (recorded test R2={rec_r2:.4f}) ===")

        ql_fit = None
        ql_op2 = None
        if not args.skip_qlattice:
            ql_fit = fit_qlattice_recorded(train_df, feats, refine_epochs=args.refine_epochs)
            m, _, _ = qlattice_predict(ql_fit, test_df)
            ql_op2 = m
            print(f"  QLattice OP2-test reproduced: R2={m['r2']:.4f} RMSE={m['rmse']:.3f} "
                  f"(recorded {rec_r2:.4f}, ops={ql_fit['ops']})")

        base_fit = fit_baselines(train_df, args.baselines)
        base_op2 = {name: baseline_predict(base_fit, name, test_df)[0] for name in args.baselines}
        for name in args.baselines:
            print(f"  {name:14s} OP2-test: R2={base_op2[name]['r2']:.4f} RMSE={base_op2[name]['rmse']:.3f}")

        yr_fit = fit_yrandom(train_df)
        yr_op2 = baseline_predict(yr_fit, "lightgbm", test_df)[0]
        print(f"  y-random(LGBM) OP2-test: R2={yr_op2['r2']:.4f}")

        op2[task] = {
            "recorded_r2": rec_r2, "feats": feats,
            "ql_fit": ql_fit, "ql_op2": ql_op2,
            "base_fit": base_fit, "base_op2": base_op2,
            "yr_fit": yr_fit, "yr_op2": yr_op2,
        }

    # ---- Score every external set under every representation ----
    rows: list[dict] = []
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for entry in sets:
        name, task = entry["name"], entry["task"]
        store = op2[task]
        for rep in args.reps:
            feat_path = featurize_external(name, rep, force=args.force_featurize)
            frame = pd.read_csv(feat_path)

            def _emit(model_label, metrics, y, pred):
                rows.append({
                    "set": name, "task": task, "rep": rep, "model": model_label,
                    "n": metrics["n"], "r2": metrics["r2"], "rmse": metrics["rmse"], "mae": metrics["mae"],
                })
                pd.DataFrame({"y_true": y, "y_pred": pred}).to_csv(
                    pred_dir / f"{name}__{rep}__{model_label}.csv", index=False)

            if store["ql_fit"] is not None:
                m, y, p = qlattice_predict(store["ql_fit"], frame)
                _emit("qlattice", m, y, p)
            for bname in args.baselines:
                m, y, p = baseline_predict(store["base_fit"], bname, frame)
                _emit(bname, m, y, p)
            m, y, p = baseline_predict(store["yr_fit"], "lightgbm", frame)
            _emit("y_random_lightgbm", m, y, p)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "external_metrics.csv", index=False)

    # ---- Build a readable report with the OP2->external gap ----
    lines: list[str] = ["# E6 — External-holdout generalization (H3)\n"]
    for task in tasks_used:
        store = op2[task]
        lines.append(f"\n## Task: {task}\n")
        lines.append(f"- Recorded OP2-test R² (QLattice): **{store['recorded_r2']:.3f}**")
        if store["ql_op2"]:
            lines.append(f"- Reproduced OP2-test R² (QLattice): **{store['ql_op2']['r2']:.3f}** "
                         f"(RMSE {store['ql_op2']['rmse']:.2f})")
        for bname in args.baselines:
            lines.append(f"- OP2-test R² ({bname}): {store['base_op2'][bname]['r2']:.3f}")
        lines.append(f"- OP2-test R² (y-random LGBM): {store['yr_op2']['r2']:.3f}")
        lines.append("")
        sub = results[results["task"] == task]
        lines.append("| set | rep | model | n | R² | RMSE | MAE | ΔR² vs OP2-test |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            if r["model"] == "qlattice" and store["ql_op2"]:
                base = store["ql_op2"]["r2"]
            elif r["model"] in store["base_op2"]:
                base = store["base_op2"][r["model"]]["r2"]
            elif r["model"] == "y_random_lightgbm":
                base = store["yr_op2"]["r2"]
            else:
                base = float("nan")
            gap = base - r["r2"]
            lines.append(f"| {r['set']} | {r['rep']} | {r['model']} | {int(r['n'])} | "
                         f"{r['r2']:.3f} | {r['rmse']:.2f} | {r['mae']:.2f} | {gap:+.3f} |")
        lines.append("")

    report = "\n".join(lines)
    (out_dir / "external_holdout_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {out_dir/'external_metrics.csv'} and report.")


if __name__ == "__main__":
    main()
