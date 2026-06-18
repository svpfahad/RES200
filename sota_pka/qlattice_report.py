"""Paper-grade assets for the QLattice symbolic study.

Consumes the artifacts written by ``run_qlattice_all`` (summaries, predictions,
pareto, feature lists) and produces:
  * predicted-vs-experimental parity plots,
  * residual plots,
  * complexity-vs-accuracy Pareto plots,
  * Williams applicability-domain plots (leverage vs standardised residual),
  * a combined benchmark table (GBM baselines + symbolic models) per task,
  * a cross-model complexity/accuracy comparison figure.

Usage:
    .venv_mac/bin/python -m sota_pka.qlattice_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .qlattice_search import prepare_feature_frame

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "RES 200-20260312T035531Z-1-001" / "RES 200"
RUNS = Path(__file__).resolve().parent / "runs"
ASSETS = Path(__file__).resolve().parent / "paper_assets"

TASK_FILES = {
    "acidic": (DATA / "train_descriptors_op2.csv", DATA / "test_descriptors_op2.csv",
               RUNS / "res200_op2_full_descriptors_wsl_full" / "metrics.csv"),
    "basic": (DATA / "train_descriptors_basic_op2.csv", DATA / "test_descriptors_basic_op2.csv",
              RUNS / "res200_basic_op2_full_descriptors" / "metrics.csv"),
}


def _parity(pred_csv: Path, out_png: Path, title: str) -> None:
    df = pd.read_csv(pred_csv)
    yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=200)
    ax.scatter(yt, yp, s=12, alpha=0.5, edgecolor="none")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("Experimental pK$_a$")
    ax.set_ylabel("Predicted pK$_a$")
    ax.set_title(f"{title}\n$R^2$={r2:.3f}  RMSE={rmse:.2f}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _residuals(pred_csv: Path, out_png: Path, title: str) -> None:
    df = pd.read_csv(pred_csv)
    yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    resid = yp - yt
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
    ax.scatter(yp, resid, s=12, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Predicted pK$_a$")
    ax.set_ylabel("Residual (pred - exp)")
    ax.set_title(f"{title}  (mean={resid.mean():.2f}, sd={resid.std():.2f})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _pareto(pareto_csv: Path, out_png: Path, title: str) -> None:
    df = pd.read_csv(pareto_csv).dropna(subset=["complexity", "val_r2"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=200)
    for k, sub in df.groupby("top_k"):
        ax.scatter(sub["complexity"], sub["val_r2"], s=14, alpha=0.5, label=f"top_k={k}")
    # frontier: best val_r2 at each complexity
    frontier = df.sort_values("complexity").groupby("complexity")["val_r2"].max().cummax()
    ax.plot(frontier.index, frontier.values, "k-", lw=1.0, label="frontier")
    ax.set_xlabel("Formula complexity (sympy ops)")
    ax.set_ylabel("CV validation $R^2$")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _williams(task: str, features: list[str], pred_csv: Path, out_png: Path, title: str) -> None:
    train_csv, test_csv, _ = TASK_FILES[task]
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    x_train, x_test, _, _ = prepare_feature_frame(train, test)
    feats = [f for f in features if f in x_train.columns and f in x_test.columns]
    if not feats:
        return
    xtr = x_train[feats].to_numpy(dtype=float)
    xte = x_test[feats].to_numpy(dtype=float)
    # standardise on train
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1.0
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    gram = xtr.T @ xtr
    gram_inv = np.linalg.pinv(gram)
    h_test = np.einsum("ij,jk,ik->i", xte, gram_inv, xte)
    p = len(feats)
    n = xtr.shape[0]
    h_star = 3.0 * (p + 1) / n

    pred = pd.read_csv(pred_csv)
    resid = (pred["y_pred"] - pred["y_true"]).to_numpy()
    std_resid = resid / (resid.std() if resid.std() else 1.0)
    m = min(len(std_resid), len(h_test))
    h_test, std_resid = h_test[:m], std_resid[:m]

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=200)
    inside = (h_test <= h_star) & (np.abs(std_resid) <= 3)
    ax.scatter(h_test[inside], std_resid[inside], s=12, alpha=0.5, label="in-domain")
    ax.scatter(h_test[~inside], std_resid[~inside], s=14, alpha=0.7, color="crimson", label="outside")
    ax.axhline(3, color="k", lw=0.7, ls="--")
    ax.axhline(-3, color="k", lw=0.7, ls="--")
    ax.axvline(h_star, color="k", lw=0.7, ls=":")
    ax.set_xlabel(f"Leverage (h*  = {h_star:.3f})")
    ax.set_ylabel("Std. residual")
    ax.set_title(f"{title}  ({100*inside.mean():.0f}% in-domain)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return {"task": task, "h_star": float(h_star), "frac_in_domain": float(inside.mean())}


def _combined_benchmark(task: str, summaries: list[dict], out_csv: Path, out_md: Path) -> pd.DataFrame:
    _, _, baseline_csv = TASK_FILES[task]
    rows = []
    if Path(baseline_csv).exists():
        base = pd.read_csv(baseline_csv)
        for _, r in base.iterrows():
            if str(r.get("status")) != "ok":
                continue
            rows.append({
                "model": r["model"], "test_r2": r.get("test_r2"), "test_rmse": r.get("test_rmse"),
                "test_mae": r.get("test_mae"), "n_features": r.get("n_features"),
                "complexity_ops": "", "type": "GBM baseline",
            })
    for s in summaries:
        if s.get("task") != task:
            continue
        rows.append({
            "model": f"QLattice ({s['mode']})", "test_r2": s["test_metrics"]["r2"],
            "test_rmse": s["test_metrics"]["rmse"], "test_mae": s["test_metrics"]["mae"],
            "n_features": s["chosen_top_k"], "complexity_ops": s["sympy_ops"],
            "type": "symbolic",
        })
    table = pd.DataFrame(rows).sort_values("test_rmse")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    # markdown
    lines = [f"### {task.capitalize()} pKa — model comparison (OP2 held-out test)", "",
             "| Model | Type | Test R² | Test RMSE | Test MAE | # feats | Complexity |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        comp = r["complexity_ops"] if r["complexity_ops"] != "" else "—"
        lines.append(f"| {r['model']} | {r['type']} | {r['test_r2']:.3f} | {r['test_rmse']:.3f} "
                     f"| {r['test_mae']:.3f} | {r['n_features']} | {comp} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def build_assets() -> None:
    summaries = json.loads((RUNS / "qlattice_combined_summary.json").read_text())
    ad_rows = []
    for s in summaries:
        task, mode = s["task"], s["mode"]
        run_dir = RUNS / f"qlattice_{task}_op2"
        out_dir = ASSETS / f"qlattice_{task}_op2"
        label = f"{task}_{mode}"
        pred_csv = run_dir / f"predictions_qlattice_{label}.csv"
        if pred_csv.exists():
            _parity(pred_csv, out_dir / f"parity_{label}.png", f"{task} / {mode}")
            _residuals(pred_csv, out_dir / f"residuals_{label}.png", f"{task} / {mode}")
        pareto_csv = run_dir / f"pareto_{label}.csv"
        if pareto_csv.exists():
            _pareto(pareto_csv, out_dir / f"pareto_{label}.png", f"{task} / {mode}: complexity vs CV $R^2$")
        if mode == "direct" and pred_csv.exists():
            ad = _williams(task, s["selected_features"], pred_csv, out_dir / f"williams_{label}.png",
                           f"{task} applicability domain")
            if ad:
                ad_rows.append(ad)

    for task in {s["task"] for s in summaries}:
        out_dir = ASSETS / f"qlattice_{task}_op2"
        _combined_benchmark(task, summaries, out_dir / "combined_benchmark.csv", out_dir / "combined_benchmark.md")

    if ad_rows:
        pd.DataFrame(ad_rows).to_csv(ASSETS / "applicability_domain_summary.csv", index=False)

    # Cross-model complexity vs accuracy
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
    for s in summaries:
        ax.scatter(s["sympy_ops"], s["test_metrics"]["r2"], s=40)
        ax.annotate(f"{s['task'][:3]}/{s['mode'][:4]}", (s["sympy_ops"], s["test_metrics"]["r2"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Formula complexity (sympy ops)")
    ax.set_ylabel("Test $R^2$")
    ax.set_title("Symbolic models: complexity vs held-out accuracy")
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "qlattice_complexity_vs_accuracy.png")
    plt.close(fig)
    print("Assets written under", ASSETS)


if __name__ == "__main__":
    build_assets()
