from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import prepare_res200_dataset, prepare_res200_plus_iupac_dataset
from .featurize import featurize_op_split
from .make_paper_assets import make_metric_table, make_prediction_plots
from .train_baselines import train_sklearn_baselines, train_y_randomization_control
from .train_neural import write_neural_integration_status
from .train_qlattice import run_qlattice_grid


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prepare_data(args: argparse.Namespace) -> None:
    if args.iupac_file:
        rows = prepare_res200_plus_iupac_dataset(Path(args.data_root), Path(args.iupac_file), Path(args.output))
    else:
        rows = prepare_res200_dataset(Path(args.data_root), Path(args.output))
    print(f"Wrote {len(rows)} standardized rows to {args.output}")


def train_baselines(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    metrics = train_sklearn_baselines(
        train,
        test,
        output_dir=Path(args.output_dir),
        target=args.target,
        models=args.models,
        seed=args.seed,
        random_forest_estimators=args.random_forest_estimators,
    )
    print(metrics.to_string(index=False))


def y_randomization(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    result = train_y_randomization_control(
        train,
        test,
        output_dir=Path(args.output_dir),
        model=args.model,
        target=args.target,
        seed=args.seed,
        random_forest_estimators=args.random_forest_estimators,
    )
    print(result.to_string(index=False))


def train_qlattice(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    results = run_qlattice_grid(train, test, output_dir=Path(args.output_dir), target=args.target)
    print(results.head(10).to_string(index=False))


def featurize_op(args: argparse.Namespace) -> None:
    out = featurize_op_split(
        Path(args.input),
        Path(args.output),
        smiles_col=args.smiles_col,
        target=args.target,
        strip_salts=not args.no_salt_strip,
    )
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]} cols to {args.output}")


def qlattice_experiment(args: argparse.Namespace) -> None:
    from .qlattice_search import run_qlattice_experiment

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    teacher_factory = None
    if args.mode == "distilled":
        def teacher_factory():
            from lightgbm import LGBMRegressor

            return LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=args.seed, n_jobs=-1, verbose=-1)

    summary = run_qlattice_experiment(
        train,
        test,
        output_dir=Path(args.output_dir),
        target=args.target,
        mode=args.mode,
        top_k_grid=tuple(args.top_k),
        max_complexity=args.max_complexity,
        criterion=args.criterion,
        cv_folds=args.cv_folds,
        coarse_epochs=args.coarse_epochs,
        refine_epochs=args.refine_epochs,
        refit_seeds=tuple(args.refit_seeds),
        corr_threshold=args.corr_threshold,
        threads=args.threads,
        teacher_factory=teacher_factory,
        label=args.label or args.mode,
    )
    print(f"[{summary['label']}] chosen_k={summary['chosen_top_k']} "
          f"complexity(ops)={summary['sympy_ops']} edges={summary['edge_count']}")
    print(f"  test R2={summary['test_metrics']['r2']:.4f} "
          f"RMSE={summary['test_metrics']['rmse']:.4f} MAE={summary['test_metrics']['mae']:.4f}")
    print(f"  formula: {summary['formula']}")


def train_neural(args: argparse.Namespace) -> None:
    status = write_neural_integration_status(Path(args.output_dir))
    print(status)


def make_assets(args: argparse.Namespace) -> None:
    table = make_metric_table(Path(args.run_dir), Path(args.output_dir))
    make_prediction_plots(Path(args.run_dir), Path(args.output_dir))
    print(table.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sota-pka")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare_data")
    p.add_argument("--data-root", default=str(_repo_root().parent / "RES 200-20260312T035531Z-1-001" / "RES 200"))
    p.add_argument("--output", default=str(_repo_root() / "data" / "processed" / "res200_standardized.csv"))
    p.add_argument("--iupac-file", default="")
    p.set_defaults(func=prepare_data)

    p = sub.add_parser("train_baselines")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="pKa")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="+", default=["elasticnet", "random_forest", "xgboost", "lightgbm", "catboost"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-forest-estimators", type=int, default=300)
    p.set_defaults(func=train_baselines)

    p = sub.add_parser("y_randomization")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="pKa")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="xgboost")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-forest-estimators", type=int, default=300)
    p.set_defaults(func=y_randomization)

    p = sub.add_parser("train_qlattice")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="pKa")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=train_qlattice)

    p = sub.add_parser("featurize_op")
    p.add_argument("--input", required=True, help="OP split CSV with SMILES + pKa")
    p.add_argument("--output", required=True)
    p.add_argument("--smiles-col", default="OriginalSmiles")
    p.add_argument("--target", default="pKa")
    p.add_argument("--no-salt-strip", action="store_true")
    p.set_defaults(func=featurize_op)

    p = sub.add_parser("qlattice_experiment")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="pKa")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["direct", "distilled"], default="direct")
    p.add_argument("--label", default="")
    p.add_argument("--top-k", type=int, nargs="+", default=[8, 12, 20])
    p.add_argument("--max-complexity", type=int, default=30)
    p.add_argument("--criterion", default="bic")
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--coarse-epochs", type=int, default=40)
    p.add_argument("--refine-epochs", type=int, default=120)
    p.add_argument("--refit-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--corr-threshold", type=float, default=0.95)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=qlattice_experiment)

    p = sub.add_parser("train_neural")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=train_neural)

    p = sub.add_parser("make_paper_assets")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=make_assets)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
