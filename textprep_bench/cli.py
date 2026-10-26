"""Command line interface: audit | run | leakage | demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_dataset, format_audit
from .data import load_csv, make_synthetic
from .experiment import run_experiment, run_leakage_experiment
from .report import write_outputs

SYN_BANNER = ("SYNTHETIC DATA - for demonstration only. These numbers say nothing about "
              "real-world behaviour and must not be reported as findings.")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="CSV file with a text column and a binary label column")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--positive-label", default=None,
                   help="value of the positive class if labels are not 0/1")
    p.add_argument("--synthetic", action="store_true", help="use built-in SYNTHETIC demo data")
    p.add_argument("--n", type=int, default=1500, help="rows of synthetic data")
    p.add_argument("--seed", type=int, default=42)


def _load(args):
    if args.synthetic:
        return make_synthetic(args.n, seed=args.seed), f"synthetic(n={args.n}, seed={args.seed})"
    if not args.data:
        raise SystemExit("error: pass --data FILE.csv or --synthetic")
    return load_csv(args.data, args.text_col, args.label_col, args.positive_label), str(args.data)


def _add_cv_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="textprep-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="data-quality audit of a dataset")
    _add_data_args(a)

    r = sub.add_parser("run", help="run the preprocessing ablation")
    _add_data_args(r)
    _add_cv_args(r)
    r.add_argument("--design", choices=["add-one", "drop-one", "both"], default="both")
    r.add_argument("--models", nargs="+", default=["logreg"], choices=["logreg", "svm", "nb"])
    r.add_argument("--metric", default="macro_f1",
                   choices=["macro_f1", "accuracy", "f1_pos", "roc_auc"])
    r.add_argument("--no-clean", action="store_true", help="skip duplicate/conflict removal")
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--out", default="results")

    lk = sub.add_parser("leakage", help="measure CV inflation caused by duplicates")
    _add_data_args(lk)
    _add_cv_args(lk)

    d = sub.add_parser("demo", help="quick end-to-end run on SYNTHETIC data")
    d.add_argument("--out", default="results/demo")

    args = ap.parse_args(argv)

    if args.cmd == "audit":
        df, src = _load(args)
        print(f"Data audit: {src}\n" + format_audit(audit_dataset(df)))
    elif args.cmd == "run":
        df, src = _load(args)
        bundle = run_experiment(df, args.design, tuple(args.models), args.folds, args.repeats,
                                args.seed, args.metric, not args.no_clean, args.n_jobs, src)
        path = write_outputs(bundle, args.out, banner=SYN_BANNER if args.synthetic else "")
        print(f"report written to {path}")
    elif args.cmd == "leakage":
        df, src = _load(args)
        print(json.dumps(run_leakage_experiment(df, args.folds, args.repeats, args.seed), indent=2))
    elif args.cmd == "demo":
        df = make_synthetic(1200, seed=0)
        bundle = run_experiment(df, "both", ("logreg",), 5, 2, 42, "macro_f1", True, 1,
                                "synthetic(n=1200, seed=0)")
        path = write_outputs(bundle, args.out, banner=SYN_BANNER)
        print(f"demo report written to {path}")


if __name__ == "__main__":
    main()
