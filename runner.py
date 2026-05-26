"""CLI benchmark runner.

This is the single-environment entry point. It runs one or more adapters that
are *installed in the current environment* against a given model config, and
writes a self-describing JSON result file.

Because two versions of the same package cannot share an interpreter, the
multi-version comparison (released shapiq vs git-main shapiq) is done by
invoking this script once per environment — see ``run_all.sh`` / the notebook.

Example
-------
    python -m treeshap_bench.runner \\
        --adapters fasttreeshap_v1 fasttreeshap_v2 shapiq \\
        --model random_forest --n-estimators 50 --max-depth 4 \\
        --tasks sv interaction \\
        --sv-samples 200 --interaction-samples 25 --rounds 3 \\
        --out results/rf_depth4.json
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .adapters import get_adapter
from .core import Environment, benchmark, save_results
from .data import DataConfig, ModelConfig, get_model


def main() -> None:
    p = argparse.ArgumentParser(description="TreeSHAP library benchmark runner.")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="Adapter names, e.g. fasttreeshap_v1 shapiq")
    p.add_argument("--model", default="random_forest",
                   choices=["random_forest", "xgboost", "lightgbm"])
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--tasks", nargs="+", default=["sv"],
                   choices=["sv", "interaction"])
    p.add_argument("--sv-samples", type=int, default=200)
    p.add_argument("--interaction-samples", type=int, default=25)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-one-hot", action="store_true",
                   help="Ordinal-encode categoricals (far fewer features).")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--out", required=True, help="Path to write result JSON.")
    args = p.parse_args()

    data_cfg = DataConfig(one_hot=not args.no_one_hot)
    model_cfg = ModelConfig(
        kind=args.model, n_estimators=args.n_estimators, max_depth=args.max_depth
    )

    print(f"Preparing model: {model_cfg.kind} "
          f"(n_estimators={model_cfg.n_estimators}, max_depth={model_cfg.max_depth})")
    model, X_test = get_model(model_cfg, data_cfg)
    X_test_arr = X_test.values
    print(f"Model ready. Test matrix: {X_test_arr.shape}")

    results = []
    for task in args.tasks:
        n = args.sv_samples if task == "sv" else args.interaction_samples
        X = X_test_arr[:n]
        for adapter_name in args.adapters:
            adapter = get_adapter(adapter_name, n_jobs=args.n_jobs)
            if task not in adapter.supported_tasks:
                print(f"  skip {adapter_name} / {task} (unsupported)")
                continue
            print(f"  run  {adapter_name} / {task} on {X.shape[0]} samples ...", flush=True)
            res = benchmark(
                adapter, model, X,
                task=task, model_kind=model_cfg.kind,
                n_estimators=model_cfg.n_estimators, max_depth=model_cfg.max_depth,
                num_rounds=args.rounds, warmup=not args.no_warmup,
            )
            if res.error:
                print(f"       ERROR: {res.error}")
            else:
                print(f"       {res.mean_seconds:.3f} ± {res.std_seconds:.3f} s "
                      f"({res.per_sample_seconds*1000:.1f} ms/sample), "
                      f"setup {res.setup_seconds:.3f} s  [{adapter.library} {res.library_version}]")
            results.append(res)

    libraries = sorted({get_adapter(a).library for a in args.adapters})
    env = Environment.capture(libraries + ["numpy", "scikit-learn"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_results(args.out, environment=env, results=results,
                 extra={"one_hot": data_cfg.one_hot})
    print(f"\nWrote {len(results)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
