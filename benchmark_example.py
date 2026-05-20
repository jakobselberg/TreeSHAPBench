#!/usr/bin/env python
"""
Example benchmarking script for TreeSHAP implementations.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_dataset
from benchmarker import TreeBenchmarker


def main():
    """Run benchmarking example."""
    print("=" * 80)
    print("TreeSHAP Implementation Benchmarking")
    print("=" * 80)

    # ── Classification ────────────────────────────────────────────────────────

    print("\n1. Loading Census Income dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        'census_income',
        data_dir='data',
        test_size=0.2,
        random_state=42
    )
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set:  {X_test.shape}")
    print(f"   Features:  {len(feature_names)}")

    print("\n2. Training XGBoost classifier...")
    benchmarker = TreeBenchmarker(model_type='xgboost', task='classification')
    benchmarker.train_model(X_train, y_train, verbose=0)
    benchmarker.evaluate_model(X_test, y_test)

    print("\n3. Benchmarking implementations (classification)...")
    cls_results = {
        'shap':            benchmarker.benchmark_shap(X_test[:100], algorithm='shap'),
        'woodelf':         benchmarker.benchmark_shap(X_test[:100], algorithm='woodelf'),
        'fastTreeShap-v1': benchmarker.benchmark_shap(X_test[:100], algorithm='fastTreeShap-v1'),
        'fastTreeShap-v2': benchmarker.benchmark_shap(X_test[:100], algorithm='fastTreeShap-v2'),
    }

    # ── Regression ────────────────────────────────────────────────────────────

    print("\n4. Loading Superconductivity dataset (regression)...")
    X_train_sc, X_test_sc, y_train_sc, y_test_sc, feature_names_sc = load_dataset(
        'superconductivity',
        data_dir='data',
        test_size=0.2,
        random_state=42
    )
    print(f"   Train set: {X_train_sc.shape}")
    print(f"   Test set:  {X_test_sc.shape}")
    print(f"   Features:  {len(feature_names_sc)}")

    print("\n5. Training XGBoost regressor...")
    benchmarker_reg = TreeBenchmarker(model_type='xgboost', task='regression')
    benchmarker_reg.train_model(X_train_sc, y_train_sc, verbose=0)
    benchmarker_reg.evaluate_model(X_test_sc, y_test_sc)

    print("\n6. Benchmarking implementations (regression)...")
    reg_results = {
        'shap':            benchmarker_reg.benchmark_shap(X_test_sc[:100], algorithm='shap'),
        'woodelf':         benchmarker_reg.benchmark_shap(X_test_sc[:100], algorithm='woodelf'),
        'fastTreeShap-v1': benchmarker_reg.benchmark_shap(X_test_sc[:100], algorithm='fastTreeShap-v1'),
        'fastTreeShap-v2': benchmarker_reg.benchmark_shap(X_test_sc[:100], algorithm='fastTreeShap-v2'),
    }

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("SUMMARY — Speed")
    print("=" * 80)

    for group, results in [("Classification", cls_results), ("Regression", reg_results)]:
        print(f"\n  {group}")
        print(f"  {'Implementation':<20} {'Compute time':>14} {'Avg/sample':>12}")
        print(f"  {'-'*20} {'-'*14} {'-'*12}")
        for name, res in results.items():
            if res.get('success'):
                print(f"  {name:<20} {res['computation_time']:>12.4f}s {res['avg_time_per_sample']*1000:>10.4f}ms")
            else:
                print(f"  {name:<20} {'ERROR':>14}  {res.get('error', '')[:40]}")

    print("\n" + "=" * 80)
    print("SUMMARY — Correctness (vs shap baseline)")
    print("=" * 80)

    print("\n  Classification")
    TreeBenchmarker.compare_shap_values(cls_results, reference='shap')

    print("\n  Regression")
    TreeBenchmarker.compare_shap_values(reg_results, reference='shap')


if __name__ == '__main__':
    main()
