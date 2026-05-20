"""
Benchmarking module for comparing TreeSHAP implementations.
"""

import json
import os
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import shap

# Python 3.10 venv that has fastTreeShap installed
_REPO_ROOT = Path(__file__).parent.parent
_FASTTREESHAP_PYTHON = _REPO_ROOT / ".venv-fasttreeshap" / "bin" / "python"
_RUNNER_SCRIPT = Path(__file__).parent / "fasttreeshap_runner.py"


class TreeBenchmarker:
    """Benchmark TreeSHAP implementations on tree models."""
    
    def __init__(self, model_type: str = 'xgboost', task: str = 'classification'):
        """
        Initialize benchmarker.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'random_forest'
            task: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.explainer = None
        self._X_train = None
        self._y_train = None
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Train a tree-based model."""
        if self.model_type == 'xgboost':
            if self.task == 'classification':
                self.model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, **kwargs)
            else:
                self.model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, **kwargs)
        
        elif self.model_type == 'lightgbm':
            if self.task == 'classification':
                self.model = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, **kwargs)
            else:
                self.model = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, **kwargs)
        
        elif self.model_type == 'random_forest':
            if self.task == 'classification':
                self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, **kwargs)
            else:
                self.model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self._X_train = X_train
        self._y_train = y_train
        print(f"Model trained. Train score: {self.model.score(X_train, y_train):.4f}")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Report test-set accuracy (classification) or R² (regression)."""
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.model.predict(X_test)
        if self.task == 'classification':
            score = accuracy_score(y_test, y_pred)
            print(f"  Test accuracy: {score:.4f}")
            return {'accuracy': score}
        else:
            score = r2_score(y_test, y_pred)
            print(f"  Test R²: {score:.4f}")
            return {'r2': score}

    @staticmethod
    def _normalize_shap(values) -> np.ndarray:
        """Reduce any SHAP output to a 2D (n_samples, n_features) float array.

        For binary classification shap returns a list of two arrays or a 3D
        array; we take the positive-class slice to get a consistent shape.
        """
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        arr = np.array(values, dtype=float)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        return arr

    @staticmethod
    def compare_shap_values(
        all_results: Dict[str, Dict],
        reference: str = 'shap',
    ) -> None:
        """Print a correctness table comparing each implementation vs reference.

        Args:
            all_results: dict mapping algorithm name → result dict from benchmark_shap().
            reference:   which algorithm to treat as the baseline.
        """
        ref = all_results.get(reference)
        if ref is None or 'shap_values' not in ref:
            print(f"Reference '{reference}' not available for comparison.")
            return

        ref_vals = TreeBenchmarker._normalize_shap(ref['shap_values'])

        notes = {
            'woodelf':         '(interventional SHAP — different flavor from shap baseline)',
            'fastTreeShap-v1': '(retrained on XGBoost 1.7.6 — model differs from baseline)',
            'fastTreeShap-v2': '(retrained on XGBoost 1.7.6 — model differs from baseline)',
        }

        print(f"\n  {'Implementation':<20} {'MAE':>10} {'Max-AE':>10}  Note")
        print(f"  {'-'*20} {'-'*10} {'-'*10}  {'-'*20}")
        for name, res in all_results.items():
            if name == reference:
                continue
            if not res.get('success') or 'shap_values' not in res:
                print(f"  {name:<20} {'N/A':>10} {'N/A':>10}  (not available)")
                continue
            vals = TreeBenchmarker._normalize_shap(res['shap_values'])
            if vals.shape != ref_vals.shape:
                print(f"  {name:<20} {'N/A':>10} {'N/A':>10}  (shape mismatch: {vals.shape} vs {ref_vals.shape})")
                continue
            diff = np.abs(vals - ref_vals)
            mae    = diff.mean()
            max_ae = diff.max()
            note = notes.get(name, '')
            print(f"  {name:<20} {mae:>10.4f} {max_ae:>10.4f}  {note}")
    
    @property
    def _registry(self) -> Dict[str, Any]:
        """
        Maps algorithm name → benchmark callable (X_sample, results) -> dict.

        To add a new library:
          1. Write _benchmark_<name>(self, X_sample, results) -> dict
          2. Add an entry here.
        """
        return {
            'shap':             self._benchmark_shap,
            'fastTreeShap-v1':  lambda X, r: self._benchmark_fast_treeshap(X, 'v1', r),
            'fastTreeShap-v2':  lambda X, r: self._benchmark_fast_treeshap(X, 'v2', r),
            'woodelf':          self._benchmark_woodelf,
        }

    def benchmark_shap(self, X_test: np.ndarray, num_samples: int = None,
                       algorithm: str = 'shap') -> Dict[str, Any]:
        """
        Benchmark a SHAP implementation.

        Args:
            X_test: Test features
            num_samples: Number of samples to use (None = all)
            algorithm: One of the keys in _registry.

        Returns:
            Dictionary with timing and result info
        """
        if num_samples is None:
            num_samples = len(X_test)

        X_sample = X_test[:num_samples]
        results = {
            'implementation': algorithm,
            'algorithm': algorithm,
            'num_samples': num_samples,
            'feature_dim': X_test.shape[1],
        }

        impl = self._registry.get(algorithm)
        if impl is None:
            results['error'] = f"Unknown algorithm '{algorithm}'. Available: {list(self._registry)}"
            results['success'] = False
            return results

        try:
            return impl(X_sample, results)
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            return results

    def _benchmark_fast_treeshap(self, X_sample: np.ndarray, variant: str, results: Dict) -> Dict:
        """Run fastTreeShap inside the dedicated Python 3.10 venv via subprocess.

        Passes raw numpy data + a config JSON so the runner trains its own model,
        avoiding cross-version XGBoost binary format incompatibilities.
        Requires benchmark_shap() to have been called after train_model(), and
        that X_train / y_train were stored via store_train_data().
        """
        if not _FASTTREESHAP_PYTHON.exists():
            results['error'] = (
                f"fastTreeShap venv not found at {_FASTTREESHAP_PYTHON}. "
                "Run: /opt/homebrew/bin/python3.10 -m venv .venv-fasttreeshap && "
                ".venv-fasttreeshap/bin/pip install 'numpy==1.24.*' fasttreeshap shap xgboost lightgbm scikit-learn"
            )
            results['success'] = False
            return results

        if self._X_train is None or self._y_train is None:
            results['error'] = "Call store_train_data(X_train, y_train) before benchmarking fastTreeShap."
            results['success'] = False
            return results

        print(f"\nBenchmarking fastTreeShap-{variant} (subprocess) with {len(X_sample)} samples...")

        config = {"model_type": self.model_type, "task": self.task}

        with tempfile.TemporaryDirectory() as tmpdir:
            X_train_path = os.path.join(tmpdir, "X_train.npy")
            y_train_path = os.path.join(tmpdir, "y_train.npy")
            X_test_path  = os.path.join(tmpdir, "X_test.npy")
            config_path  = os.path.join(tmpdir, "config.json")
            result_path       = os.path.join(tmpdir, "result.json")
            shap_values_path  = os.path.join(tmpdir, "shap_values.npy")

            np.save(X_train_path, self._X_train)
            np.save(y_train_path, self._y_train)
            np.save(X_test_path, X_sample)
            with open(config_path, "w") as f:
                json.dump(config, f)

            proc = subprocess.run(
                [str(_FASTTREESHAP_PYTHON), str(_RUNNER_SCRIPT),
                 X_train_path, y_train_path, X_test_path, config_path,
                 result_path, shap_values_path, variant],
                capture_output=True,
            )

            if proc.returncode != 0:
                results['error'] = proc.stderr.decode(errors='replace').strip()
                results['success'] = False
                return results

            with open(result_path) as f:
                data = json.load(f)
            if os.path.exists(shap_values_path):
                data['shap_values'] = np.load(shap_values_path)
        results.update(data)

        if results.get('success'):
            print(f"  Explainer creation: {results['explainer_creation_time']:.4f}s")
            print(f"  SHAP computation: {results['computation_time']:.4f}s")
            print(f"  Avg per sample: {results['avg_time_per_sample']:.6f}s")

        return results

    def _benchmark_woodelf(self, X_sample: np.ndarray, results: Dict) -> Dict:
        """Benchmark WoodelfExplainer (O(n+m) interventional SHAP)."""
        import pandas as pd
        from woodelf import WoodelfExplainer

        print(f"\nBenchmarking woodelf with {len(X_sample)} samples...")

        cols = [f"f{i}" for i in range(X_sample.shape[1])]
        X_train_df = pd.DataFrame(self._X_train, columns=cols)
        X_sample_df = pd.DataFrame(X_sample, columns=cols)

        start_time = time.time()
        explainer = WoodelfExplainer(self.model, X_train_df)
        explainer_creation_time = time.time() - start_time

        start_time = time.time()
        shap_values = explainer.shap_values(X_sample_df)
        computation_time = time.time() - start_time

        results['explainer_creation_time'] = explainer_creation_time
        results['computation_time'] = computation_time
        results['avg_time_per_sample'] = computation_time / len(X_sample)
        results['shap_values'] = shap_values
        results['shap_values_shape'] = list(np.array(shap_values).shape)
        results['success'] = True

        print(f"  Explainer creation: {explainer_creation_time:.4f}s")
        print(f"  SHAP computation:   {computation_time:.4f}s")
        print(f"  Avg per sample:     {results['avg_time_per_sample']:.6f}s")

        return results

    def _benchmark_shap(self, X_sample: np.ndarray, results: Dict) -> Dict:
        """Benchmark shap.TreeExplainer (standard algorithm)."""
        print(f"\nBenchmarking shap.TreeExplainer with {len(X_sample)} samples...")

        start_time = time.time()
        explainer = shap.TreeExplainer(self.model)
        explainer_creation_time = time.time() - start_time

        start_time = time.time()
        shap_values = explainer.shap_values(X_sample)
        computation_time = time.time() - start_time

        results['explainer_creation_time'] = explainer_creation_time
        results['computation_time'] = computation_time
        results['avg_time_per_sample'] = computation_time / len(X_sample)
        results['shap_values'] = shap_values
        results['success'] = True

        if isinstance(shap_values, list):
            results['shap_values_shape'] = [sv.shape for sv in shap_values]
        else:
            results['shap_values_shape'] = shap_values.shape

        print(f"  Explainer creation: {explainer_creation_time:.4f}s")
        print(f"  SHAP computation:   {computation_time:.4f}s")
        print(f"  Avg per sample:     {results['avg_time_per_sample']:.6f}s")

        return results
