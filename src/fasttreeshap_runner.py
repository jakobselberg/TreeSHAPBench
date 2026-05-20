"""
Subprocess runner for fastTreeShap benchmarks inside .venv-fasttreeshap (Python 3.10).

Trains its own model from raw numpy data so there are no cross-version pickle
or XGBoost binary-format incompatibilities. Returns results via a JSON file.

Invoked by benchmarker.py:
    .venv-fasttreeshap/bin/python src/fasttreeshap_runner.py \
        <X_train.npy> <y_train.npy> <X_test.npy> <config.json> <result.json> <shap_values.npy> [v1|v2]
"""

import json
import sys
import time

import numpy as np


def train_model(X_train, y_train, model_type: str, task: str):
    if model_type == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if task == "classification" else XGBRegressor
        model = cls(n_estimators=100, max_depth=6, random_state=42)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        model = cls(n_estimators=100, max_depth=6, random_state=42)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = cls(n_estimators=100, max_depth=6, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def run(X_train_path, y_train_path, X_test_path, config_path, algorithm="v2"):
    import fasttreeshap

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)

    with open(config_path) as f:
        config = json.load(f)

    model = train_model(X_train, y_train, config["model_type"], config["task"])

    start = time.time()
    explainer = fasttreeshap.TreeExplainer(model, algorithm=algorithm, n_jobs=-1)
    explainer_time = time.time() - start

    start = time.time()
    shap_values = explainer(X_test, check_additivity=False)
    computation_time = time.time() - start

    values = shap_values.values
    return values, {
        "implementation": f"fastTreeShap-{algorithm}",
        "num_samples": len(X_test),
        "feature_dim": X_test.shape[1],
        "explainer_creation_time": explainer_time,
        "computation_time": computation_time,
        "avg_time_per_sample": computation_time / len(X_test),
        "shap_values_shape": list(values.shape),
        "success": True,
    }


if __name__ == "__main__":
    X_train_path      = sys.argv[1]
    y_train_path      = sys.argv[2]
    X_test_path       = sys.argv[3]
    config_path       = sys.argv[4]
    result_path       = sys.argv[5]
    shap_values_path  = sys.argv[6]
    algorithm         = sys.argv[7] if len(sys.argv) > 7 else "v2"

    try:
        values, result = run(X_train_path, y_train_path, X_test_path, config_path, algorithm)
        np.save(shap_values_path, values)
    except Exception as e:
        import traceback
        result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    with open(result_path, "w") as f:
        json.dump(result, f)
