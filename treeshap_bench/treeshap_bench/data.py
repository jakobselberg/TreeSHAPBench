"""Dataset loading and model training.

Kept separate from the adapters so that every benchmark target is guaranteed to
run on the *identical* fitted model — fairness depends on this. Models are
trained once and cached to disk (joblib) keyed by their config, so repeated
runs (and separate per-environment subprocess runs) reuse the same model.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    test_size: float = 0.25
    random_state: int = 0
    # one_hot=False uses ordinal encoding -> far fewer features, which matters a
    # lot for shapiq's per-node cost. Default True to mirror the original notebook.
    one_hot: bool = True


@dataclass(frozen=True)
class ModelConfig:
    kind: str          # "random_forest" | "xgboost" | "lightgbm"
    n_estimators: int
    max_depth: int
    random_state: int = 0

    def cache_key(self, data_cfg: DataConfig) -> str:
        raw = f"{self.kind}-{self.n_estimators}-{self.max_depth}-{self.random_state}-" \
              f"{data_cfg.one_hot}-{data_cfg.test_size}-{data_cfg.random_state}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_adult(cfg: DataConfig):
    """Return (X_train, X_test, y_train, y_test) as DataFrames/arrays."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder

    adult = fetch_openml("adult", version=2, as_frame=True)
    X_raw = adult.data.copy()
    y = (adult.target == ">50K").astype(int).values

    categorical = X_raw.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in categorical:
        X_raw[col] = X_raw[col].astype(str).replace("?", np.nan)
    mask = X_raw.notna().all(axis=1)
    X_raw, y = X_raw[mask].reset_index(drop=True), y[mask.values]

    if cfg.one_hot:
        X = pd.get_dummies(X_raw, columns=categorical, drop_first=False)
        X = X.astype({c: np.int8 for c in X.columns if X[c].dtype == bool})
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X = X_raw.copy()
        X[categorical] = enc.fit_transform(X_raw[categorical])
        X = X.astype(float)

    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )


def _build_model(cfg: ModelConfig):
    if cfg.kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            n_jobs=-1, random_state=cfg.random_state,
        )
    if cfg.kind == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            learning_rate=0.1, n_jobs=-1, eval_metric="logloss",
            random_state=cfg.random_state,
        )
    if cfg.kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            learning_rate=0.1, n_jobs=-1, random_state=cfg.random_state,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model kind: {cfg.kind}")


def get_model(model_cfg: ModelConfig, data_cfg: DataConfig, cache_dir: str = "model_cache"):
    """Train (or load from cache) the model and return (model, X_test_df)."""
    import joblib

    os.makedirs(cache_dir, exist_ok=True)
    key = model_cfg.cache_key(data_cfg)
    model_path = os.path.join(cache_dir, f"model-{key}.joblib")
    xtest_path = os.path.join(cache_dir, f"xtest-{key}.joblib")

    if os.path.exists(model_path) and os.path.exists(xtest_path):
        return joblib.load(model_path), joblib.load(xtest_path)

    X_train, X_test, y_train, _ = load_adult(data_cfg)
    model = _build_model(model_cfg)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(X_test, xtest_path)
    return model, X_test
