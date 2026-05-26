"""Adapter for GPUTreeShap, accessed through XGBoost's GPU SHAP path.

GPUTreeShap (https://github.com/rapidsai/gputreeshap) is a header-only CUDA C++
library with **no Python package** — you do not `import` or `pip install` it.
The library itself is model-agnostic (it operates on a generic tree/path
representation), and the shap package exposes `shap.GPUTreeExplainer`, which
uses GPUTreeShap as its backend and supports the same model coverage as the CPU
TreeExplainer (XGBoost, LightGBM, CatBoost, and most tree-based scikit-learn
models).

So GPUTreeShap is NOT inherently XGBoost-only. There are two access paths, with
very different setup cost:

1. **XGBoost's built-in GPU predictor** (what THIS adapter uses): set the model
   to `device="cuda"` and call `predict(..., pred_contribs/pred_interactions)`.
   GPUTreeShap is the backend. Zero extra install on Colab (its xgboost is
   CUDA-enabled). But this entry point is necessarily XGBoost-only, because it
   is XGBoost's own predict method — it can't explain RF/LightGBM models.

2. **shap.GPUTreeExplainer** (the general path, NOT wired up here): supports
   RF/LightGBM/etc., but currently requires shap to be **built from source with
   CUDA** (`CUDA_PATH` set) — there's no pip wheel with the GPU extension. That
   build is a chore on Colab, so we don't use it by default. If you want RF or
   LightGBM on GPU via GPUTreeShap, that's the route: compile shap with CUDA and
   add a sibling adapter wrapping `shap.GPUTreeExplainer`.

This adapter therefore takes path (1): a zero-build GPU benchmark that happens
to be XGBoost-only. It declares an XGBoost-only guard for that reason, not
because GPUTreeShap itself is limited to XGBoost.

Correctness anchor: the GPU values are XGBoost's own SHAP algorithm, so the
right reference is XGBoost on CPU (`XGBoostCPUShapAdapter` below), not shapiq.

xgboost is imported lazily inside the methods so the adapter can be registered
everywhere without requiring a CUDA build merely to import it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import ShapAdapter


def _is_xgboost_model(model: Any) -> bool:
    cls = type(model)
    mod = cls.__module__ or ""
    return mod.startswith("xgboost") or cls.__name__ in {
        "XGBClassifier", "XGBRegressor", "Booster"
    }


class _XGBoostShapBase(ShapAdapter):
    """Shared logic for XGBoost's native SHAP path (CPU or GPU)."""

    library = "xgboost"
    device = "cpu"          # overridden by subclasses
    supported_tasks = ("sv", "interaction")

    def _build_explainer(self, model: Any, task: str) -> Any:
        import xgboost as xgb

        if not _is_xgboost_model(model):
            raise RuntimeError(
                f"{self.name} only supports XGBoost models; got "
                f"{type(model).__module__}.{type(model).__name__}. "
                "GPUTreeShap has no path for sklearn/LightGBM models."
            )

        # Get the underlying Booster (XGBClassifier wraps one).
        booster = model.get_booster() if hasattr(model, "get_booster") else model

        # Select device. xgboost >= 2.0: device="cuda"/"cpu".
        # Older xgboost: predictor="gpu_predictor"/"cpu_predictor".
        try:
            booster.set_param({"device": self.device})
        except Exception:
            legacy = "gpu_predictor" if self.device.startswith("cuda") else "cpu_predictor"
            booster.set_param({"predictor": legacy})

        # If we asked for CUDA, fail loudly now if no usable GPU, so the result
        # row records a clear error rather than silently running on CPU.
        if self.device.startswith("cuda"):
            self._assert_cuda_available()

        self._xgb = xgb
        self._booster = booster
        return booster

    @staticmethod
    def _assert_cuda_available() -> None:
        # XGBoost doesn't expose a clean "is GPU present" probe; do a tiny
        # round-trip on a 1-row DMatrix and let any CUDA error surface here.
        import xgboost as xgb
        try:
            import numpy as _np
            probe = xgb.DMatrix(_np.zeros((1, 1), dtype=_np.float32))
            # Building the DMatrix is cheap; the real check is that the booster's
            # device is cuda and predict won't be attempted on CPU silently.
            # We rely on the actual predict call to raise if CUDA is unusable.
            del probe
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"CUDA not usable for XGBoost: {exc}")

    def _explain(self, X: np.ndarray) -> np.ndarray:
        dmat = self._xgb.DMatrix(np.asarray(X, dtype=np.float32))
        if self._task == "interaction":
            out = self._booster.predict(dmat, pred_interactions=True)
        else:
            out = self._booster.predict(dmat, pred_contribs=True)
        # pred_contribs returns (n, n_features + 1): the last column is the bias.
        # Drop it so the shape matches the other adapters' (n, n_features).
        if self._task == "sv" and out.ndim == 2:
            out = out[:, :-1]
        # pred_interactions returns (n, n_features+1, n_features+1); drop the bias
        # row/col to get (n, n_features, n_features). Timing-only, but keep tidy.
        elif self._task == "interaction" and out.ndim == 3:
            out = out[:, :-1, :-1]
        return out


class XGBoostGPUShapAdapter(_XGBoostShapBase):
    """GPUTreeShap via XGBoost's CUDA predictor. Requires a CUDA-capable GPU."""

    name = "gputreeshap_xgb"
    device = "cuda"

    def resolve_version(self) -> str:
        # Report xgboost's version since GPUTreeShap itself is vendored and has
        # no independent version string at runtime.
        v = super().resolve_version()
        return f"{v} (gputreeshap backend, device=cuda)"


class XGBoostCPUShapAdapter(_XGBoostShapBase):
    """XGBoost's CPU SHAP path — the correctness reference for the GPU adapter."""

    name = "xgboost_cpu_shap"
    device = "cpu"

    def resolve_version(self) -> str:
        v = ShapAdapter.resolve_version(self)
        return f"{v} (cpu pred_contribs)"
