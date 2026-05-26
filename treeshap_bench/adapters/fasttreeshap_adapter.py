"""Adapter for LinkedIn's FastTreeSHAP (algorithms v1 and v2).

fasttreeshap is pinned at 0.1.6 (Jun 2023) and predates NumPy 2.0, so the host
environment must use ``numpy<2``. v2 does not implement interaction values, so
that variant only advertises the "sv" task.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import ShapAdapter


class _FastTreeSHAPBase(ShapAdapter):
    library = "fasttreeshap"
    algorithm = "v1"  # overridden by subclasses

    def _build_explainer(self, model: Any, task: str) -> Any:
        import fasttreeshap

        # shortcut=False forces fasttreeshap's own code path rather than delegating
        # to the model library's built-in TreeSHAP (which would not be a fair
        # comparison of fasttreeshap itself).
        return fasttreeshap.TreeExplainer(
            model,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
            shortcut=False,
        )

    def _explain(self, X: np.ndarray) -> np.ndarray:
        interactions = self._task == "interaction"
        # fasttreeshap accepts a DataFrame or ndarray; we pass ndarray.
        out = self._explainer(X, interactions=interactions).values
        if self._task == "sv":
            # Binary classifiers (sklearn RF, LightGBM) return (..., n_classes);
            # slice the positive class to match shapiq's class_index. XGBoost
            # returns a single output, so leave it alone.
            if out.ndim == 3 and self.class_index is not None:
                out = out[..., self.class_index]
        return out


class FastTreeSHAPv1(_FastTreeSHAPBase):
    name = "fasttreeshap_v1"
    algorithm = "v1"
    supported_tasks = ("sv", "interaction")


class FastTreeSHAPv2(_FastTreeSHAPBase):
    name = "fasttreeshap_v2"
    algorithm = "v2"
    supported_tasks = ("sv",)  # v2 has no interaction implementation
