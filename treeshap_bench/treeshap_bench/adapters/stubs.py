"""Stub adapters for libraries not yet wired up.

These exist to demonstrate that the interface is genuinely extensible and to
give you a fill-in-the-blanks starting point. Each raises NotImplementedError
with a pointer to what needs verifying, rather than guessing at an API.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import ShapAdapter


class GPUTreeShapAdapter(ShapAdapter):
    """NVIDIA GPUTreeShap (https://github.com/rapidsai/gputreeshap).

    NOTE: GPUTreeShap is a CUDA C++ library; in Python it is normally accessed
    *through* XGBoost's / LightGBM's GPU predictor (``pred_interactions`` on a
    GPU-resident model) rather than as a standalone pip package. It will not run
    on a CPU-only Colab instance. Wire this up only on a CUDA box and decide
    whether you're benchmarking the standalone library or the XGBoost GPU path —
    they are different things and you should document which.
    """

    name = "gputreeshap"
    library = "gputreeshap"
    supported_tasks = ("sv", "interaction")

    def _build_explainer(self, model: Any, task: str) -> Any:
        raise NotImplementedError(
            "GPUTreeShap requires a CUDA device and is typically invoked via "
            "the model library's GPU predictor. Implement against your actual "
            "GPU setup and remove this guard."
        )

    def _explain(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class WoodelfAdapter(ShapAdapter):
    """woodelf — UNVERIFIED.

    I was unable to confirm woodelf's distribution name or public API from
    available sources. Before using this adapter, verify:
      * the pip/distribution name (set ``library`` accordingly so the version
        is captured correctly),
      * the import name and explainer construction call (in ``_build_explainer``),
      * the explain call and the shape/layout of what it returns (in ``_explain``).
    Point me at the repo and I'll fill this in precisely.
    """

    name = "woodelf"
    library = "woodelf"  # <-- verify: distribution name may differ from import name
    supported_tasks = ()  # set once the real capabilities are known

    def _build_explainer(self, model: Any, task: str) -> Any:
        raise NotImplementedError(
            "woodelf adapter is an unverified stub. Confirm the library's "
            "import name and API, then implement _build_explainer / _explain."
        )

    def _explain(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
