"""Adapter for shapiq's TreeExplainer (the TreeSHAP-IQ algorithm).

This adapter is version-agnostic: the *same* code works against released
shapiq (1.4.1, the slow SV path) and against git-main (which, per issue #453,
routes SV/BV through LinearTreeSHAP and should be dramatically faster). To
compare the two, install each into its own environment and run the benchmark
once per environment — they cannot coexist in one interpreter.

The captured ``library_version`` distinguishes them in the result JSON. For a
git checkout the version string is typically something like ``1.5.0.devN+gHASH``,
which is exactly what you want to cite back to the maintainers.

Task mapping:
  * "sv"          -> max_order=1, index="SV"
  * "interaction" -> max_order=2, index="k-SII" (shapiq's recommended default;
                     pass index="SII" for the closest numerical analog of
                     TreeSHAP's interaction matrix — compute cost is ~identical
                     at order 2).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import ShapAdapter


class ShapiqAdapter(ShapAdapter):
    name = "shapiq"
    library = "shapiq"
    supported_tasks = ("sv", "interaction")

    #: index used for the interaction task; override per-instance if desired.
    interaction_index = "k-SII"

    def _build_explainer(self, model: Any, task: str) -> Any:
        import shapiq

        if task == "sv":
            max_order, index = 1, "SV"
        else:
            max_order, index = 2, self.interaction_index

        return shapiq.TreeExplainer(
            model=model,
            max_order=max_order,
            min_order=1,
            index=index,
            class_index=self.class_index,
        )

    def _explain(self, X: np.ndarray) -> np.ndarray:
        # explain_X accepts a 2D array and parallelises across samples via joblib.
        out = self._explainer.explain_X(np.asarray(X), n_jobs=self.n_jobs)
        if self._task == "sv":
            return self._sv_to_array(out, X.shape[1])
        # For interactions we return the raw list of InteractionValues; the
        # benchmark only times this, and value-level comparison with
        # fasttreeshap's matrix layout is handled separately (different layouts).
        return out

    @staticmethod
    def _sv_to_array(result: Any, n_features: int) -> np.ndarray:
        """Coerce shapiq's order-1 output into (n_samples, n_features).

        explain_X may return a list of InteractionValues (one per sample) or a
        batched object depending on version; handle both defensively.
        """
        if isinstance(result, list):
            rows = []
            for iv in result:
                # get_n_order_values(1) returns the order-1 (Shapley) vector.
                rows.append(np.asarray(iv.get_n_order_values(1)).ravel())
            return np.vstack(rows)
        return np.asarray(getattr(result, "values", result))


class ShapiqAdapterSII(ShapiqAdapter):
    """Variant using index='SII' for interactions (closest to TreeSHAP's matrix)."""

    name = "shapiq_sii"
    interaction_index = "SII"
