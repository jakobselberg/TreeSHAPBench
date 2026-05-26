"""Core benchmark primitives: the adapter interface, the timing harness, and
self-describing result records.

The design separates three concerns deliberately:

1. **Construction / setup** (``adapter.setup(model)``) — one-time work such as
   converting a fitted model into the library's internal tree representation.
   This is *not* timed as part of the explain benchmark, but its duration is
   recorded separately because it can dominate for some libraries.

2. **Explain** (``adapter.explain(X)``) — the operation we actually benchmark.

3. **Reporting** — every run captures the full environment (library versions,
   CPU, core count, OS, Python) so a result JSON is reproducible and portable
   enough to attach to a GitHub issue.

Numba/Cython libraries compile on first call, so the harness always runs one
discarded **warm-up** before the timed rounds.
"""
from __future__ import annotations

import abc
import dataclasses
import importlib.metadata
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """A single (adapter, task, model-config) measurement."""

    adapter: str                      # e.g. "fasttreeshap_v2"
    library: str                      # e.g. "fasttreeshap"
    library_version: str              # resolved at runtime
    task: str                         # "sv" or "interaction"
    model_kind: str                   # "random_forest" | "xgboost" | "lightgbm"
    n_estimators: int
    max_depth: int
    n_features: int
    n_samples_explained: int
    n_jobs: int

    setup_seconds: float              # one-time model conversion, not part of timed rounds
    round_seconds: list[float] = field(default_factory=list)  # one entry per timed round

    # Optional cross-library correctness anchor: max|phi - reference_phi|.
    # Populated by the runner when a reference is available.
    max_abs_diff_vs_reference: Optional[float] = None
    reference_adapter: Optional[str] = None

    # Free-form notes (e.g. "explained per-sample due to API shape").
    notes: str = ""
    error: Optional[str] = None       # set if the run failed instead of producing timings

    @property
    def mean_seconds(self) -> Optional[float]:
        return statistics.mean(self.round_seconds) if self.round_seconds else None

    @property
    def std_seconds(self) -> Optional[float]:
        return statistics.pstdev(self.round_seconds) if len(self.round_seconds) > 1 else 0.0

    @property
    def per_sample_seconds(self) -> Optional[float]:
        m = self.mean_seconds
        if m is None or self.n_samples_explained == 0:
            return None
        return m / self.n_samples_explained

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["mean_seconds"] = self.mean_seconds
        d["std_seconds"] = self.std_seconds
        d["per_sample_seconds"] = self.per_sample_seconds
        return d


@dataclass
class Environment:
    """Captured once per benchmark process so results are self-describing."""

    python_version: str
    platform: str
    processor: str
    cpu_count: Optional[int]
    library_versions: dict[str, str]

    @staticmethod
    def capture(libraries: list[str]) -> "Environment":
        import os

        versions: dict[str, str] = {}
        for lib in libraries:
            try:
                versions[lib] = importlib.metadata.version(lib)
            except importlib.metadata.PackageNotFoundError:
                versions[lib] = "not-installed"
        return Environment(
            python_version=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor() or "unknown",
            cpu_count=os.cpu_count(),
            library_versions=versions,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# The adapter interface
# ---------------------------------------------------------------------------
class ShapAdapter(abc.ABC):
    """Common interface every benchmark target implements.

    A new library is added by subclassing this and registering it in
    ``adapters/__init__.py``. Two versions of the *same* package cannot be
    compared in one process (Python binds one version of a module name per
    interpreter) — run them in separate environments and aggregate the JSON.
    """

    #: Short stable identifier, e.g. "shapiq_sv". Used as a dict key and filename.
    name: str = "base"

    #: The pip/distribution name used to resolve the installed version.
    library: str = "base"

    #: Tasks this adapter supports. Subset of {"sv", "interaction"}.
    supported_tasks: tuple[str, ...] = ()

    def __init__(self, *, n_jobs: int = -1, class_index: Optional[int] = 1):
        self.n_jobs = n_jobs
        self.class_index = class_index
        self._explainer: Any = None
        self._task: Optional[str] = None
        self._setup_seconds: float = 0.0

    # -- lifecycle -----------------------------------------------------------
    @abc.abstractmethod
    def _build_explainer(self, model: Any, task: str) -> Any:
        """Construct and return the library's explainer object for ``task``.

        Implementations should do the expensive one-time model→internal-tree
        conversion here. The base class times this call.
        """

    @abc.abstractmethod
    def _explain(self, X: np.ndarray) -> np.ndarray:
        """Run the actual explanation on ``X`` and return an array of shape
        ``(n_samples, n_features)`` for the SV task, or whatever the library
        natively returns for interactions. The base class times this call.
        """

    def setup(self, model: Any, task: str) -> None:
        if task not in self.supported_tasks:
            raise ValueError(f"{self.name} does not support task '{task}'")
        self._task = task
        t0 = time.perf_counter()
        self._explainer = self._build_explainer(model, task)
        self._setup_seconds = time.perf_counter() - t0

    def explain_once(self, X: np.ndarray) -> np.ndarray:
        if self._explainer is None:
            raise RuntimeError("Call setup() before explain_once().")
        return self._explain(X)

    # -- version resolution --------------------------------------------------
    def resolve_version(self) -> str:
        try:
            return importlib.metadata.version(self.library)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"


# ---------------------------------------------------------------------------
# The timing harness
# ---------------------------------------------------------------------------
def benchmark(
    adapter: ShapAdapter,
    model: Any,
    X: np.ndarray,
    *,
    task: str,
    model_kind: str,
    n_estimators: int,
    max_depth: int,
    num_rounds: int = 3,
    warmup: bool = True,
) -> RunResult:
    """Run one adapter on one (model, X, task) and return a RunResult.

    Catches and records exceptions instead of raising, so one failing target
    (e.g. a GPU library on a CPU box) does not abort the whole sweep.
    """
    n_samples, n_features = X.shape
    result = RunResult(
        adapter=adapter.name,
        library=adapter.library,
        library_version=adapter.resolve_version(),
        task=task,
        model_kind=model_kind,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_features=n_features,
        n_samples_explained=n_samples,
        n_jobs=adapter.n_jobs,
        setup_seconds=0.0,
    )

    try:
        adapter.setup(model, task)
        result.setup_seconds = adapter._setup_seconds

        if warmup:
            # First call triggers numba/Cython compilation; explain a tiny slice
            # and discard the timing.
            adapter.explain_once(X[: min(2, n_samples)])

        for _ in range(num_rounds):
            t0 = time.perf_counter()
            adapter.explain_once(X)
            result.round_seconds.append(time.perf_counter() - t0)

    except Exception as exc:  # noqa: BLE001 - we intentionally record any failure
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def save_results(
    path: str,
    *,
    environment: Environment,
    results: list[RunResult],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "environment": environment.to_dict(),
        "results": [r.to_dict() for r in results],
        "extra": extra or {},
        "schema_version": 1,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_results(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
