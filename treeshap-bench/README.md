# treeshap-bench

An extensible benchmark harness comparing tree-SHAP libraries (FastTreeSHAP v1/v2,
shapiq's TreeSHAP-IQ, and pluggable stubs for GPUTreeShap / woodelf).

## Why a package and not a notebook

Timings must be reproducible and version-pinned, and **two versions of the same
package (e.g. released `shapiq==1.4.1` vs git-main `shapiq`) cannot coexist in one
Python interpreter.** So the benchmark logic lives in importable modules, each
target runs in its own environment via a CLI, and results are written as
self-describing JSON that the notebook merges and plots.

## Layout

| Path | Purpose |
|---|---|
| `treeshap_bench/core.py` | `ShapAdapter` interface, timing harness (with warm-up), `RunResult`, environment capture, JSON (de)serialization |
| `treeshap_bench/data.py` | Adult dataset + cached model training, shared across all targets for fairness |
| `treeshap_bench/runner.py` | CLI: runs adapters installed in the *current* env, writes one JSON |
| `treeshap_bench/adapters/` | One module per library; register in `__init__.py` |
| `run_all.sh` | Orchestrates the multi-environment sweep (incl. git-main shapiq in its own venv) |
| `analysis.ipynb` | Thin notebook: kick off runs, load JSON, plot |
| `benchmark_colab.ipynb` | Colab/Kaggle variant: same harness, swaps pip state between cells instead of using venvs |

## Quick start (base environment)

```bash
pip install -e .          # registers the treeshap_bench package (REQUIRED for `python -m`)
pip install 'numpy<2' 'fasttreeshap==0.1.6' 'shapiq==1.4.1' xgboost lightgbm matplotlib
# restart kernel if you downgraded numpy

python -m treeshap_bench.runner \
  --adapters fasttreeshap_v1 fasttreeshap_v2 shapiq \
  --model random_forest --n-estimators 50 --max-depth 4 \
  --tasks sv interaction --sv-samples 200 --interaction-samples 25 --rounds 3 \
  --out results/envA_released.json
```

> **Repo name vs package name.** The repository folder is `treeshap-bench`
> (hyphen — this is the distribution name in `pyproject.toml`). The importable
> Python package nested inside it is `treeshap_bench` (underscore). You always
> `import treeshap_bench` / `python -m treeshap_bench.runner`; the hyphenated
> name only appears in `pip install`. `python -m` resolves the package against
> the installed environment, **not** your current directory — which is why the
> `pip install -e .` step above is required, not optional.

Keep `--max-depth` small (<=4) for released shapiq: its SV path is super-linear
in depth (see issue #453 and the depth-sweep cell in the notebook).

## Comparing released vs git-main shapiq

```bash
bash run_all.sh   # Env A: fasttreeshap + shapiq 1.4.1; Env B: git-main shapiq (py>=3.12 venv)
```

Both write into `results/`; the notebook merges by captured `library_version`.

## Running on Colab / Kaggle (no venvs needed)

Use `benchmark_colab.ipynb`. It avoids virtualenvs entirely by exploiting the
fact that each runner call is a **subprocess**: a subprocess sees whatever is
pip-installed when it launches, so you swap the kernel's install state between
cells instead of building separate environments.

1. Install `numpy<2` + fasttreeshap + `shapiq==1.4.1`, run them (subprocess
   picks up numpy<2 on its own — no kernel restart needed).
2. `pip install -U git+...shapiq` to replace 1.4.1 in place, run shapiq only.
3. Merge + plot (identical cells to `analysis.ipynb`).

Requirements / caveats:

* git-main shapiq (→ v1.5.0) needs **Python ≥ 3.12**. Colab's 2026 default
  runtime is 3.12, so this works today; the notebook's Section 0 checks and
  warns if not.
* The kernel's package state changes partway through — run cells **in order**.
  Each result JSON records the versions it was produced under, so the artifacts
  remain self-documenting despite the mid-notebook numpy upgrade.
* Pin the Colab runtime version (Runtime → "Use a past runtime version") if you
  want byte-stable library versions across re-runs for a wide audience.

## Adding a library

Subclass `ShapAdapter`, implement `_build_explainer` + `_explain`, register in
`adapters/__init__.py`. See `adapters/stubs.py` for GPUTreeShap and woodelf
starting points (both currently raise `NotImplementedError` with TODOs).
