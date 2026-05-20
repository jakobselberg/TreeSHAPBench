# TreeSHAPBench

A benchmarking framework for comparing TreeSHAP implementations across speed and correctness.

## Implementations benchmarked

| Algorithm | Library | Notes |
|---|---|---|
| `shap` | [shap](https://github.com/shap/shap) | Standard TreeSHAP baseline |
| `woodelf` | [woodelf](https://github.com/ron-wettenstein/woodelf) | O(n+m) interventional SHAP |
| `fastTreeShap-v1` | [fastTreeShap](https://github.com/linkedin/FastTreeSHAP) | Fast TreeSHAP v1 algorithm |
| `fastTreeShap-v2` | [fastTreeShap](https://github.com/linkedin/FastTreeSHAP) | Fast TreeSHAP v2 algorithm |

Planned: shapiq, GPUTreeSHAP.

## Datasets

Stored in `data/`:

| Dataset | Task | Samples | Features | Source |
|---|---|---|---|---|
| Census Income (Adult) | Classification | ~32k | 13 | [UCI](https://archive.ics.uci.edu/dataset/20/census+income) |
| Superconductivity | Regression | ~21k | 81 | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |

## Setup

`fastTreeShap` requires Python 3.10 and XGBoost 1.x, which are incompatible with the main environment (Python 3.14). Two virtual environments are needed.

**Main environment** (Python 3.14):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**fastTreeShap environment** (Python 3.10):
```bash
/opt/homebrew/bin/python3.10 -m venv .venv-fasttreeshap
.venv-fasttreeshap/bin/pip install "numpy==1.24.*" "xgboost==1.7.6" fasttreeshap shap scikit-learn lightgbm
```

## Usage

```bash
source .venv/bin/activate
python benchmark_example.py
```

The `fastTreeShap-v1/v2` benchmarks automatically spawn a subprocess in `.venv-fasttreeshap` — no manual switching needed.

## Adding a new library

1. Add a `_benchmark_<name>(self, X_sample, results)` method to `src/benchmarker.py`
2. Register it in the `_registry` property — one line
3. Add a call and summary entry in `benchmark_example.py`

## Project structure

```
src/
  benchmarker.py          # TreeBenchmarker class + implementation registry
  data_loader.py          # Dataset loading and preprocessing
  fasttreeshap_runner.py  # Subprocess entry point for fastTreeShap (runs in .venv-fasttreeshap)
benchmark_example.py      # End-to-end benchmark script
data/                     
requirements.txt          
```
