"""Build the thin analysis/orchestration notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# TreeSHAP Library Benchmark — fasttreeshap vs shapiq (and beyond)

This notebook is deliberately **thin**. All the actual benchmarking lives in the
`treeshap_bench` package; the notebook only (1) kicks off runs and (2) loads the
resulting JSON and plots it. That separation keeps timings reproducible and
keeps the heavy, version-sensitive code out of notebook cells.

## Architecture

```
treeshap_bench/
  core.py                  # ShapAdapter ABC, timing harness, RunResult, env capture
  data.py                  # dataset + model training (cached, shared across targets)
  runner.py                # CLI: runs adapters in ONE env -> writes a result JSON
  adapters/
    fasttreeshap_adapter.py  # v1, v2
    shapiq_adapter.py        # works for released AND git-main shapiq
    stubs.py                 # gputreeshap, woodelf (fill-in-the-blanks)
run_all.sh                 # orchestrates the multi-ENVIRONMENT sweep
```

## The one rule that shapes everything

**Two versions of the same package cannot live in one Python process.** Released
`shapiq==1.4.1` and git-main `shapiq` are the same import name at different
versions, so they *cannot* be compared inside a single kernel. The harness
handles this by running each target in its own environment via `run_all.sh` and
merging the JSON afterwards. Different *packages* (fasttreeshap vs shapiq) are
fine together — those just need adapters.""")

md("## 1. Install the package (editable) and a benchmark environment")

code("""# Install the harness itself plus the "base" env targets.
# fasttreeshap needs numpy<2; shapiq 1.4.1 is the released slow-SV version.
# !pip install -e .
# !pip install 'numpy<2' 'fasttreeshap==0.1.6' 'shapiq==1.4.1' xgboost lightgbm matplotlib
# (Restart the kernel after installing numpy<2.)""")

md("""## 2. Run the benchmark

You can call the runner directly from the notebook for the base environment. The
git-main shapiq target needs its own venv (Python >=3.12) — use `run_all.sh` for
that, or run the cell below in a 3.12+ kernel that has git-main installed.

Start shallow (`--max-depth 4`) so released shapiq actually finishes; the
depth-scaling cell later shows why.""")

code("""# Base environment: fasttreeshap v1/v2 + released shapiq, on a shallow forest.
!python -m treeshap_bench.runner \\
    --adapters fasttreeshap_v1 fasttreeshap_v2 shapiq \\
    --model random_forest --n-estimators 50 --max-depth 4 \\
    --tasks sv interaction \\
    --sv-samples 200 --interaction-samples 25 --rounds 3 \\
    --out results/envA_released.json""")

code("""# Full multi-environment sweep (spawns an isolated venv for git-main shapiq):
# !bash run_all.sh""")

md("## 3. Load and merge all result JSONs")

code("""import glob, json
import pandas as pd
from treeshap_bench.core import load_results

rows = []
envs = {}
for path in sorted(glob.glob("results/*.json")):
    payload = load_results(path)
    src = path.split("/")[-1]
    envs[src] = payload["environment"]
    for r in payload["results"]:
        r = dict(r)
        r["source_file"] = src
        # disambiguate same-named adapter across versions (e.g. shapiq 1.4.1 vs git)
        r["target"] = f"{r['library']} {r['library_version']} [{r['adapter']}]"
        rows.append(r)

df = pd.DataFrame(rows)
# show the environments we pulled from
for src, e in envs.items():
    print(src, "->", e["library_versions"], "| py", e["python_version"], "|", e["cpu_count"], "cores")
df[["target", "task", "model_kind", "max_depth", "n_samples_explained",
    "mean_seconds", "std_seconds", "per_sample_seconds", "setup_seconds", "error"]]""")

md("## 4. Timing comparison (per-sample, log scale)")

code("""import matplotlib.pyplot as plt
import numpy as np

ok = df[df["error"].isna()].copy()

for task in ["sv", "interaction"]:
    sub = ok[ok["task"] == task]
    if sub.empty:
        continue
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sub) + 1.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["per_sample_seconds"] * 1000)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["target"])
    ax.set_xlabel("ms per sample")
    ax.set_xscale("log")
    ax.set_title(f"Per-sample explain time — task={task} "
                 f"(model={sub['model_kind'].iloc[0]}, depth={sub['max_depth'].iloc[0]})")
    ax.invert_yaxis()
    for yi, v in zip(y, sub["per_sample_seconds"] * 1000):
        ax.text(v, yi, f" {v:.1f}", va="center")
    plt.tight_layout()
    plt.show()""")

md("""## 5. The real story: depth scaling

The single-number table hides *why* released shapiq struggles. This sweep runs
each target across tree depths and plots per-sample time — the divergence is the
finding worth sending to the maintainers (it corroborates their own issue #453,
where the SV path is acknowledged as slow because TreeSHAP-IQ treats Shapley
values as a special case of interactions).""")

code("""import subprocess, os
os.makedirs("results/depth_sweep", exist_ok=True)
for depth in [3, 4, 5, 6, 8]:
    out = f"results/depth_sweep/rf_depth{depth}.json"
    if os.path.exists(out):
        continue
    subprocess.run([
        "python", "-m", "treeshap_bench.runner",
        "--adapters", "fasttreeshap_v1", "fasttreeshap_v2", "shapiq",
        "--model", "random_forest", "--n-estimators", "20", "--max-depth", str(depth),
        "--tasks", "sv", "--sv-samples", "20", "--rounds", "2",
        "--out", out,
    ], check=True)
print("depth sweep done")""")

code("""dep_rows = []
for path in sorted(glob.glob("results/depth_sweep/*.json")):
    for r in load_results(path)["results"]:
        if r["error"] is None and r["task"] == "sv":
            dep_rows.append(r)
dep = pd.DataFrame(dep_rows)

fig, ax = plt.subplots(figsize=(8, 5))
for target, g in dep.groupby("adapter"):
    g = g.sort_values("max_depth")
    ax.plot(g["max_depth"], g["per_sample_seconds"] * 1000, marker="o", label=target)
ax.set_xlabel("max_depth")
ax.set_ylabel("ms per sample")
ax.set_yscale("log")
ax.set_title("SV explain cost vs tree depth (RF, 20 trees)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()""")

md("""## 6. Adding a new library

1. Subclass `ShapAdapter` in `adapters/`, implementing `_build_explainer` and
   `_explain` (see `stubs.py` for `gputreeshap` / `woodelf` starting points).
2. Register it in `adapters/__init__.py`.
3. Pass its name to `--adapters`. If it needs a conflicting dependency, give it
   its own environment and merge the JSON like the git-shapiq target.

The harness already handles warm-up (numba/Cython first-call compilation is
discarded), separates one-time model conversion (`setup_seconds`) from the timed
rounds, captures the full environment, and records failures instead of aborting
the sweep — so a GPU-only library on a CPU box just shows up as an error row.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
with open("/home/claude/treeshap_bench/analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("notebook written with", len(cells), "cells")
