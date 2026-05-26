"""Build the Colab/Kaggle-friendly benchmark notebook.

This variant does NOT use venvs. It relies on the fact that every runner call is
a separate `python -m treeshap_bench.runner` subprocess, so the kernel's
pip-install state at launch time IS the environment. We swap that state between
cells: install fasttreeshap + released shapiq, run them, then upgrade shapiq to
git-main in place and run only shapiq. The merge/plot cells are identical to
analysis.ipynb so the two notebooks stay in sync.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ---------------------------------------------------------------------------
md("""# TreeSHAP Benchmark — Colab / Kaggle edition

Runs the `treeshap_bench` harness on a free cloud VM (Colab or Kaggle) so the
numbers come from a standardized, reproducible environment. Comparison targets:

* **fasttreeshap v1 / v2** (needs `numpy<2`)
* **shapiq 1.4.1** (released — the slow-SV TreeExplainer)
* **shapiq git-main** (the reworked SV path from issue #453)

## How this works without virtualenvs

Two versions of `shapiq` can't be imported in one Python process. But each
benchmark run is a **separate subprocess** (`!python -m treeshap_bench.runner`),
and a subprocess sees whatever is pip-installed *at the moment it launches*. So
we don't need venvs — we just change what's installed between cells:

1. Install fasttreeshap + shapiq 1.4.1 → run them → write `results/envA.json`.
2. `pip install -U git+...shapiq` (replaces 1.4.1 in place) → run shapiq only →
   write `results/envB.json`.
3. Merge both JSONs and plot.

> ⚠️ **The kernel's package state changes partway through this notebook.** That
> is intentional and required (fasttreeshap wants `numpy<2`; git-main shapiq
> wants modern numpy + Python ≥3.12 — they cannot coexist). Run the cells **in
> order, top to bottom.** "Run all" works; re-running an earlier cell after a
> later `pip install` does **not** without re-installing. Each result JSON
> records the exact library versions it was produced under, so the artifacts
> stay self-documenting even though the kernel state shifts.""")

# ---------------------------------------------------------------------------
md("## 0. Environment check (records versions, warns if git target won't work)")

code("""import sys, platform, json, os
print("Python :", platform.python_version())
print("Platform:", platform.platform())
print("CPU cores (os.cpu_count):", os.cpu_count())

PY_OK_FOR_GIT = sys.version_info >= (3, 12)
if PY_OK_FOR_GIT:
    print("\\n✅ Python >= 3.12 — git-main shapiq target is available.")
else:
    print(f"\\n⚠️ Python {platform.python_version()} < 3.12 — git-main shapiq "
          "(v1.5.0) will FAIL to install. The released-shapiq comparison still "
          "works; skip Section 3 (the git target) on this runtime.")
    print("   Colab tip: a default 2026 runtime is 3.12. If you're on an older "
          "pinned runtime, switch to the latest runtime version.")""")

# ---------------------------------------------------------------------------
md("""## 1. Get the harness code and install it

Clone your repo (or upload the `treeshap-bench/` folder via the Files panel),
then **`pip install -e`** it. The editable install is what makes
`python -m treeshap_bench.runner` work from the notebook's subprocess cells:
without it, `-m` resolves the package name against the current working
directory, which is fragile and breaks if the CWD isn't the package's parent.
Installing registers the package on the environment path so it resolves from
anywhere.""")

code("""# Set this to your repo URL, or leave "" and upload the treeshap-bench/ folder
# via the Files panel so it sits next to this notebook.
REPO_URL = ""  # e.g. "https://github.com/<you>/treeshap-bench.git"

import os, subprocess, sys

if REPO_URL:
    if not os.path.isdir("treeshap-bench"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "treeshap-bench"], check=True)
    repo_dir = "treeshap-bench"
else:
    # Find the repo root: it's the directory that CONTAINS the importable
    # `treeshap_bench` package (i.e. has treeshap_bench/__init__.py under it).
    candidates = [".", "treeshap-bench", "treeshap_bench"]
    repo_dir = next(
        (c for c in candidates
         if os.path.isfile(os.path.join(c, "treeshap_bench", "__init__.py"))),
        None,
    )
    assert repo_dir is not None, (
        "Could not find the treeshap_bench package. Upload the treeshap-bench/ "
        "folder via the Files panel (it must contain treeshap_bench/__init__.py), "
        "or set REPO_URL above."
    )

# Editable install. --no-deps here so we control the numpy<2 pin in Section 2;
# the package's own deps (numpy/pandas/sklearn/joblib) come with the Env-A install.
print(f"Installing package from: {os.path.abspath(repo_dir)}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", repo_dir, "--no-deps"], check=True)

# Verify it resolves as a subprocess module (this is exactly what the runner cells do).
chk = subprocess.run([sys.executable, "-m", "treeshap_bench.runner", "--help"],
                     capture_output=True, text=True)
assert chk.returncode == 0, f"Runner not importable:\\n{chk.stderr}"
print("✅ treeshap_bench.runner resolves correctly.")""")

# ---------------------------------------------------------------------------
md("""## 2. Environment A — fasttreeshap v1/v2 + released shapiq

Installs `numpy<2` (required by fasttreeshap) and pinned versions of both
libraries, then runs the benchmark in a subprocess.

Keep `--max-depth` small (≤4): released shapiq's SV path is super-linear in tree
depth, so a deep forest will hang. The depth sweep in Section 4 shows exactly
why.""")

code("""# Install the Env-A stack (the package itself is already installed from Section 1).
# The pin on numpy<2 is the key constraint for fasttreeshap.
!pip install -q 'numpy<2' 'fasttreeshap==0.1.6' 'shapiq==1.4.1' \\
    scikit-learn xgboost lightgbm joblib

# NOTE: On Colab you do NOT need to restart the kernel here, because the
# benchmark runs in a SUBPROCESS (next cell) that starts fresh and picks up
# numpy<2 on its own. The notebook kernel itself never imports numpy<2.""")

code("""import os
os.makedirs("results", exist_ok=True)

# Subprocess picks up the freshly-installed Env-A packages.
!python -m treeshap_bench.runner \\
    --adapters fasttreeshap_v1 fasttreeshap_v2 shapiq \\
    --model random_forest --n-estimators 50 --max-depth 4 \\
    --tasks sv interaction \\
    --sv-samples 200 --interaction-samples 25 --rounds 3 \\
    --out results/envA_released.json

print("\\nEnv A done.")""")

# ---------------------------------------------------------------------------
md("""## 3. Environment B — shapiq git-main  *(skip if Python < 3.12)*

This upgrades shapiq to the development version **in place**. After this cell,
fasttreeshap may stop working (numpy gets dragged forward) — that's fine, its
numbers are already saved in `envA_released.json`. We only run the shapiq target
here.""")

code("""import sys
if sys.version_info >= (3, 12):
    # Replaces shapiq 1.4.1 with git-main. Modern numpy comes along for the ride.
    !pip install -q -U "git+https://github.com/mmschlk/shapiq.git"
    print("Installed git-main shapiq. (fasttreeshap may now be broken — expected.)")
else:
    print("Skipped: Python < 3.12, git-main shapiq is unavailable on this runtime.")""")

code("""import sys
if sys.version_info >= (3, 12):
    # Subprocess now sees git-main shapiq. Note: the cached model from Env A is
    # reused (treeshap_bench.data caches by config), so the model is identical.
    !python -m treeshap_bench.runner \\
        --adapters shapiq \\
        --model random_forest --n-estimators 50 --max-depth 4 \\
        --tasks sv interaction \\
        --sv-samples 200 --interaction-samples 25 --rounds 3 \\
        --out results/envB_git.json
    print("\\nEnv B done.")
else:
    print("Skipped Env B run.")""")

# ---------------------------------------------------------------------------
md("""## 4. (Optional) Depth sweep — the real finding

Runs released shapiq vs fasttreeshap across tree depths and plots per-sample
time. The divergence is the artifact worth attaching to shapiq issue #453. This
uses Env-A packages, so run it **before** Section 3's git upgrade, or
re-install the Env-A stack first.""")

code("""import os, subprocess, sys
os.makedirs("results/depth_sweep", exist_ok=True)
for depth in [3, 4, 5, 6, 8]:
    out = f"results/depth_sweep/rf_depth{depth}.json"
    if os.path.exists(out):
        continue
    subprocess.run([
        sys.executable, "-m", "treeshap_bench.runner",
        "--adapters", "fasttreeshap_v1", "fasttreeshap_v2", "shapiq",
        "--model", "random_forest", "--n-estimators", "20", "--max-depth", str(depth),
        "--tasks", "sv", "--sv-samples", "20", "--rounds", "2",
        "--out", out,
    ], check=True)
print("depth sweep done")""")

# ---------------------------------------------------------------------------
md("""## 5. Merge results and plot

From here down the cells are **identical** to `analysis.ipynb` — they only read
the JSON in `results/`, so they don't care which environment produced each file
or whether you're on Colab, Kaggle, or a laptop.""")

# --- merge cell (verbatim from analysis.ipynb) ---
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

# --- timing plot cell (verbatim) ---
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

# --- depth-sweep plot cell (verbatim from analysis.ipynb section 5) ---
code("""dep_rows = []
for path in sorted(glob.glob("results/depth_sweep/*.json")):
    for r in load_results(path)["results"]:
        if r["error"] is None and r["task"] == "sv":
            dep_rows.append(r)

if dep_rows:
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
    plt.show()
else:
    print("No depth-sweep results yet — run Section 4.")""")

md("""## 6. Save results off the VM

Colab/Kaggle VMs are ephemeral. Download the JSON (or commit it back to your
repo) so the run is preserved and citable.""")

code("""# Colab download:
# from google.colab import files
# import shutil
# shutil.make_archive("treeshap_results", "zip", "results")
# files.download("treeshap_results.zip")

# Kaggle: results/ is already under /kaggle/working and is saved with the notebook output.
print("results/ contents:")
for p in sorted(glob.glob("results/**/*.json", recursive=True)):
    print(" ", p)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_colab.ipynb"), "w") as f:
    nbf.write(nb, f)
print("Colab notebook written with", len(cells), "cells")
