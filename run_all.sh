#!/usr/bin/env bash
# Orchestrates the full sweep, including the released-vs-git shapiq comparison
# that REQUIRES separate environments (two versions of one package cannot share
# an interpreter).
#
# Strategy:
#   * Env A ("main"): numpy<2 + fasttreeshap + released shapiq==1.4.1 + models.
#                     Runs fasttreeshap v1/v2 AND shapiq 1.4.1.
#   * Env B ("git"):  git-main shapiq (needs Python >=3.12 per CHANGELOG v1.5.0).
#                     Runs ONLY shapiq from git. fasttreeshap is intentionally
#                     absent here to avoid the numpy<2 vs modern-numpy conflict.
#
# Results from both land in results/ and are merged by the notebook.
#
# Adjust MODEL/DEPTH knobs at the top. depth<=4 keeps released shapiq tractable.

set -euo pipefail

MODEL=${MODEL:-random_forest}
N_EST=${N_EST:-50}
DEPTH=${DEPTH:-4}
SV_SAMPLES=${SV_SAMPLES:-200}
INT_SAMPLES=${INT_SAMPLES:-25}
ROUNDS=${ROUNDS:-3}
RESULTS=${RESULTS:-results}

mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Env A: fasttreeshap + released shapiq (this is your current/base env)
# ---------------------------------------------------------------------------
echo "=== Env A: fasttreeshap v1/v2 + shapiq (released) ==="
python -m treeshap_bench.runner \
    --adapters fasttreeshap_v1 fasttreeshap_v2 shapiq \
    --model "$MODEL" --n-estimators "$N_EST" --max-depth "$DEPTH" \
    --tasks sv interaction \
    --sv-samples "$SV_SAMPLES" --interaction-samples "$INT_SAMPLES" --rounds "$ROUNDS" \
    --out "$RESULTS/envA_released.json"

# ---------------------------------------------------------------------------
# Env B: git-main shapiq, isolated venv. Skip gracefully if Python < 3.12.
# ---------------------------------------------------------------------------
echo "=== Env B: shapiq (git-main), isolated venv ==="
PYVER=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)'; then
    GITENV=$(mktemp -d)/shapiq-git
    python -m venv "$GITENV"
    # shellcheck disable=SC1091
    source "$GITENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet "git+https://github.com/mmschlk/shapiq.git" \
        scikit-learn xgboost lightgbm pandas joblib
    # Install this package into the venv too (editable) so the runner is importable.
    pip install --quiet -e .
    python -m treeshap_bench.runner \
        --adapters shapiq \
        --model "$MODEL" --n-estimators "$N_EST" --max-depth "$DEPTH" \
        --tasks sv interaction \
        --sv-samples "$SV_SAMPLES" --interaction-samples "$INT_SAMPLES" --rounds "$ROUNDS" \
        --out "$RESULTS/envB_git.json"
    deactivate
else
    echo "  SKIPPED: git-main shapiq (v1.5.0) needs Python >= 3.12; this env is $PYVER."
    echo "  Run this block in a 3.12+ environment to capture the git target."
fi

echo "Done. Merge results in the notebook from $RESULTS/*.json"
