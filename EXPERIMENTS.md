# TreeSHAPBench — Experiment Design

A structured guide to the axes worth exploring in this benchmark, the checks that validate correctness, and the gaps still left to fill.

---

## 1. Model Complexity / Training Budget

The most important axis for tree-based SHAP because algorithm complexity depends **directly on tree structure**.

| Parameter | Values to test | Expected scaling |
|---|---|---|
| `n_estimators` | 50 → 200 → 500 → 1 000 | Linear — verify empirically |
| `max_depth` | 3–4 (shallow), 6–8 (medium), 12+ (deep), unlimited | O(L · D²) per sample |
| Growth strategy | depth-wise (sklearn RF, XGBoost) vs leaf-wise (LightGBM) | Unbalanced trees at same `max_depth` behave very differently |

> **Note on interactions:** interaction values are O(L · D² · M) where M = number of features. Deep trees expose the gap between methods dramatically.

---

## 2. Dataset Properties

### 2a. Dimensionality

| Property | Values | Notes |
|---|---|---|
| Features `d` | 10, 50, 100, 200+ | Biggest driver of interaction cost — O(d²) |
| Samples to explain `n` | varies | Should scale linearly for all methods; confirm with a log-log plot |

Synthetic datasets let you control `d` cleanly without confounding factors.

### 2b. Feature Correlation

High correlation makes **path-dependent** and **interventional** SHAP diverge significantly.
- Test with a correlated synthetic dataset, e.g. multivariate Gaussian with ρ ≈ 0.8.

### 2c. Sparsity

Tabular datasets with many zero-valued features (e.g. one-hot encoded text) can behave very differently from dense numerical data.

### 2d. Task Type

| Task | Notes |
|---|---|
| Binary classification | Standard case |
| Regression | Standard case |
| Multiclass | Output is a list of K arrays — libraries often disagree here |

---

## 3. Correctness / Axiomatic Checks

These validate the **values**, not just the timing.

| Axiom | Check |
|---|---|
| **Efficiency** | `sum(SHAP_i) == f(x) − E[f(x)]` for every sample — should hold exactly |
| **Dummy** | A feature that never appears in any split must receive zero attribution |
| **Symmetry** | If two features always co-occur identically in splits, their Shapley values must be equal |
| **Interaction consistency** | Row-sums of the interaction matrix must equal the Shapley values: Σⱼ Φᵢⱼ = φᵢ |
| **Cross-method agreement** | When methods should agree (path-dependent, order-1 SV), quantify residual error — not just `max|diff|` but the full error distribution |

---

## 4. Perturbation Mode — Path-Dependent vs Interventional

| Mode | Speed | Handles correlation? | Notes |
|---|---|---|---|
| Path-dependent (no background) | Fast | ✗ | Can assign non-zero values to dummy features in correlated settings |
| Interventional (with background data) | Slower | ✓ | Sensitive to background set size |

### What to measure

1. **Timing difference** between modes.
2. **Value difference** between modes on a correlated dataset.
3. **Background set size effect:** 100 vs 500 vs full training set — shows how approximation error shrinks.

---

## 5. Interaction Index Choice

Not all libraries compute the same thing:

| Index | Definition | Supported by |
|---|---|---|
| TreeSHAP interaction values | Lundberg et al. 2020 | `shap`, `woodelf` |
| SII | Shapley Interaction Index | `shapiq` |
| k-SII | Aggregated higher-order | `shapiq` |
| STII | Shapley-Taylor | `shapiq` |
| BII | Banzhaf interaction | `woodelf`, `shapiq` |

**Key questions to answer:**
- Do SII row-sums equal SVs? (They should by definition.)
- When do k-SII and SII agree at order 2?

---

## 6. Scalability Checks (Log-Log Plots)

| Dimension | Expected scaling |
|---|---|
| `n` samples | O(n) for all methods |
| `d` features — Shapley values | O(d) — one pass per feature split |
| `d` features — interactions | O(d²) — verify empirically |
| `n_estimators` | O(n_estimators) |
| `max_depth` | O(D²) — critical to verify for deep trees |

A log-log plot of each dimension vs. runtime confirms or breaks these assumptions.

---

## 7. Synthetic Ground-Truth Datasets

Real datasets have unknown ground truth. Synthetic ones let you **verify correctness analytically**.

| Dataset | Why it's useful |
|---|---|
| Linear model with additive features | SVs equal linear coefficients × (x − mean) — exact closed form |
| Single decision tree | SVs are analytically tractable |
| Correlated Gaussian + linear model | Shows where path-dependent SHAP assigns wrong attribution |
| Interaction-only model (e.g. f = x₁ · x₂) | All attribution should live in the interaction term with zero main effects — tests whether interaction values correctly capture this |

---

## 8. Gaps Worth Filling

The current notebooks cover multiple model types, SV vs. interactions, and a basic correctness check.
The following experiments are still missing:

- [ ] **Scaling plots** — `n_estimators × runtime` and `max_depth × runtime` (log-log)
- [ ] **Feature dimensionality sweep** — synthetic datasets at d ∈ {10, 50, 200}
- [ ] **Path-dependent vs interventional comparison** on a correlated dataset
- [ ] **Axiomatic checks** — efficiency axiom for every sample; interaction row-sum = SV
- [ ] **Interaction index comparison** — SII vs k-SII vs TreeSHAP on a simple synthetic case
