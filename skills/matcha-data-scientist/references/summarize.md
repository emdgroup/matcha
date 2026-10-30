# summarize

**CLI command:** `matcha summarize --config summarize.yaml`

Aggregates evaluation results from multiple model runs and performs statistical testing to determine whether performance differences are significant. Supports two input modes: directory mode (point at a folder of result subdirectories) and MLflow mode (pull runs from an MLflow experiment).

---

## YAML Schema (`CLISummarizeInputModel`)

Two mutually exclusive modes. Use **one** of them — never both.

### Directory Mode
```yaml
root_dir: <str>                      # required for directory mode
                                     # each immediate subdirectory = one model run
output_path: <str>                   # optional — where to write outputs (default: "./outputs")
statistical_test: <str>              # optional — "non-parametric" (default), "parametric", or "bootstrap"
runs: <list[str]>                    # optional — subset of subdirectory names to compare
```

### MLflow Mode
```yaml
experiment_name: <str>               # required for MLflow mode
mlruns_path: <str>                   # required for MLflow mode — path to mlruns directory
output_path: <str>                   # optional (default: "./outputs")
statistical_test: <str>              # optional — "non-parametric" (default), "parametric", or "bootstrap"
runs: <list[str]>                    # optional — subset of MLflow run names to compare
```

> **Validation:** Providing both `root_dir` and `experiment_name`/`mlruns_path` raises a `ValueError`. Providing neither also raises an error.

---

## Step-by-Step Config Generation

### 1. Input source mode
Ask: "Did your evaluate/baseline runs log to MLflow, or should we compare result directories directly?"

- **Directory mode** — simpler; point at a folder where each subdirectory is one model run.
- **MLflow mode** — use when runs were tracked via MLflow with `output.mlflow.experiment_name`.

### 2. Run filtering (optional)
Leave `runs` unset to compare all runs/subdirectories, or set it to a list of names to restrict the comparison.

### 3. Statistical test mode
Ask: "Do you want the default non-parametric test, the parametric test, or the bootstrap test?"

**Non-parametric (default — recommended):**
- Friedman test → Wilcoxon signed-rank test (pairwise) → Benjamini-Hochberg FDR correction.
- Use when: small number of splits, non-normal metric distributions, or when unsure.

**Parametric:**
- Repeated-measures ANOVA → Tukey HSD (pairwise).
- Use when: many splits (≥ 10), metrics are approximately normally distributed, and sphericity holds.

**Bootstrap:**
- No omnibus test. Bootstrap percentile CIs + empirical p-values (pairwise) → Holm-Bonferroni correction.
- **Requires** `split.n_bootstrap > 1` in the evaluate config — split keys must follow the `{split_idx}_{bootstrap_idx}` pattern written by `evaluate`.
- Use when: you ran evaluate with bootstrap resampling and want CI-based pairwise comparisons instead of rank tests.

### 4. Generate YAML, confirm, and run

```
matcha summarize --config summarize.yaml
```

---

## Directory Mode Example

```yaml
root_dir: ./results
output_path: ./summary
statistical_test: non-parametric
runs:
  - chemprop_eval
  - gatedgcn_eval
  - baseline_rf
```

## MLflow Mode Example

```yaml
experiment_name: evaluating-fu
mlruns_path: ./mlruns
output_path: ./summary
statistical_test: non-parametric
```

---

## Key Behaviors

- **Per-endpoint results:** Statistical tests are run independently for each endpoint.
- **Run labels:** In directory mode, the subdirectory name is the run label. Rename subdirectories before running summarize to get clean labels in plots.
- **Non-parametric path:** Friedman test (omnibus) → Wilcoxon signed-rank test (pairwise) → Benjamini-Hochberg FDR correction.
- **Parametric path:** Repeated-measures ANOVA → Tukey HSD (pairwise). Requires normality and sphericity.
- **Bootstrap path:** No omnibus test. Bootstrap percentile CIs + empirical p-values (pairwise) → Holm-Bonferroni correction. Requires `split.n_bootstrap > 1` in the evaluate config.
- **Output:** `summary_analysis.json` with per-endpoint metric means, standard deviations, and statistical test p-values, plus HTML comparison plots.
