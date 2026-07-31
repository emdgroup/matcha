"""Statistical comparison of model performance across paired data splits.

The ``summarize`` CLI compares several models that were all evaluated on the
*same* set of data splits (CV folds, time windows, clusters, or external
files). Because every model sees the same splits, the observations are
**paired** (a within-subject / repeated-measures design): the split-to-split
variance is shared across models and must be partitioned out rather than
pooled into the error term.

This module provides two comparison strategies that both respect the paired
design:

* :class:`ParametricComparison` -- one-way **repeated-measures ANOVA** as the
  omnibus test, followed by **Tukey HSD computed against the RM-ANOVA error
  term** for the pairwise comparisons. Both are implemented directly on top of
  numpy + scipy (no extra runtime dependency). Bonferroni correction is
  applied across the endpoint-metric combinations at the omnibus level; the
  studentized-range Tukey controls the family-wise error rate of the pairwise
  tests intrinsically, so no further pairwise correction is applied.
* :class:`NonParametricComparison` -- Friedman chi-squared omnibus test
  followed by pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg
  correction. This path is unchanged in behaviour from the previous inline
  implementation in ``summarize.py``.

The parametric methodology (RM-ANOVA + Tukey against the ANOVA error term)
follows the Polaris model-comparison example:
https://github.com/polaris-hub/polaris-method-comparison/blob/main/ADME_example/model_comparison.py
"""

import re
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control, friedmanchisquare


def _rm_anova(matrix: np.ndarray) -> Tuple[float, float, float, int]:
    """One-way repeated-measures ANOVA.

    Partitions the total sum of squares into a between-conditions term, a
    between-subjects term, and the residual error term. The F statistic tests
    the between-conditions effect against the residual, with the subject
    (split) variance removed -- the key difference from a one-way ANOVA, which
    would wrongly pool the subject variance into the error.

    :param matrix: 2-D array shaped ``(n_subjects, k_conditions)``. Here rows
        are data splits (the within-subject factor) and columns are models
        (the grouping factor).
    :returns: Tuple ``(f_stat, p_value, ms_error, df_error)``. ``ms_error``
        and ``df_error`` are returned so the pairwise Tukey HSD can reuse the
        same residual term.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_subjects, k_conditions = matrix.shape

    grand_mean = matrix.mean()
    condition_means = matrix.mean(axis=0)  # one mean per model
    subject_means = matrix.mean(axis=1)  # one mean per split

    ss_conditions = n_subjects * np.sum((condition_means - grand_mean) ** 2)
    ss_subjects = k_conditions * np.sum((subject_means - grand_mean) ** 2)
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_conditions - ss_subjects

    df_conditions = k_conditions - 1
    df_error = df_conditions * (n_subjects - 1)

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error

    # Perfect agreement between models leaves no residual variance; report an
    # infinite F (p == 0) rather than dividing by zero.
    if ms_error <= 0:
        f_stat = np.inf
        p_value = 0.0
    else:
        f_stat = ms_conditions / ms_error
        p_value = float(stats.f.sf(f_stat, df_conditions, df_error))

    return float(f_stat), p_value, float(ms_error), int(df_error)


def _tukey_hsd_rm(
    condition_means: np.ndarray,
    ms_error: float,
    df_error: int,
    n_subjects: int,
    k_conditions: int,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Tukey HSD pairwise p-values using the RM-ANOVA error term.

    Computes the studentized range statistic
    ``q = |mean_i - mean_j| / sqrt(MS_error / n_subjects)`` for every pair of
    conditions and evaluates it against the studentized range distribution
    with ``k_conditions`` groups and ``df_error`` degrees of freedom. The
    studentized range controls the family-wise error rate across all pairwise
    comparisons intrinsically, so no additional correction is applied.

    :param condition_means: Mean metric value per condition (model).
    :param ms_error: Residual mean square from :func:`_rm_anova`.
    :param df_error: Residual degrees of freedom from :func:`_rm_anova`.
    :param n_subjects: Number of data splits (within-subject observations).
    :param k_conditions: Number of models being compared.
    :returns: Dict mapping ``(i, j)`` condition index pairs to
        ``(q_stat, p_value)``.
    """
    se = np.sqrt(ms_error / n_subjects) if ms_error > 0 else 0.0
    results: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for i, j in combinations(range(k_conditions), 2):
        if se == 0:
            q = np.inf
        else:
            q = abs(condition_means[i] - condition_means[j]) / se
        p = float(stats.studentized_range.sf(q, k_conditions, df_error))
        results[(i, j)] = (float(q), p)
    return results


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction.

    Sorts p-values ascending; for the i-th rank (0-indexed), the adjusted
    p-value is ``min(p * (m - i), 1.0)``. Monotone non-decrease is enforced
    via a running maximum so the output ordering is preserved.

    :param p_values: Uncorrected p-values in original input order.
    :returns: Corrected p-values in the same original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, (original_idx, p) in enumerate(indexed):
        adjusted = min(p * (m - rank), 1.0)
        running_max = max(running_max, adjusted)
        corrected[original_idx] = running_max
    return corrected


class _PairedComparison:
    """Shared logic for paired model comparisons across splits.

    Both subclasses operate on the ``organized_data`` structure produced by
    ``MLflowExperimentSummarizer.extract_split_metrics``:
    ``{split_id: {endpoint: {metric: [value_per_run]}}}``, where the per-run
    list is ordered to match ``run_id_order``.

    Subclasses provide the omnibus statistic (:meth:`_omnibus`), the pairwise
    comparison (:meth:`run_pairwise`), the pairwise correction
    (:meth:`correct_pairwise`), and the descriptive method-name attributes
    used in the summary report.
    """

    #: Per-result label stored under the ``"test"`` key of each omnibus dict.
    omnibus_name: str = ""
    #: Method names recorded in the summary metadata.
    omnibus_method: str = ""
    pairwise_method: str = ""
    correction_method: str = ""

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def _endpoint_metrics(organized_data: Dict) -> set:
        """Collect every ``(endpoint, metric)`` combination present in the data."""
        combos = set()
        for split_data in organized_data.values():
            for endpoint, endpoint_data in split_data.items():
                for metric_name in endpoint_data:
                    combos.add((endpoint, metric_name))
        return combos

    @staticmethod
    def _build_matrix(
        organized_data: Dict, endpoint: str, metric_name: str
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Build per-run metric arrays across splits for one endpoint-metric.

        :returns: Tuple ``(valid_arrays, valid_indices, split_ids)`` where
            *valid_arrays* holds one array per run that has a value in every
            split (each array indexed by split), *valid_indices* are the
            corresponding run positions in ``run_id_order``, and *split_ids*
            are the sorted split identifiers contributing data.
        """
        split_ids = sorted(
            s
            for s in organized_data
            if metric_name in organized_data[s].get(endpoint, {})
        )
        if not split_ids:
            return [], [], []

        n_runs = len(organized_data[split_ids[0]][endpoint][metric_name])
        run_values: List[List[float]] = [[] for _ in range(n_runs)]
        for split_id in split_ids:
            metric_vals = organized_data[split_id][endpoint].get(metric_name, [])
            for run_idx in range(min(n_runs, len(metric_vals))):
                run_values[run_idx].append(metric_vals[run_idx])

        valid_indices = [
            i for i in range(n_runs) if len(run_values[i]) == len(split_ids)
        ]
        valid_arrays = [np.array(run_values[i]) for i in valid_indices]
        return valid_arrays, valid_indices, split_ids

    def run_omnibus(
        self, organized_data: Dict
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Run the omnibus test for each endpoint-metric, Bonferroni-corrected.

        Assesses overall group differences across runs for every
        endpoint-metric combination, then applies a Bonferroni correction
        across the number of combinations tested.

        :param organized_data: Metrics organized by split/endpoint/metric.
        :returns: Dict mapping ``(endpoint, metric)`` tuples to result dicts
            with ``statistic``, ``p_value``, ``p_value_corrected``, ``test``,
            and related keys.
        """
        omnibus_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        combos = self._endpoint_metrics(organized_data)
        self.logger.info(
            f"Running omnibus tests ({self.omnibus_name}) for {len(combos)} "
            "endpoint-metric combinations"
        )

        for endpoint, metric_name in combos:
            valid_arrays, _, split_ids = self._build_matrix(
                organized_data, endpoint, metric_name
            )
            if len(valid_arrays) < 2:
                self.logger.warning(
                    "Not enough runs with complete data for omnibus test on "
                    f"{endpoint}-{metric_name}"
                )
                continue
            if len(split_ids) < 2:
                self.logger.warning(
                    f"Omnibus test requires at least 2 splits, got {len(split_ids)} "
                    f"for {endpoint}-{metric_name}"
                )
                continue

            try:
                result = self._omnibus(valid_arrays)
            except Exception as e:
                self.logger.warning(
                    f"Failed {self.omnibus_name} omnibus test for "
                    f"{endpoint}-{metric_name}: {e}"
                )
                continue

            result.update(
                {
                    "test": self.omnibus_name,
                    "n_runs": len(valid_arrays),
                    "n_splits": len(split_ids),
                }
            )
            omnibus_results[(endpoint, metric_name)] = result

        self._apply_bonferroni(omnibus_results)
        return omnibus_results

    def _apply_bonferroni(self, omnibus_results: Dict) -> None:
        """Bonferroni-correct omnibus p-values in place across all metrics."""
        if not omnibus_results:
            return
        n_tests = len(omnibus_results)
        for result in omnibus_results.values():
            corrected_p = min(result["p_value"] * n_tests, 1.0)
            result["p_value_corrected"] = corrected_p
            result["bonferroni_n_tests"] = n_tests
            result["significant_05"] = corrected_p < 0.05
            result["significant_01"] = corrected_p < 0.01
        self.logger.info(
            f"Applied Bonferroni correction (n={n_tests}) to omnibus test p-values"
        )

    def _omnibus(self, valid_arrays: List[np.ndarray]) -> Dict[str, Any]:
        """Compute the omnibus statistic for one endpoint-metric.

        :param valid_arrays: One array per run, each indexed by split.
        :returns: Dict with at least ``statistic`` and ``p_value``.
        """
        raise NotImplementedError

    def run_pairwise(
        self, organized_data: Dict, run_id_order: List[str], run_id_to_name: Dict
    ) -> List[Dict]:
        """Run pairwise comparisons for each endpoint-metric combination."""
        raise NotImplementedError

    def correct_pairwise(self, comparison_results: List[Dict]) -> List[Dict]:
        """Apply any multiple-testing correction to the pairwise results."""
        raise NotImplementedError


class ParametricComparison(_PairedComparison):
    """Repeated-measures ANOVA omnibus + Tukey-on-residuals pairwise tests."""

    omnibus_name = "rm_anova"
    omnibus_method = "rm_anova"
    pairwise_method = "tukey_hsd"
    correction_method = "bonferroni (omnibus) + tukey_hsd studentized-range (pairwise)"

    def _omnibus(self, valid_arrays: List[np.ndarray]) -> Dict[str, Any]:
        # Rows = splits (subjects), columns = models (conditions).
        matrix = np.column_stack(valid_arrays)
        f_stat, p_value, ms_error, df_error = _rm_anova(matrix)
        return {
            "statistic": f_stat,
            "p_value": p_value,
            "ms_error": ms_error,
            "df_error": df_error,
        }

    def run_pairwise(
        self, organized_data: Dict, run_id_order: List[str], run_id_to_name: Dict
    ) -> List[Dict]:
        """Tukey HSD pairwise comparisons against the RM-ANOVA error term.

        Tukey's studentized range controls the family-wise error rate within
        each endpoint-metric combination, so no additional correction is
        applied downstream.
        """
        comparison_results: List[Dict] = []

        for endpoint, metric_name in self._endpoint_metrics(organized_data):
            valid_arrays, valid_indices, split_ids = self._build_matrix(
                organized_data, endpoint, metric_name
            )
            if len(valid_arrays) < 2 or len(split_ids) < 2:
                continue

            try:
                matrix = np.column_stack(valid_arrays)
                _, _, ms_error, df_error = _rm_anova(matrix)
                condition_means = matrix.mean(axis=0)
                pair_stats = _tukey_hsd_rm(
                    condition_means,
                    ms_error,
                    df_error,
                    n_subjects=len(split_ids),
                    k_conditions=len(valid_arrays),
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed Tukey HSD for {endpoint}-{metric_name}: {e}"
                )
                continue

            for (idx_i, idx_j), (q_stat, p_value) in pair_stats.items():
                run_i_id = run_id_order[valid_indices[idx_i]]
                run_j_id = run_id_order[valid_indices[idx_j]]
                run_i_name = run_id_to_name.get(run_i_id, f"run_{run_i_id[:8]}")
                run_j_name = run_id_to_name.get(run_j_id, f"run_{run_j_id[:8]}")

                comparison_results.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric_name,
                        "run_pair": f"{run_i_name}_vs_{run_j_name}",
                        "run_i_name": run_i_name,
                        "run_j_name": run_j_name,
                        "run_i_id": run_i_id,
                        "run_j_id": run_j_id,
                        "statistic": q_stat,
                        "p_value": p_value,
                        "p_value_corrected": p_value,  # studentized range self-corrects
                        "significant_05": p_value < 0.05,
                        "significant_01": p_value < 0.01,
                        "n_observations": len(split_ids),
                        "test": self.pairwise_method,
                    }
                )

        self.logger.info(
            f"Completed {len(comparison_results)} Tukey HSD pairwise comparisons"
        )
        return comparison_results

    def correct_pairwise(self, comparison_results: List[Dict]) -> List[Dict]:
        """No-op: Tukey's studentized range already controls family-wise error."""
        return comparison_results


class NonParametricComparison(_PairedComparison):
    """Friedman omnibus + Wilcoxon signed-rank pairwise with BH correction."""

    omnibus_name = "Friedman"
    omnibus_method = "friedman_chi_squared"
    pairwise_method = "wilcoxon_signed_rank"
    correction_method = "bonferroni (omnibus) + benjamini_hochberg (pairwise)"

    def _omnibus(self, valid_arrays: List[np.ndarray]) -> Dict[str, Any]:
        # Friedman treats each array as one model's scores across the paired
        # splits; it requires at least two splits (enforced by run_omnibus).
        statistic, p_value = friedmanchisquare(*valid_arrays)
        return {"statistic": float(statistic), "p_value": float(p_value)}

    def run_pairwise(
        self, organized_data: Dict, run_id_order: List[str], run_id_to_name: Dict
    ) -> List[Dict]:
        """Pairwise Wilcoxon signed-rank tests across paired splits."""
        comparison_results: List[Dict] = []

        # Get all unique run pairs
        first_split = list(organized_data.keys())[0]
        first_endpoint = list(organized_data[first_split].keys())[0]
        first_metric = list(organized_data[first_split][first_endpoint].keys())[0]
        n_runs = len(organized_data[first_split][first_endpoint][first_metric])

        run_pairs = list(combinations(range(n_runs), 2))
        self.logger.info(f"Running pairwise comparisons for {len(run_pairs)} run pairs")

        for split_id, split_data in organized_data.items():
            for endpoint, endpoint_data in split_data.items():
                for metric_name, metric_values in endpoint_data.items():
                    for i, j in run_pairs:
                        # Extract paired values across all splits for this metric-endpoint combination
                        values_run_i = []
                        values_run_j = []

                        # Collect values for this metric-endpoint pair across all splits
                        for other_split_id, other_split_data in organized_data.items():
                            if other_split_id in ["mean", "std"]:
                                continue
                            if (
                                endpoint in other_split_data
                                and metric_name in other_split_data[endpoint]
                            ):
                                if len(other_split_data[endpoint][metric_name]) > max(
                                    i, j
                                ):
                                    values_run_i.append(
                                        other_split_data[endpoint][metric_name][i]
                                    )
                                    values_run_j.append(
                                        other_split_data[endpoint][metric_name][j]
                                    )

                        if len(values_run_i) > 1:  # Need at least 2 paired observations
                            try:
                                # Wilcoxon signed-rank test for paired samples
                                statistic, p_value = stats.wilcoxon(
                                    values_run_i,
                                    values_run_j,
                                    alternative="two-sided",
                                    zero_method="wilcox",
                                )

                                # Get run names from indices
                                run_i_id = run_id_order[i]
                                run_j_id = run_id_order[j]
                                run_i_name = run_id_to_name.get(
                                    run_i_id, f"run_{run_i_id[:8]}"
                                )
                                run_j_name = run_id_to_name.get(
                                    run_j_id, f"run_{run_j_id[:8]}"
                                )

                                comparison_results.append(
                                    {
                                        "split_id": split_id,
                                        "endpoint": endpoint,
                                        "metric": metric_name,
                                        "run_pair": f"{run_i_name}_vs_{run_j_name}",
                                        "run_i_name": run_i_name,
                                        "run_j_name": run_j_name,
                                        "run_i_id": run_i_id,
                                        "run_j_id": run_j_id,
                                        "statistic": float(statistic),
                                        "p_value": float(p_value),
                                        "n_observations": len(values_run_i),
                                    }
                                )

                            except Exception as e:
                                self.logger.warning(
                                    f"Failed Wilcoxon test for {endpoint}-{metric_name} "
                                    f"runs {i} vs {j}: {e}"
                                )

        self.logger.info(f"Completed {len(comparison_results)} pairwise comparisons")
        return comparison_results

    def correct_pairwise(self, comparison_results: List[Dict]) -> List[Dict]:
        """Benjamini-Hochberg correction, grouped by endpoint-metric combination."""
        # Group results by endpoint and metric
        grouped_results: Dict[Tuple[str, str], List[Dict]] = {}
        for result in comparison_results:
            key = (result["endpoint"], result["metric"])
            grouped_results.setdefault(key, []).append(result)

        corrected_results: List[Dict] = []
        total_corrections = 0

        # Apply BH correction separately for each endpoint-metric combination
        for (endpoint, metric), group_results in grouped_results.items():
            # Extract p-values for this group
            p_values = [result["p_value"] for result in group_results]

            # Apply Benjamini-Hochberg correction
            try:
                corrected_p_values = false_discovery_control(p_values, method="bh")
            except Exception as e:
                self.logger.warning(
                    f"Failed BH correction for {endpoint}-{metric}: {e}"
                )
                corrected_p_values = p_values

            # Add corrected p-values to results
            for i, result in enumerate(group_results):
                result["p_value_corrected"] = float(corrected_p_values[i])
                result["significant_05"] = bool(corrected_p_values[i] < 0.05)
                result["significant_01"] = bool(corrected_p_values[i] < 0.01)

            corrected_results.extend(group_results)
            total_corrections += 1

            self.logger.info(
                f"Applied BH correction for {endpoint}-{metric}: {len(p_values)} comparisons"
            )

        self.logger.info(
            f"Applied Benjamini-Hochberg correction to {len(corrected_results)} p-values "
            f"across {total_corrections} endpoint-metric combinations"
        )

        return corrected_results


class BootstrapComparison(_PairedComparison):
    """Bootstrap percentile CIs + Holm-Bonferroni pairwise comparisons.

    Uses the ``"{split_idx}_{bootstrap_idx}"`` keys written by ``evaluate``
    when ``split.n_bootstrap > 1``. Each bootstrap sample is paired across
    runs because all models were evaluated with the same fixed seed, so
    element-wise differences form a valid paired distribution.
    """

    omnibus_name = "bootstrap_skipped"
    omnibus_method = "none"
    pairwise_method = "bootstrap_percentile"
    correction_method = "holm_bonferroni (pairwise, per endpoint-metric)"

    def run_omnibus(self, organized_data: Dict) -> Dict:
        """Bootstrap has no omnibus test; logs a warning and returns empty."""
        self.logger.warning(
            "Bootstrap comparison has no omnibus test; skipping omnibus step."
        )
        return {}

    def run_pairwise(
        self, organized_data: Dict, run_id_order: List[str], run_id_to_name: Dict
    ) -> List[Dict]:
        """Pairwise bootstrap percentile CIs and empirical p-values."""
        bootstrap_data = {
            k: v for k, v in organized_data.items() if re.match(r"^\d+_\d+$", str(k))
        }
        if not bootstrap_data:
            raise ValueError(
                "No bootstrap keys found in organized_data. "
                "Set split.n_bootstrap > 1 in the evaluate config."
            )

        comparison_results: List[Dict] = []

        for endpoint, metric_name in self._endpoint_metrics(organized_data):
            valid_arrays, valid_indices, split_ids = self._build_matrix(
                bootstrap_data, endpoint, metric_name
            )
            if len(valid_arrays) < 2:
                continue

            n_bs = len(split_ids)
            for idx_i, idx_j in combinations(range(len(valid_arrays)), 2):
                deltas = valid_arrays[idx_i] - valid_arrays[idx_j]
                statistic = float(np.mean(deltas))
                ci_low = float(np.percentile(deltas, 2.5))
                ci_high = float(np.percentile(deltas, 97.5))
                frac_pos = float(np.sum(deltas > 0)) / n_bs
                frac_neg = float(np.sum(deltas < 0)) / n_bs
                p_value = 2.0 * min(frac_pos, frac_neg)

                run_i_id = run_id_order[valid_indices[idx_i]]
                run_j_id = run_id_order[valid_indices[idx_j]]
                run_i_name = run_id_to_name.get(run_i_id, f"run_{run_i_id[:8]}")
                run_j_name = run_id_to_name.get(run_j_id, f"run_{run_j_id[:8]}")

                comparison_results.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric_name,
                        "run_pair": f"{run_i_name}_vs_{run_j_name}",
                        "run_i_name": run_i_name,
                        "run_j_name": run_j_name,
                        "run_i_id": run_i_id,
                        "run_j_id": run_j_id,
                        "statistic": statistic,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                        "n_observations": n_bs,
                        "test": self.pairwise_method,
                    }
                )

        self.logger.info(
            f"Completed {len(comparison_results)} bootstrap pairwise comparisons"
        )
        return comparison_results

    def correct_pairwise(self, comparison_results: List[Dict]) -> List[Dict]:
        """Holm-Bonferroni correction grouped by endpoint-metric combination."""
        grouped: Dict[Tuple[str, str], List[Dict]] = {}
        for result in comparison_results:
            key = (result["endpoint"], result["metric"])
            grouped.setdefault(key, []).append(result)

        corrected_results: List[Dict] = []
        for (endpoint, metric), group in grouped.items():
            p_values = [r["p_value"] for r in group]
            corrected = _holm_bonferroni(p_values)
            for r, p_corr in zip(group, corrected):
                r["p_value_corrected"] = p_corr
                r["significant_05"] = p_corr < 0.05
                r["significant_01"] = p_corr < 0.01
            corrected_results.extend(group)
            self.logger.info(
                f"Applied Holm-Bonferroni correction for {endpoint}-{metric}: "
                f"{len(p_values)} comparisons"
            )

        self.logger.info(
            f"Applied Holm-Bonferroni correction to {len(corrected_results)} p-values"
        )
        return corrected_results


def build_comparison(statistical_test: str, logger) -> _PairedComparison:
    """Construct the comparison strategy for the requested testing mode.

    :param statistical_test: ``"parametric"`` or ``"non-parametric"``.
    :param logger: Logger passed through to the comparison instance.
    :returns: A :class:`ParametricComparison` or
        :class:`NonParametricComparison`.
    :raises ValueError: If *statistical_test* is not a supported mode.
    """
    if statistical_test == "parametric":
        return ParametricComparison(logger)
    if statistical_test == "non-parametric":
        return NonParametricComparison(logger)
    if statistical_test == "bootstrap":
        return BootstrapComparison(logger)
    raise ValueError(
        f"statistical_test must be 'parametric', 'non-parametric', or 'bootstrap', "
        f"got '{statistical_test}'"
    )
