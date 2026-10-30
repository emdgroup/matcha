"""Tests for matcha.cli.statistical_tests (issue #420).

Covers the new parametric path (repeated-measures ANOVA + Tukey HSD against
the RM-ANOVA error term) and a regression guard confirming the relocated
non-parametric path (Friedman + Wilcoxon + BH) still matches the underlying
scipy calls.
"""

import logging

import numpy as np
import pytest
from scipy import stats
from scipy.stats import false_discovery_control, friedmanchisquare

from matcha.cli.statistical_tests import (
    BootstrapComparison,
    NonParametricComparison,
    ParametricComparison,
    _holm_bonferroni,
    _rm_anova,
    _tukey_hsd_rm,
    build_comparison,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_statistical_tests")


# A 4-split x 3-model fixture with a hand-derived RM-ANOVA solution:
#   grand mean = 25/6
#   SS_conditions = 22.16667 (df=2), SS_subjects = 29.66667 (df=3),
#   SS_total = 53.66667, SS_error = 11/6 (df_error=6)
#   MS_error = 11/36, MS_conditions = 133/12
#   F = (133/12) / (11/36) = 399/11 = 36.2727...
FIXTURE_MATRIX = np.array(
    [
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 5.0],
        [3.0, 3.0, 6.0],
        [5.0, 6.0, 9.0],
    ]
)


# =========================================================================
# _rm_anova
# =========================================================================


class TestRmAnova:
    def test_matches_hand_computed_fixture(self):
        f_stat, p_value, ms_error, df_error = _rm_anova(FIXTURE_MATRIX)
        assert np.isclose(f_stat, 399 / 11)
        assert np.isclose(ms_error, 11 / 36)
        assert df_error == 6
        assert np.isclose(p_value, stats.f.sf(399 / 11, 2, 6))
        assert p_value < 0.001

    def test_partitions_subject_variance_out_of_error(self):
        # A one-way ANOVA would pool the large split-to-split (subject)
        # variance into the error term, deflating F. RM-ANOVA removes it, so
        # its F must exceed the one-way F on the same data.
        f_rm, _, _, _ = _rm_anova(FIXTURE_MATRIX)
        f_oneway, _ = stats.f_oneway(*FIXTURE_MATRIX.T)
        assert f_rm > f_oneway

    def test_perfect_agreement_gives_infinite_f(self):
        # Identical columns -> zero residual variance -> F == inf, p == 0,
        # handled without a ZeroDivisionError.
        matrix = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        f_stat, p_value, ms_error, df_error = _rm_anova(matrix)
        assert np.isinf(f_stat)
        assert p_value == 0.0
        assert np.isclose(ms_error, 0.0)


# =========================================================================
# _tukey_hsd_rm
# =========================================================================


class TestTukeyHsdRm:
    def test_matches_studentized_range_formula(self):
        _, _, ms_error, df_error = _rm_anova(FIXTURE_MATRIX)
        means = FIXTURE_MATRIX.mean(axis=0)
        n_subjects, k = FIXTURE_MATRIX.shape

        results = _tukey_hsd_rm(means, ms_error, df_error, n_subjects, k)
        se = np.sqrt(ms_error / n_subjects)

        # Every pair of conditions is present.
        assert set(results.keys()) == {(0, 1), (0, 2), (1, 2)}

        for (i, j), (q, p) in results.items():
            expected_q = abs(means[i] - means[j]) / se
            assert np.isclose(q, expected_q)
            assert np.isclose(p, stats.studentized_range.sf(expected_q, k, df_error))

    def test_larger_separation_gives_smaller_p(self):
        _, _, ms_error, df_error = _rm_anova(FIXTURE_MATRIX)
        means = FIXTURE_MATRIX.mean(axis=0)
        n_subjects, k = FIXTURE_MATRIX.shape
        results = _tukey_hsd_rm(means, ms_error, df_error, n_subjects, k)
        # means are [2.75, 3.75, 6.0]; pair (0,2) is the most separated.
        assert results[(0, 2)][1] < results[(0, 1)][1]


# =========================================================================
# Organized-data fixture shared by the comparison-class tests
# =========================================================================


@pytest.fixture
def organized_data():
    # 4 splits, 3 runs (models). Per-split list order matches run_id_order.
    return {
        "split_0": {"ep": {"r2": [0.80, 0.70, 0.60]}},
        "split_1": {"ep": {"r2": [0.82, 0.69, 0.58]}},
        "split_2": {"ep": {"r2": [0.79, 0.71, 0.62]}},
        "split_3": {"ep": {"r2": [0.81, 0.68, 0.59]}},
    }


@pytest.fixture
def run_id_order():
    return ["A", "B", "C"]


@pytest.fixture
def run_id_to_name():
    return {"A": "modelA", "B": "modelB", "C": "modelC"}


# =========================================================================
# ParametricComparison
# =========================================================================


class TestParametricComparison:
    def test_omnibus_reports_rm_anova_with_bonferroni(self, logger, organized_data):
        comp = ParametricComparison(logger)
        results = comp.run_omnibus(organized_data)

        assert set(results.keys()) == {("ep", "r2")}
        res = results[("ep", "r2")]
        assert res["test"] == "rm_anova"
        assert res["n_splits"] == 4
        assert res["n_runs"] == 3
        # Bonferroni keys are added; with a single test the factor is 1.
        assert res["bonferroni_n_tests"] == 1
        assert np.isclose(res["p_value_corrected"], res["p_value"])
        # Cross-check the F against a direct RM-ANOVA on the same matrix.
        matrix = np.array(
            [
                [0.80, 0.70, 0.60],
                [0.82, 0.69, 0.58],
                [0.79, 0.71, 0.62],
                [0.81, 0.68, 0.59],
            ]
        )
        f_stat, _, _, _ = _rm_anova(matrix)
        assert np.isclose(res["statistic"], f_stat)

    def test_pairwise_returns_tukey_results(
        self, logger, organized_data, run_id_order, run_id_to_name
    ):
        comp = ParametricComparison(logger)
        results = comp.run_pairwise(organized_data, run_id_order, run_id_to_name)

        # 3 models -> 3 pairwise comparisons for the single endpoint-metric.
        assert len(results) == 3
        pairs = {r["run_pair"] for r in results}
        assert pairs == {
            "modelA_vs_modelB",
            "modelA_vs_modelC",
            "modelB_vs_modelC",
        }
        for r in results:
            assert r["test"] == "tukey_hsd"
            assert r["n_observations"] == 4
            # Tukey self-corrects: corrected p equals the raw p.
            assert r["p_value_corrected"] == r["p_value"]
            assert "significant_05" in r and "significant_01" in r

    def test_correct_pairwise_is_noop(
        self, logger, organized_data, run_id_order, run_id_to_name
    ):
        comp = ParametricComparison(logger)
        results = comp.run_pairwise(organized_data, run_id_order, run_id_to_name)
        assert comp.correct_pairwise(results) is results


# =========================================================================
# NonParametricComparison — regression guard against direct scipy calls
# =========================================================================


class TestNonParametricComparison:
    def test_omnibus_matches_scipy_friedman(self, logger, organized_data):
        comp = NonParametricComparison(logger)
        results = comp.run_omnibus(organized_data)

        res = results[("ep", "r2")]
        assert res["test"] == "Friedman"
        # Direct scipy call on the three per-model arrays across splits.
        run_a = [0.80, 0.82, 0.79, 0.81]
        run_b = [0.70, 0.69, 0.71, 0.68]
        run_c = [0.60, 0.58, 0.62, 0.59]
        stat, p = friedmanchisquare(run_a, run_b, run_c)
        assert np.isclose(res["statistic"], stat)
        assert np.isclose(res["p_value"], p)

    def test_pairwise_matches_scipy_wilcoxon(
        self, logger, organized_data, run_id_order, run_id_to_name
    ):
        comp = NonParametricComparison(logger)
        results = comp.run_pairwise(organized_data, run_id_order, run_id_to_name)

        per_run = {
            "A": [0.80, 0.82, 0.79, 0.81],
            "B": [0.70, 0.69, 0.71, 0.68],
            "C": [0.60, 0.58, 0.62, 0.59],
        }
        name_to_runs = {
            "modelA_vs_modelB": ("A", "B"),
            "modelA_vs_modelC": ("A", "C"),
            "modelB_vs_modelC": ("B", "C"),
        }
        # Each result must match a direct Wilcoxon test on the paired arrays.
        for r in results:
            ri, rj = name_to_runs[r["run_pair"]]
            stat, p = stats.wilcoxon(
                per_run[ri],
                per_run[rj],
                alternative="two-sided",
                zero_method="wilcox",
            )
            assert np.isclose(r["statistic"], stat)
            assert np.isclose(r["p_value"], p)
            assert r["n_observations"] == 4

    def test_correct_pairwise_applies_benjamini_hochberg(
        self, logger, organized_data, run_id_order, run_id_to_name
    ):
        comp = NonParametricComparison(logger)
        results = comp.run_pairwise(organized_data, run_id_order, run_id_to_name)
        corrected = comp.correct_pairwise(results)

        # BH is applied per endpoint-metric group. Dedup by run_pair (the
        # relocated Wilcoxon loop emits one row per outer split) and compare
        # the corrected p-values against a direct false_discovery_control call.
        seen = {}
        for r in corrected:
            seen.setdefault(r["run_pair"], r)
            assert "significant_05" in r and "significant_01" in r

        raw = [r["p_value"] for r in seen.values()]
        expected = false_discovery_control(raw, method="bh")
        got = [r["p_value_corrected"] for r in seen.values()]
        assert np.allclose(sorted(got), sorted(expected))


# =========================================================================
# Bootstrap fixtures and helpers
# =========================================================================


@pytest.fixture
def bootstrap_organized_data():
    # 4 bootstrap samples (keys "0_0", "0_1", "1_0", "1_1"), 2 models, 1 endpoint-metric.
    # modelA values: [0.80, 0.82, 0.79, 0.81]
    # modelB values: [0.70, 0.69, 0.71, 0.68]
    # deltas (A - B):  [0.10, 0.13, 0.08, 0.13]
    return {
        "0_0": {"ep": {"mae": [0.80, 0.70]}},
        "0_1": {"ep": {"mae": [0.82, 0.69]}},
        "1_0": {"ep": {"mae": [0.79, 0.71]}},
        "1_1": {"ep": {"mae": [0.81, 0.68]}},
    }


@pytest.fixture
def bootstrap_run_id_order():
    return ["A", "B"]


@pytest.fixture
def bootstrap_run_id_to_name():
    return {"A": "modelA", "B": "modelB"}


# =========================================================================
# _holm_bonferroni
# =========================================================================


class TestHolmBonferroni:
    def test_single_value_unchanged(self):
        result = _holm_bonferroni([0.04])
        assert len(result) == 1
        assert np.isclose(result[0], 0.04)

    def test_three_values_hand_computed(self):
        # m=3, sorted ascending: 0.01 (rank 0), 0.04 (rank 1), 0.20 (rank 2)
        # adjusted: 0.01*(3-0)=0.03, 0.04*(3-1)=0.08, 0.20*(3-2)=0.20
        # running max enforces non-decrease: [0.03, 0.08, 0.20]
        result = _holm_bonferroni([0.01, 0.04, 0.20])
        assert np.allclose(result, [0.03, 0.08, 0.20])

    def test_monotone_output(self):
        p_values = [0.30, 0.01, 0.15, 0.05]
        result = _holm_bonferroni(p_values)
        # output in original order must be non-decreasing when sorted
        assert result == sorted(result) or all(
            result[i] <= result[i + 1] + 1e-12
            for i in range(len(result) - 1)
            if sorted(p_values).index(p_values[i])
            < sorted(p_values).index(p_values[i + 1])
        )
        # simpler check: each corrected >= corresponding raw
        for raw, corr in zip(p_values, result):
            assert corr >= raw - 1e-12

    def test_caps_at_one(self):
        # 0.50 * 2 = 1.0; 0.80 * 1 = 0.80 but running_max forces it to stay >= 1.0
        result = _holm_bonferroni([0.50, 0.80])
        assert result[0] == 1.0
        assert result[1] == 1.0

    def test_empty_input(self):
        assert _holm_bonferroni([]) == []

    def test_original_order_preserved(self):
        # Input [0.20, 0.01]: sorted ascending gives 0.01 at rank 0, 0.20 at rank 1.
        # adjusted: 0.01*2=0.02, 0.20*1=0.20
        result = _holm_bonferroni([0.20, 0.01])
        assert np.isclose(result[0], 0.20)
        assert np.isclose(result[1], 0.02)


# =========================================================================
# TestBootstrapComparison
# =========================================================================


class TestBootstrapComparison:
    def test_omnibus_returns_empty_dict(self, logger, bootstrap_organized_data):
        comp = BootstrapComparison(logger)
        result = comp.run_omnibus(bootstrap_organized_data)
        assert result == {}

    def test_pairwise_ci_computation(
        self,
        logger,
        bootstrap_organized_data,
        bootstrap_run_id_order,
        bootstrap_run_id_to_name,
    ):
        comp = BootstrapComparison(logger)
        results = comp.run_pairwise(
            bootstrap_organized_data,
            bootstrap_run_id_order,
            bootstrap_run_id_to_name,
        )
        assert len(results) == 1
        r = results[0]
        deltas = np.array([0.80 - 0.70, 0.82 - 0.69, 0.79 - 0.71, 0.81 - 0.68])
        assert np.isclose(r["ci_low"], np.percentile(deltas, 2.5))
        assert np.isclose(r["ci_high"], np.percentile(deltas, 97.5))
        assert np.isclose(r["statistic"], float(np.mean(deltas)))

    def test_pairwise_p_value_derivation(
        self,
        logger,
        bootstrap_organized_data,
        bootstrap_run_id_order,
        bootstrap_run_id_to_name,
    ):
        comp = BootstrapComparison(logger)
        results = comp.run_pairwise(
            bootstrap_organized_data,
            bootstrap_run_id_order,
            bootstrap_run_id_to_name,
        )
        r = results[0]
        deltas = np.array([0.80 - 0.70, 0.82 - 0.69, 0.79 - 0.71, 0.81 - 0.68])
        n_bs = len(deltas)
        frac_pos = float(np.sum(deltas > 0)) / n_bs
        frac_neg = float(np.sum(deltas < 0)) / n_bs
        expected_p = 2.0 * min(frac_pos, frac_neg)
        assert np.isclose(r["p_value"], expected_p)

    def test_pairwise_raises_on_missing_bootstrap_keys(self, logger):
        comp = BootstrapComparison(logger)
        non_bootstrap_data = {
            0: {"ep": {"mae": [0.8, 0.7]}},
            1: {"ep": {"mae": [0.82, 0.69]}},
        }
        with pytest.raises(ValueError, match="split.n_bootstrap"):
            comp.run_pairwise(
                non_bootstrap_data, ["A", "B"], {"A": "modelA", "B": "modelB"}
            )

    def test_pairwise_result_schema(
        self,
        logger,
        bootstrap_organized_data,
        bootstrap_run_id_order,
        bootstrap_run_id_to_name,
    ):
        comp = BootstrapComparison(logger)
        results = comp.run_pairwise(
            bootstrap_organized_data,
            bootstrap_run_id_order,
            bootstrap_run_id_to_name,
        )
        r = results[0]
        for field in (
            "endpoint",
            "metric",
            "run_pair",
            "run_i_name",
            "run_j_name",
            "run_i_id",
            "run_j_id",
            "statistic",
            "ci_low",
            "ci_high",
            "p_value",
            "n_observations",
            "test",
        ):
            assert field in r, f"Missing field: {field}"
        assert r["test"] == "bootstrap_percentile"
        assert r["n_observations"] == 4

    def test_correct_pairwise_applies_holm_bonferroni(
        self,
        logger,
        bootstrap_organized_data,
        bootstrap_run_id_order,
        bootstrap_run_id_to_name,
    ):
        comp = BootstrapComparison(logger)
        results = comp.run_pairwise(
            bootstrap_organized_data,
            bootstrap_run_id_order,
            bootstrap_run_id_to_name,
        )
        corrected = comp.correct_pairwise(results)
        for r in corrected:
            assert "p_value_corrected" in r
            assert "significant_05" in r
            assert "significant_01" in r
            assert r["p_value_corrected"] >= r["p_value"] - 1e-12

    def test_correct_pairwise_multiple_pairs(self, logger):
        # 3-model scenario: 3 pairs per endpoint-metric -> Holm-Bonferroni applied
        comp = BootstrapComparison(logger)
        raw_results = [
            {
                "endpoint": "ep",
                "metric": "mae",
                "run_pair": "A_vs_B",
                "run_i_name": "A",
                "run_j_name": "B",
                "run_i_id": "A",
                "run_j_id": "B",
                "statistic": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
                "p_value": 0.01,
                "n_observations": 100,
                "test": "bootstrap_percentile",
            },
            {
                "endpoint": "ep",
                "metric": "mae",
                "run_pair": "A_vs_C",
                "run_i_name": "A",
                "run_j_name": "C",
                "run_i_id": "A",
                "run_j_id": "C",
                "statistic": 0.05,
                "ci_low": 0.02,
                "ci_high": 0.08,
                "p_value": 0.04,
                "n_observations": 100,
                "test": "bootstrap_percentile",
            },
            {
                "endpoint": "ep",
                "metric": "mae",
                "run_pair": "B_vs_C",
                "run_i_name": "B",
                "run_j_name": "C",
                "run_i_id": "B",
                "run_j_id": "C",
                "statistic": 0.05,
                "ci_low": 0.01,
                "ci_high": 0.09,
                "p_value": 0.20,
                "n_observations": 100,
                "test": "bootstrap_percentile",
            },
        ]
        corrected = comp.correct_pairwise(raw_results)
        assert len(corrected) == 3
        expected = _holm_bonferroni([0.01, 0.04, 0.20])
        got = [r["p_value_corrected"] for r in corrected]
        assert np.allclose(got, expected)


# =========================================================================
# build_comparison factory
# =========================================================================


class TestBuildComparison:
    def test_parametric(self, logger):
        assert isinstance(build_comparison("parametric", logger), ParametricComparison)

    def test_non_parametric(self, logger):
        assert isinstance(
            build_comparison("non-parametric", logger), NonParametricComparison
        )

    def test_bootstrap(self, logger):
        assert isinstance(build_comparison("bootstrap", logger), BootstrapComparison)

    def test_invalid_mode_raises(self, logger):
        with pytest.raises(ValueError, match="bootstrap"):
            build_comparison("bayesian", logger)
