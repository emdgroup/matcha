"""Tests for matcha.utils.plotting – plot_regression, plot_classification, save_plot.

Validates the matcha-specific plotting helpers (figure creation, NaN
handling, trace counts, save to HTML).  Does *not* re-test Plotly internals.
"""

import os

import numpy as np

from matcha.utils.plotting import plot_classification, plot_regression, save_plot


# =========================================================================
# plot_regression
# =========================================================================


class TestPlotRegression:
    """Tests for plot_regression figure factory."""

    def test_returns_figure(self, regression_labels, regression_predictions):
        fig = plot_regression(regression_labels, regression_predictions)
        assert fig is not None
        # Should have at minimum: scatter, trendline, diagonal, 2-fold x2, 3-fold x2
        assert len(fig.data) >= 5

    def test_log10_mode_extra_traces(self, regression_labels, regression_predictions):
        # Shift to positive for log10
        labels = regression_labels - regression_labels.min() + 1
        preds = regression_predictions - regression_predictions.min() + 1
        fig = plot_regression(labels, preds, is_log10=True)
        assert fig is not None

    def test_custom_title(self, regression_labels, regression_predictions):
        fig = plot_regression(
            regression_labels, regression_predictions, plot_title="Custom Title"
        )
        assert "Custom Title" in fig.layout.title.text

    def test_with_labels(self, regression_labels, regression_predictions):
        labels_list = [f"mol_{i}" for i in range(len(regression_labels))]
        fig = plot_regression(
            regression_labels, regression_predictions, labels=labels_list
        )
        assert fig is not None

    def test_all_nan_returns_none(self):
        true_vals = np.array([np.nan, np.nan])
        pred_vals = np.array([np.nan, np.nan])
        fig = plot_regression(true_vals, pred_vals)
        assert fig is None

    def test_handles_2d_column_inputs(self):
        true_vals = np.array([[1.0], [2.0], [3.0]])
        pred_vals = np.array([[1.1], [2.1], [3.1]])
        fig = plot_regression(true_vals, pred_vals)
        assert fig is not None


# =========================================================================
# plot_classification
# =========================================================================


class TestPlotClassification:
    """Tests for plot_classification figure factory."""

    def test_returns_figure(
        self,
        classification_labels,
        classification_predictions,
        classification_probabilities,
    ):
        fig = plot_classification(
            classification_labels,
            classification_predictions,
            classification_probabilities,
        )
        assert fig is not None
        # ROC line, diagonal, PR line, baseline, 2 histograms = 6 traces
        assert len(fig.data) >= 6

    def test_all_nan_returns_none(self):
        nans = np.array([np.nan, np.nan])
        fig = plot_classification(nans, nans, nans)
        assert fig is None

    def test_custom_title(
        self,
        classification_labels,
        classification_predictions,
        classification_probabilities,
    ):
        fig = plot_classification(
            classification_labels,
            classification_predictions,
            classification_probabilities,
            plot_title="My Title",
        )
        assert "My Title" in fig.layout.title.text


# =========================================================================
# save_plot
# =========================================================================


class TestSavePlot:
    """Tests for save_plot HTML export."""

    def test_saves_html_file(self, tmp_path, regression_labels, regression_predictions):
        fig = plot_regression(regression_labels, regression_predictions)
        html_path = str(tmp_path / "plots" / "fig.html")
        save_plot(fig, html_path)
        assert os.path.isfile(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "<html>" in content.lower() or "plotly" in content.lower()
