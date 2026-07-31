"""CLI command for summarizing and comparing MLflow experiment results.

Loads performance artifacts from MLflow runs, runs statistical tests
(parametric or non-parametric) to compare models, generates interactive
plots, and saves a summary report as MLflow artifacts.
"""

import os
import re
import json
import argparse
import yaml
import mlflow
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.cli import CLISummarizeInputModel
from matcha.cli.statistical_tests import build_comparison


class ExperimentSummarizer(ABC):
    """Abstract base class for experiment summarizers.

    Holds all shared analysis logic (artifact loading, statistical plots,
    summary statistics). Subclasses implement run discovery
    (:meth:`find_experiment_runs`) and result persistence
    (:meth:`save_summary_to_backend`).
    """

    def __init__(self, logger, statistical_test: str = "non-parametric"):
        """Initialize the summarizer.

        :param logger: Logger instance for status messages.
        :param statistical_test: Testing mode — ``"parametric"`` for
            repeated-measures ANOVA + Tukey HSD; ``"non-parametric"`` for
            Friedman + Wilcoxon with BH correction; or ``"bootstrap"`` for
            bootstrap percentile CIs + Holm-Bonferroni (requires
            ``split.n_bootstrap > 1`` in the evaluate config).
        """
        self.logger = logger
        self.comparison = build_comparison(statistical_test, logger)
        self.statistical_test = statistical_test
        self.logger.info(f"Statistical testing mode: {self.statistical_test}")
        self.source_identifier_key: str = ""
        self.source_identifier: str = ""

    @abstractmethod
    def find_experiment_runs(
        self, selected_runs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover runs and return a list of run-info dicts.

        Each dict must contain ``run_id``, ``run_name``, and
        ``artifact_path``.
        """
        ...

    @abstractmethod
    def save_summary_to_backend(
        self, summary_data: Dict, plot_files: Dict[str, str] = None
    ):
        """Persist the summary JSON and plot files to the appropriate backend."""
        ...

    def load_performance_artifacts(
        self, run_info_list: List[Dict[str, Any]]
    ) -> Tuple[Dict, Dict, int, Dict]:
        """Load performance.json and performance_log10.json from all runs."""
        performance_data = {}
        performance_log10_data = {}
        run_id_to_name = {}

        for run_info in run_info_list:
            run_id = run_info["run_id"]
            run_name = run_info["run_name"]
            artifact_path = run_info["artifact_path"]

            # Store run name mapping
            run_id_to_name[run_id] = run_name

            # Load performance.json
            perf_file = os.path.join(artifact_path, "performance.json")
            perf_log10_file = os.path.join(artifact_path, "performance_log10.json")

            if os.path.exists(perf_file):
                try:
                    with open(perf_file, "r") as f:
                        performance_data[run_id] = json.load(f)
                    self.logger.info(f"Loaded performance.json for run {run_id}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load performance.json for run {run_id}: {e}"
                    )
            else:
                self.logger.warning(f"performance.json not found for run {run_id}")

            if os.path.exists(perf_log10_file):
                try:
                    with open(perf_log10_file, "r") as f:
                        performance_log10_data[run_id] = json.load(f)
                    self.logger.info(f"Loaded performance_log10.json for run {run_id}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load performance_log10.json for run {run_id}: {e}"
                    )
            else:
                self.logger.warning(
                    f"performance_log10.json not found for run {run_id}"
                )

        num_experiments = len(performance_data)
        self.logger.info(
            f"Successfully loaded performance data from {num_experiments} experiments"
        )

        return performance_data, performance_log10_data, num_experiments, run_id_to_name

    def load_predictions_artifacts(
        self, run_info_list: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load prediction CSV files from all runs.

        :param run_info_list: List of run info dicts from
            :meth:`find_experiment_runs`.
        :returns: Nested dict ``{run_id: {endpoint_split: DataFrame}}``.
        """
        predictions_data = {}

        for run_info in run_info_list:
            run_id = run_info["run_id"]
            artifact_path = run_info["artifact_path"]

            # Find all prediction CSV files in the artifact path
            pred_files = {}
            if os.path.exists(artifact_path):
                for file in os.listdir(artifact_path):
                    if file.endswith("_preds.csv"):
                        # Parse filename to extract endpoint and split
                        # Format: {endpoint}_{split_id}_preds.csv
                        parts = file.replace("_preds.csv", "").rsplit("_", 1)
                        if len(parts) == 2:
                            endpoint, split_id = parts
                            key = f"{endpoint}_{split_id}"

                            try:
                                csv_path = os.path.join(artifact_path, file)
                                df = pd.read_csv(csv_path)
                                pred_files[key] = df
                                self.logger.debug(f"Loaded {file} for run {run_id}")
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to load {file} for run {run_id}: {e}"
                                )

            if pred_files:
                predictions_data[run_id] = pred_files
                self.logger.info(
                    f"Loaded {len(pred_files)} prediction files for run {run_id}"
                )

        self.logger.info(
            f"Successfully loaded predictions from {len(predictions_data)} runs"
        )
        return predictions_data

    def extract_split_metrics(
        self, performance_data: Dict
    ) -> Tuple[Dict[str, Dict[str, Dict[str, List]]], List[str]]:
        """Extract metrics organized by split, endpoint, and metric type.

        :param performance_data: Per-run performance dict from
            :meth:`load_performance_artifacts`.
        :returns: Tuple of ``(organized_data, run_id_order)`` where
            *organized_data* maps
            ``{split_id: {endpoint: {metric: [values_across_runs]}}}``
            and *run_id_order* is a list of run IDs.
        """
        organized_data = {}
        run_id_order = list(performance_data.keys())

        for run_id, run_data in performance_data.items():
            for split_id, split_data in run_data.items():
                if split_id in ["mean", "std"]:
                    continue  # Skip aggregated data

                if split_id not in organized_data:
                    organized_data[split_id] = {}

                for endpoint, metrics in split_data.items():
                    if endpoint not in organized_data[split_id]:
                        organized_data[split_id][endpoint] = {}

                    for metric_name, metric_value in metrics.items():
                        if metric_name not in organized_data[split_id][endpoint]:
                            organized_data[split_id][endpoint][metric_name] = []

                        organized_data[split_id][endpoint][metric_name].append(
                            metric_value
                        )

        return organized_data, run_id_order

    def create_performance_plots(
        self,
        performance_data: Dict,
        performance_log10_data: Dict,
        run_id_to_name: Dict,
        corrected_results: List[Dict],
        corrected_log10_results: List[Dict] | None,
        omnibus_results: Optional[Dict[Tuple[str, str], Dict]] = None,
        omnibus_log10_results: Optional[Dict[Tuple[str, str], Dict]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create scatter plots for each metric-endpoint combination.

        Shows mean performance across runs with standard deviation error
        bars, for both unscaled and log10-scaled data. Points are
        color-coded based on statistical significance compared to the best
        model: green for best, light green for non-significant, light blue
        for significantly different.

        :param output_dir: Directory to write HTML files. If ``None``, files
            are written to the current working directory.
        :returns: Dict mapping plot names to HTML file paths.
        """
        plot_files = {}

        # Define colors
        best_color = "#149b5f"
        non_significant_color = "#a5cd50"
        significant_color = "#96d7d2"

        # Process both datasets
        datasets = [
            ("unscaled", performance_data, corrected_results, omnibus_results),
            (
                "log10",
                performance_log10_data,
                corrected_log10_results,
                omnibus_log10_results,
            ),
        ]

        for scale_type, data, comparison_results, scale_omnibus_results in datasets:
            if not data:
                self.logger.warning(f"No {scale_type} performance data found")
                continue

            # Extract mean and std data from performance_data
            mean_data = {}
            std_data = {}

            for run_id, run_data in data.items():
                if "mean" in run_data and "std" in run_data:
                    mean_data[run_id] = run_data["mean"]
                    std_data[run_id] = run_data["std"]

            if not mean_data:
                self.logger.warning(
                    f"No mean/std data found in {scale_type} performance data"
                )
                continue

            # Get all unique endpoint-metric combinations
            all_combinations = set()
            for run_data in mean_data.values():
                for endpoint, metrics in run_data.items():
                    for metric_name in metrics.keys():
                        all_combinations.add((endpoint, metric_name))

            # Create plots for each combination
            for endpoint, metric_name in all_combinations:
                try:
                    # Extract data for this combination
                    run_names = []
                    mean_values = []
                    std_values = []
                    run_ids_ordered = []

                    for run_id in mean_data.keys():
                        if (
                            endpoint in mean_data[run_id]
                            and metric_name in mean_data[run_id][endpoint]
                            and endpoint in std_data[run_id]
                            and metric_name in std_data[run_id][endpoint]
                        ):
                            run_name = run_id_to_name.get(run_id, f"run_{run_id[:8]}")
                            run_names.append(run_name)
                            run_ids_ordered.append(run_id)
                            mean_values.append(mean_data[run_id][endpoint][metric_name])
                            std_values.append(std_data[run_id][endpoint][metric_name])

                    if not run_names:
                        continue

                    # Determine best model based on metric type
                    # Higher is better for R2, Within2Fold, Within3Fold, Spearman
                    # Lower is better for MAE, RMSE
                    higher_is_better_metrics = [
                        # reg
                        "r2",
                        "within2fold",
                        "within3fold",
                        "spearman",
                        # clf
                        "balanced_accuracy",
                        "cohen_kappa",
                        "matthews_corrcoef",
                        "f1_score",
                        "precision",
                        "recall",
                        "ef_10",
                        "roc_auc",
                        "pr_auc",
                    ]
                    is_higher_better = any(
                        metric.lower() in metric_name.lower()
                        for metric in higher_is_better_metrics
                    )

                    if is_higher_better:
                        best_idx = np.argmax(mean_values)
                    else:
                        best_idx = np.argmin(mean_values)

                    best_run_id = run_ids_ordered[best_idx]
                    best_run_name = run_names[best_idx]

                    # Find statistical comparisons for this endpoint-metric combination
                    relevant_comparisons = [
                        comp
                        for comp in comparison_results
                        if comp["endpoint"] == endpoint
                        and comp["metric"] == metric_name
                    ]

                    # Determine colors for each run
                    colors = []
                    legend_colors = []
                    for i, (run_id, run_name) in enumerate(
                        zip(run_ids_ordered, run_names)
                    ):
                        if run_id == best_run_id:
                            colors.append(best_color)
                            legend_colors.append("Best model")
                        else:
                            # Check if this run is significantly different from the best
                            is_significant = False
                            for comp in relevant_comparisons:
                                if (
                                    comp["run_i_id"] == best_run_id
                                    and comp["run_j_id"] == run_id
                                ) or (
                                    comp["run_j_id"] == best_run_id
                                    and comp["run_i_id"] == run_id
                                ):
                                    if comp[
                                        "significant_05"
                                    ]:  # Using 0.05 significance level
                                        is_significant = True
                                        break

                            if is_significant:
                                colors.append(significant_color)
                                legend_colors.append("Significantly different")
                            else:
                                colors.append(non_significant_color)
                                legend_colors.append("Not significantly different")

                    # Create the scatter plot with color coding
                    fig = go.Figure()

                    # Group points by color for legend
                    color_groups = {}
                    for i, (color, legend_label) in enumerate(
                        zip(colors, legend_colors)
                    ):
                        if legend_label not in color_groups:
                            color_groups[legend_label] = {"indices": [], "color": color}
                        color_groups[legend_label]["indices"].append(i)

                    # Add traces for each color group
                    for legend_label, group_info in color_groups.items():
                        indices = group_info["indices"]
                        color = group_info["color"]

                        # Extract data for this color group
                        group_x = [mean_values[i] for i in indices]
                        group_y = [run_names[i] for i in indices]
                        group_std = [std_values[i] for i in indices]

                        fig.add_trace(
                            go.Scatter(
                                x=group_x,
                                y=group_y,
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=color,
                                    line=dict(width=2, color=color),
                                ),
                                error_x=dict(
                                    type="data",
                                    array=group_std,
                                    visible=True,
                                    color=color,
                                    thickness=2,
                                    width=4,
                                ),
                                name=legend_label,
                                showlegend=True,
                            )
                        )

                    # Update layout
                    plot_title = f"{metric_name}_{endpoint}_{scale_type}"

                    # Add omnibus test p-value to title if available
                    title_text = plot_title
                    if (
                        scale_omnibus_results
                        and (endpoint, metric_name) in scale_omnibus_results
                    ):
                        omnibus_info = scale_omnibus_results[(endpoint, metric_name)]
                        omnibus_test_name = omnibus_info["test"]
                        omnibus_p = omnibus_info["p_value_corrected"]
                        title_text = f"{plot_title}<br>{omnibus_test_name} p={omnibus_p:.4g} (Bonferroni corrected)"

                    x_axis_title = f"{metric_name}" + (
                        " (log10)" if scale_type == "log10" else ""
                    )

                    fig.update_layout(
                        title=dict(
                            text=title_text,
                            x=0.5,  # Center the title
                            xanchor="center",
                            font=dict(size=16, color="black"),
                        ),
                        xaxis_title=x_axis_title,
                        yaxis_title="Run Names",
                        xaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=1),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="lightgray",
                            gridwidth=1,
                            categoryorder="array",
                            categoryarray=run_names,
                        ),
                        plot_bgcolor="white",
                        width=800,
                        height=max(400, len(run_names) * 50 + 100),
                        margin=dict(l=200, r=50, t=80, b=80),
                        legend=dict(
                            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
                        ),
                    )

                    # Save as HTML
                    plot_filename = f"{plot_title.replace(' ', '_')}_plot.html"
                    full_path = (
                        os.path.join(output_dir, plot_filename)
                        if output_dir
                        else plot_filename
                    )
                    fig.write_html(full_path)
                    plot_files[plot_title] = full_path

                    self.logger.info(
                        f"Created {scale_type} plot for {endpoint}-{metric_name} with best model: {best_run_name}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create {scale_type} plot for {endpoint}-{metric_name}: {e}"
                    )

        return plot_files

    def create_correlation_heatmaps(
        self,
        predictions_data: Dict[str, Dict[str, pd.DataFrame]],
        run_id_to_name: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create correlation heatmaps between predictions and residuals across methods.

        Aggregates predictions across all splits for each endpoint. Creates
        two heatmaps per endpoint: one for prediction correlation and one
        for residual correlation.

        :param predictions_data: Per-run prediction DataFrames from
            :meth:`load_predictions_artifacts`.
        :param run_id_to_name: Mapping from run IDs to display names.
        :param output_dir: Directory to write HTML files. If ``None``, files
            are written to the current working directory.
        :returns: Dict mapping plot names to HTML file paths.
        """
        plot_files = {}

        if not predictions_data:
            self.logger.warning(
                "No predictions data available for correlation analysis"
            )
            return plot_files

        # Aggregate data by endpoint (across all splits)
        endpoint_aggregated = {}

        for run_id, run_data in predictions_data.items():
            for endpoint_split, df in run_data.items():
                # Extract endpoint name (format: {endpoint}_{split_id})
                parts = endpoint_split.rsplit("_", 1)
                if len(parts) == 2:
                    endpoint = parts[0]
                else:
                    endpoint = endpoint_split

                if endpoint not in endpoint_aggregated:
                    endpoint_aggregated[endpoint] = {}

                if run_id not in endpoint_aggregated[endpoint]:
                    endpoint_aggregated[endpoint][run_id] = {
                        "predictions": [],
                        "true_values": [],
                        "residuals": [],
                    }

                # Collect predictions and true values
                if "pred" in df.columns and "true" in df.columns:
                    preds = df["pred"].values
                    trues = df["true"].values

                    # Create mask to filter out NaN values in true labels
                    valid_mask = ~np.isnan(trues)
                    preds_filtered = preds[valid_mask]
                    trues_filtered = trues[valid_mask]

                    endpoint_aggregated[endpoint][run_id]["predictions"].extend(
                        preds_filtered
                    )
                    endpoint_aggregated[endpoint][run_id]["true_values"].extend(
                        trues_filtered
                    )
                    endpoint_aggregated[endpoint][run_id]["residuals"].extend(
                        trues_filtered - preds_filtered
                    )
                elif "pred_log10" in df.columns and "true_log10" in df.columns:
                    preds = df["pred_log10"].values
                    trues = df["true_log10"].values

                    # Create mask to filter out NaN values in true labels
                    valid_mask = ~np.isnan(trues)
                    preds_filtered = preds[valid_mask]
                    trues_filtered = trues[valid_mask]

                    endpoint_aggregated[endpoint][run_id]["predictions"].extend(
                        preds_filtered
                    )
                    endpoint_aggregated[endpoint][run_id]["true_values"].extend(
                        trues_filtered
                    )
                    endpoint_aggregated[endpoint][run_id]["residuals"].extend(
                        trues_filtered - preds_filtered
                    )
                else:
                    self.logger.warning(
                        f"No valid prediction/true columns found in {endpoint_split} for run {run_id}"
                    )

        # Convert lists to numpy arrays
        for endpoint, runs_data in endpoint_aggregated.items():
            for run_id, data in runs_data.items():
                data["predictions"] = np.array(data["predictions"])
                data["true_values"] = np.array(data["true_values"])
                data["residuals"] = np.array(data["residuals"])

        # Create heatmaps for each endpoint
        for endpoint, runs_data in endpoint_aggregated.items():
            if len(runs_data) < 2:
                self.logger.warning(
                    f"Not enough runs ({len(runs_data)}) for correlation analysis on {endpoint}"
                )
                continue

            try:
                run_ids = list(runs_data.keys())
                run_names = [
                    run_id_to_name.get(run_id, f"run_{run_id[:8]}")
                    for run_id in run_ids
                ]
                n_runs = len(run_ids)

                # Create prediction correlation matrix
                pred_correlation_matrix = np.zeros((n_runs, n_runs))

                for i in range(n_runs):
                    for j in range(n_runs):
                        if i == j:
                            pred_correlation_matrix[i, j] = 1.0
                        else:
                            # Calculate Pearson correlation for predictions
                            pred_i = runs_data[run_ids[i]]["predictions"]
                            pred_j = runs_data[run_ids[j]]["predictions"]

                            # Ensure same length
                            min_len = min(len(pred_i), len(pred_j))
                            pred_i = pred_i[:min_len]
                            pred_j = pred_j[:min_len]

                            try:
                                corr, _ = stats.pearsonr(pred_i, pred_j)
                                pred_correlation_matrix[i, j] = corr
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to compute prediction correlation for runs {i},{j}: {e}"
                                )
                                pred_correlation_matrix[i, j] = np.nan

                # Create prediction correlation heatmap
                fig_pred = go.Figure(
                    data=go.Heatmap(
                        z=pred_correlation_matrix,
                        x=run_names,
                        y=run_names,
                        colorscale=[
                            [0.0, "#f0f0f0"],  # Light gray for low correlation
                            [0.5, "#a5cd50"],  # Light green for medium correlation
                            [0.75, "#7eb83a"],  # Medium green
                            [1.0, "#149b5f"],  # Dark green for high correlation
                        ],
                        text=np.round(pred_correlation_matrix, 3),
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorbar=dict(
                            title="Correlation", tickmode="linear", tick0=0, dtick=0.2
                        ),
                        zmin=0,
                        zmax=1,
                        hoverongaps=False,
                        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
                    )
                )

                plot_title_pred = f"prediction_correlation_heatmap_{endpoint}"

                fig_pred.update_layout(
                    title=dict(
                        text=f"Prediction Correlation - {endpoint}",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16, color="black"),
                    ),
                    xaxis_title="Run Names",
                    yaxis_title="Run Names",
                    xaxis=dict(tickangle=-45, side="bottom"),
                    yaxis=dict(autorange="reversed"),
                    width=max(600, n_runs * 80),
                    height=max(600, n_runs * 80),
                    margin=dict(l=150, r=150, t=100, b=150),
                )

                # Save prediction correlation heatmap
                plot_filename_pred = f"{plot_title_pred.replace(' ', '_')}_plot.html"
                full_path_pred = (
                    os.path.join(output_dir, plot_filename_pred)
                    if output_dir
                    else plot_filename_pred
                )
                fig_pred.write_html(full_path_pred)
                plot_files[plot_title_pred] = full_path_pred

                self.logger.info(
                    f"Created prediction correlation heatmap for {endpoint} with {n_runs} runs"
                )

                # Create residual correlation matrix
                res_correlation_matrix = np.zeros((n_runs, n_runs))

                for i in range(n_runs):
                    for j in range(n_runs):
                        if i == j:
                            res_correlation_matrix[i, j] = 1.0
                        else:
                            # Calculate Pearson correlation for residuals
                            res_i = runs_data[run_ids[i]]["residuals"]
                            res_j = runs_data[run_ids[j]]["residuals"]

                            # Ensure same length
                            min_len = min(len(res_i), len(res_j))
                            res_i = res_i[:min_len]
                            res_j = res_j[:min_len]

                            try:
                                corr, _ = stats.pearsonr(res_i, res_j)
                                res_correlation_matrix[i, j] = corr
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to compute residual correlation for runs {i},{j}: {e}"
                                )
                                res_correlation_matrix[i, j] = np.nan

                # Create residual correlation heatmap
                fig_res = go.Figure(
                    data=go.Heatmap(
                        z=res_correlation_matrix,
                        x=run_names,
                        y=run_names,
                        colorscale=[
                            [0.0, "#f0f0f0"],  # Light gray for low correlation
                            [0.5, "#a5cd50"],  # Light green for medium correlation
                            [0.75, "#7eb83a"],  # Medium green
                            [1.0, "#149b5f"],  # Dark green for high correlation
                        ],
                        text=np.round(res_correlation_matrix, 3),
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorbar=dict(
                            title="Correlation", tickmode="linear", tick0=0, dtick=0.2
                        ),
                        zmin=-1,  # Residuals can have negative correlation
                        zmax=1,
                        hoverongaps=False,
                        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
                    )
                )

                plot_title_res = f"residual_correlation_heatmap_{endpoint}"

                fig_res.update_layout(
                    title=dict(
                        text=f"Residual Correlation - {endpoint}",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16, color="black"),
                    ),
                    xaxis_title="Run Names",
                    yaxis_title="Run Names",
                    xaxis=dict(tickangle=-45, side="bottom"),
                    yaxis=dict(autorange="reversed"),
                    width=max(600, n_runs * 80),
                    height=max(600, n_runs * 80),
                    margin=dict(l=150, r=150, t=100, b=150),
                )

                # Save residual correlation heatmap
                plot_filename_res = f"{plot_title_res.replace(' ', '_')}_plot.html"
                full_path_res = (
                    os.path.join(output_dir, plot_filename_res)
                    if output_dir
                    else plot_filename_res
                )
                fig_res.write_html(full_path_res)
                plot_files[plot_title_res] = full_path_res

                self.logger.info(
                    f"Created residual correlation heatmap for {endpoint} with {n_runs} runs"
                )

            except Exception as e:
                self.logger.warning(
                    f"Failed to create correlation heatmaps for {endpoint}: {e}"
                )

        return plot_files

    def create_summary_statistics(
        self,
        performance_data: Dict,
        num_experiments: int,
        comparison_results: List[Dict],
        omnibus_results: Optional[Dict[Tuple[str, str], Dict]] = None,
    ) -> Dict:
        """Create comprehensive summary statistics."""

        # Count significant comparisons
        sig_05_count = sum(1 for r in comparison_results if r["significant_05"])
        sig_01_count = sum(1 for r in comparison_results if r["significant_01"])

        # Count unique endpoint-metric combinations for correction info
        unique_combinations = set(
            (r["endpoint"], r["metric"]) for r in comparison_results
        )

        summary = {
            self.source_identifier_key: self.source_identifier,
            "num_experiments": num_experiments,
            "statistical_test": self.statistical_test,
            "omnibus_method": self.comparison.omnibus_method,
            "pairwise_method": self.comparison.pairwise_method,
            "total_comparisons": len(comparison_results),
            "significant_comparisons_05": sig_05_count,
            "significant_comparisons_01": sig_01_count,
            "correction_method": self.comparison.correction_method,
            "correction_groups": f"Applied separately for each endpoint-metric combination ({len(unique_combinations)} groups)",
            "pairwise_comparisons": comparison_results,
        }

        # Add omnibus test results if available
        if omnibus_results:
            serializable_omnibus = {}
            for (endpoint, metric), result in omnibus_results.items():
                key = f"{endpoint}__{metric}"
                serializable_omnibus[key] = result
            summary["omnibus_tests"] = serializable_omnibus

        # Add basic descriptive statistics
        if performance_data:
            first_run = list(performance_data.keys())[0]
            splits = [
                k
                for k in performance_data[first_run].keys()
                if k not in ["mean", "std"]
            ]
            endpoints = []
            metrics = []

            for split_data in performance_data[first_run].values():
                if isinstance(split_data, dict):
                    endpoints.extend(split_data.keys())
                    for endpoint_data in split_data.values():
                        if isinstance(endpoint_data, dict):
                            metrics.extend(endpoint_data.keys())

            summary["metadata"] = {
                "splits": list(set(splits)),
                "endpoints": list(set(endpoints)),
                "metrics": list(set(metrics)),
                "num_splits": len(set(splits)),
                "num_endpoints": len(set(endpoints)),
                "num_metrics": len(set(metrics)),
            }

        return summary


class MLflowExperimentSummarizer(ExperimentSummarizer):
    """Summarizes MLflow experiments with statistical analysis.

    Supports three statistical testing modes:
    - "parametric": repeated-measures ANOVA (omnibus) + Tukey HSD against the
      RM-ANOVA error term (pairwise)
    - "non-parametric": Friedman chi-squared (omnibus) + Wilcoxon signed-rank + BH correction (pairwise)
    - "bootstrap": no omnibus test; bootstrap percentile CIs + Holm-Bonferroni
      pairwise comparisons (requires ``split.n_bootstrap > 1`` in evaluate config)

    The statistical tests themselves live in
    :mod:`matcha.cli.statistical_tests`; this class delegates omnibus,
    pairwise, and correction work to a comparison strategy selected by
    *statistical_test*.
    """

    def __init__(
        self,
        experiment_name: str,
        mlruns_path: str,
        logger,
        statistical_test: str = "non-parametric",
    ):
        """Initialize the summarizer.

        :param experiment_name: Name of the MLflow experiment to summarize.
        :param mlruns_path: Path to the ``mlruns`` directory.
        :param logger: Logger instance for status messages.
        :param statistical_test: Testing mode — ``"parametric"`` for
            repeated-measures ANOVA + Tukey HSD; ``"non-parametric"`` for
            Friedman + Wilcoxon with BH correction; or ``"bootstrap"`` for
            bootstrap percentile CIs + Holm-Bonferroni (requires
            ``split.n_bootstrap > 1`` in the evaluate config).
        :raises ValueError: If *statistical_test* is not one of the
            supported modes.
        """
        super().__init__(logger, statistical_test)
        self.experiment_name = experiment_name
        self.mlruns_path = mlruns_path
        self.source_identifier_key = "experiment_name"
        self.source_identifier = experiment_name

    def find_experiment_runs(
        self, selected_runs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find all runs in the specified experiment and their artifact paths.

        :param selected_runs: Optional list of run names to select. If
            ``None``, all runs are returned. If provided, only runs whose
            names match an entry in this list are included.
        :returns: List of dicts with ``run_id``, ``run_name``, and
            ``artifact_path`` for each qualifying run.
        """
        try:
            # Set MLflow tracking URI to local mlruns directory
            mlflow.set_tracking_uri(f"file://{os.path.abspath(self.mlruns_path)}")

            # Get experiment by name
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{self.experiment_name}' not found")

            # Get all runs in the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            run_info = []
            for _, run in runs.iterrows():
                run_id = run["run_id"]
                run_name = run.get("tags.mlflow.runName", f"run_{run_id[:8]}")

                # Skip summary runs
                if "summary-analysis" in run_name.lower():
                    continue

                # Only process runs with "_metrics" in the name
                if "_metrics" not in run_name.lower():
                    continue

                # Find artifact path for this run
                artifact_path = os.path.join(
                    self.mlruns_path, experiment.experiment_id, run_id, "artifacts"
                )

                run_info.append(
                    {
                        "run_id": run_id,
                        "run_name": run_name,
                        "artifact_path": artifact_path,
                    }
                )

            self.logger.info(
                f"Found {len(run_info)} runs in experiment '{self.experiment_name}'"
            )

            # Filter to specific runs if requested
            if selected_runs is not None:
                run_info = [r for r in run_info if r["run_name"] in selected_runs]
                not_found = set(selected_runs) - {r["run_name"] for r in run_info}
                if not_found:
                    self.logger.warning(f"Requested runs not found: {not_found}")
                self.logger.info(
                    f"Selected {len(run_info)} runs after filtering by run names"
                )

            return run_info

        except Exception as e:
            self.logger.error(f"Error finding experiment runs: {e}")
            raise

    def save_summary_to_backend(
        self, summary_data: Dict, plot_files: Dict[str, str] = None
    ):
        """Save summary analysis as a new MLflow run."""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(f"file://{os.path.abspath(self.mlruns_path)}")

            # Get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            with mlflow.start_run(
                experiment_id=experiment.experiment_id, run_name="summary-analysis"
            ) as run:
                # Log summary metrics
                mlflow.log_metric("num_experiments", summary_data["num_experiments"])
                mlflow.log_metric(
                    "total_comparisons", summary_data["total_comparisons"]
                )
                mlflow.log_metric(
                    "significant_comparisons_05",
                    summary_data["significant_comparisons_05"],
                )
                mlflow.log_metric(
                    "significant_comparisons_01",
                    summary_data["significant_comparisons_01"],
                )

                if "metadata" in summary_data:
                    mlflow.log_metric(
                        "num_splits", summary_data["metadata"]["num_splits"]
                    )
                    mlflow.log_metric(
                        "num_endpoints", summary_data["metadata"]["num_endpoints"]
                    )
                    mlflow.log_metric(
                        "num_metrics", summary_data["metadata"]["num_metrics"]
                    )

                # Log summary as artifact
                summary_path = "summary_analysis.json"
                with open(summary_path, "w") as f:
                    json.dump(summary_data, f, indent=2)

                mlflow.log_artifact(summary_path)

                # Log plot files as artifacts
                if plot_files:
                    for plot_name, plot_path in plot_files.items():
                        if os.path.exists(plot_path):
                            mlflow.log_artifact(plot_path)
                            self.logger.info(f"Logged plot artifact: {plot_name}")

                # Clean up temporary files
                if os.path.exists(summary_path):
                    os.remove(summary_path)

                if plot_files:
                    for plot_path in plot_files.values():
                        if os.path.exists(plot_path):
                            os.remove(plot_path)

                self.logger.info(
                    f"Summary analysis saved to MLflow run: {run.info.run_id}"
                )

        except Exception as e:
            self.logger.error(f"Error saving summary to MLflow: {e}")
            raise


class DirectoryExperimentSummarizer(ExperimentSummarizer):
    """Summarizes model runs stored as subdirectories on disk.

    Each immediate subdirectory of *root_dir* is treated as one run.
    ``performance.json``, ``performance_log10.json``, and ``*_preds.csv`` are
    loaded directly from each subdirectory. Results are written to
    *output_path* on disk.
    """

    def __init__(
        self,
        root_dir: str,
        output_path: str,
        logger,
        statistical_test: str = "non-parametric",
    ):
        """Initialize the directory summarizer.

        :param root_dir: Root directory whose immediate subdirectories are
            treated as individual model runs.
        :param output_path: Directory where the summary JSON and plots are
            written.
        :param logger: Logger instance for status messages.
        :param statistical_test: Testing mode — ``"parametric"``,
            ``"non-parametric"``, or ``"bootstrap"``.
        """
        super().__init__(logger, statistical_test)
        self.root_dir = os.path.abspath(root_dir)
        self.output_path = os.path.abspath(output_path)
        self.source_identifier_key = "root_dir"
        self.source_identifier = self.root_dir
        os.makedirs(self.output_path, exist_ok=True)

    def find_experiment_runs(
        self, selected_runs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover runs as subdirectories of *root_dir*.

        Each subdirectory that contains a ``performance.json`` file is
        included. Subdirectories without it are skipped with a warning.

        :param selected_runs: Optional list of subdirectory names to include.
            If ``None``, all qualifying subdirectories are returned.
        :returns: List of dicts with ``run_id``, ``run_name``, and
            ``artifact_path`` for each qualifying subdirectory.
        """
        subdirs = sorted(
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

        run_info = []
        for d in subdirs:
            subdir_path = os.path.join(self.root_dir, d)
            if not os.path.exists(os.path.join(subdir_path, "performance.json")):
                self.logger.warning(
                    f"Skipping '{d}': performance.json not found in {subdir_path}"
                )
                continue
            run_info.append({"run_id": d, "run_name": d, "artifact_path": subdir_path})

        self.logger.info(
            f"Found {len(run_info)} valid runs in directory '{self.root_dir}'"
        )

        if selected_runs is not None:
            run_info = [r for r in run_info if r["run_name"] in selected_runs]
            not_found = set(selected_runs) - {r["run_name"] for r in run_info}
            if not_found:
                self.logger.warning(f"Requested runs not found: {not_found}")
            self.logger.info(
                f"Selected {len(run_info)} runs after filtering by run names"
            )

        return run_info

    def save_summary_to_backend(
        self, summary_data: Dict, plot_files: Dict[str, str] = None
    ):
        """Write summary JSON to *output_path*.

        Plots are already written to *output_path* by the plot methods via
        ``output_dir``; this method only persists the JSON and logs paths.
        """
        os.makedirs(self.output_path, exist_ok=True)
        summary_path = os.path.join(self.output_path, "summary_analysis.json")
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        self.logger.info(f"Summary analysis written to: {summary_path}")

        if plot_files:
            for plot_name, plot_path in plot_files.items():
                self.logger.info(f"Plot written: {plot_name} → {plot_path}")


def main(cfg=None):
    """Run experiment summarization from a YAML configuration.

    Supports two modes selected by config fields:

    - **MLflow mode** (``experiment_name`` + ``mlruns_path``): loads artifacts
      from an MLflow tracking store, saves summary as MLflow artifacts.
    - **Directory mode** (``root_dir``): treats each immediate subdirectory as
      a run, reads artifacts directly from disk, writes results to
      ``output_path``.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLISummarizeInputModel` instance
        or a raw dict that will be validated.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Summarize MLflow experiments")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        CLISummarizeInputModel.model_validate(config)
    elif isinstance(cfg, CLISummarizeInputModel):
        config = cfg.model_dump()
    else:
        config = cfg
        CLISummarizeInputModel.model_validate(config)

    experiment_name = config.get("experiment_name")
    mlruns_path = config.get("mlruns_path", "./mlruns")
    root_dir = config.get("root_dir")
    output_path = config.get("output_path", "./outputs")
    statistical_test = config.get("statistical_test", "non-parametric")
    runs = config.get("runs", None)

    # Setup logging
    logger = get_default_logger("SUMMARIZE")

    try:
        # Factory: select subclass based on config mode
        if experiment_name:
            logger.info(
                f"Starting MLflow experiment summarization for: {experiment_name}"
            )
            logger.info(f"MLruns path: {os.path.abspath(mlruns_path)}")
            summarizer = MLflowExperimentSummarizer(
                experiment_name, mlruns_path, logger, statistical_test=statistical_test
            )
            output_dir = None
        else:
            logger.info(f"Starting directory-based summarization for: {root_dir}")
            logger.info(f"Output path: {os.path.abspath(output_path)}")
            summarizer = DirectoryExperimentSummarizer(
                root_dir, output_path, logger, statistical_test=statistical_test
            )
            output_dir = output_path

        # Find all runs in the experiment
        run_info_list = summarizer.find_experiment_runs(selected_runs=runs)

        if not run_info_list:
            logger.warning("No runs found in the experiment")
            return

        # Load performance artifacts
        performance_data, performance_log10_data, num_experiments, run_id_to_name = (
            summarizer.load_performance_artifacts(run_info_list)
        )

        if not performance_data:
            logger.warning("No performance data found")
            return

        # Organize data for statistical analysis
        organized_data, run_id_order = summarizer.extract_split_metrics(
            performance_data
        )
        organized_log10_data, run_id_order_log10 = summarizer.extract_split_metrics(
            performance_log10_data
        )

        has_bootstrap = any(re.match(r"^\d+_\d+$", str(k)) for k in organized_data)
        if has_bootstrap and summarizer.statistical_test != "bootstrap":
            logger.warning(
                "Bootstrap split keys detected in the data but statistical_test is "
                f"'{summarizer.statistical_test}'. Use statistical_test='bootstrap' "
                "to run bootstrap pairwise comparisons."
            )

        # Run pairwise comparisons on both datasets
        logger.info("Running statistical comparisons on performance data...")

        # Run omnibus tests (RM-ANOVA or Friedman with Bonferroni correction)
        logger.info("Running omnibus tests on performance data...")
        omnibus_results = summarizer.comparison.run_omnibus(organized_data)

        comparison_results = summarizer.comparison.run_pairwise(
            organized_data, run_id_order, run_id_to_name
        )

        # Apply pairwise correction (BH for non-parametric; no-op for
        # parametric, where Tukey's studentized range self-corrects).
        corrected_results = summarizer.comparison.correct_pairwise(comparison_results)

        # Create comprehensive summary
        summary_data = summarizer.create_summary_statistics(
            performance_data, num_experiments, corrected_results, omnibus_results
        )

        omnibus_log10_results = None
        try:
            logger.info("Running statistical comparisons on log10 performance data...")

            # Run omnibus tests on log10 data
            logger.info("Running omnibus tests on log10 performance data...")
            omnibus_log10_results = summarizer.comparison.run_omnibus(
                organized_log10_data
            )

            comparison_log10_results = summarizer.comparison.run_pairwise(
                organized_log10_data, run_id_order_log10, run_id_to_name
            )

            corrected_log10_results = summarizer.comparison.correct_pairwise(
                comparison_log10_results
            )

            unique_log10_combinations = set(
                (r["endpoint"], r["metric"]) for r in corrected_log10_results
            )
            summary_data["pairwise_comparisons_log10"] = corrected_log10_results
            summary_data["significant_comparisons_log10_05"] = sum(
                1 for r in corrected_log10_results if r["significant_05"]
            )
            summary_data["significant_comparisons_log10_01"] = sum(
                1 for r in corrected_log10_results if r["significant_01"]
            )
            summary_data["correction_groups_log10"] = (
                f"Applied separately for each endpoint-metric combination ({len(unique_log10_combinations)} groups)"
            )
            if omnibus_log10_results:
                serializable_omnibus_log10 = {}
                for (ep, met), result in omnibus_log10_results.items():
                    serializable_omnibus_log10[f"{ep}__{met}"] = result
                summary_data["omnibus_tests_log10"] = serializable_omnibus_log10
        except Exception:
            logger.info("Found no log10-transformed data...")
            corrected_log10_results = None

        # Create performance visualization plots
        logger.info("Creating performance visualization plots...")
        plot_files = summarizer.create_performance_plots(
            performance_data,
            performance_log10_data,
            run_id_to_name,
            corrected_results,
            corrected_log10_results,
            omnibus_results=omnibus_results,
            omnibus_log10_results=omnibus_log10_results,
            output_dir=output_dir,
        )

        # Load predictions and create correlation heatmaps
        logger.info("Loading prediction artifacts for correlation analysis...")
        predictions_data = summarizer.load_predictions_artifacts(run_info_list)

        if predictions_data:
            logger.info("Creating correlation heatmaps...")
            correlation_plot_files = summarizer.create_correlation_heatmaps(
                predictions_data, run_id_to_name, output_dir=output_dir
            )
            # Merge correlation plots with performance plots
            plot_files.update(correlation_plot_files)
            logger.info(
                f"Created and logged {len(correlation_plot_files)} correlation heatmaps"
            )
        else:
            logger.warning("No predictions data found for correlation analysis")

        # Save to backend with plots
        summarizer.save_summary_to_backend(summary_data, plot_files)

        logger.info("Experiment summarization completed successfully")
        logger.info(f"Created and logged {len(plot_files)} total visualization plots")

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise


if __name__ == "__main__":
    main()
