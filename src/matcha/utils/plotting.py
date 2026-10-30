"""Plotting utilities for regression and classification model evaluation.

Provides interactive Plotly-based visualizations including scatter plots with
trendlines and fold-error boundaries for regression, and ROC/PR curves with
probability histograms for classification.
"""

import numpy as np
from matcha.sklearn.base_sklearn_model import BaseScikitLearnModel
from matcha.utils.sanitize import ensure_1d_array
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import os


def plot_regression(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    plot_title: str = "Regression performance",
    labels: list = None,
    is_log10: bool = False,
):
    """Create a scatter plot of true vs predicted values with trendline using Plotly.

    Includes OLS trendline, perfect prediction diagonal, and 2-fold/3-fold
    error boundary lines.

    :param numpy.ndarray true_values: Array of true target values.
    :param numpy.ndarray pred_values: Array of predicted target values.
    :param str plot_title: Title for the plot.
    :param list labels: Optional list of labels for hover data corresponding to
        each data point.
    :param bool is_log10: Whether the values are log10-transformed. Adjusts
        fold-error boundary calculations accordingly.
    :returns: A Plotly figure object, or ``None`` if no valid data points exist.
    :rtype: plotly.graph_objects.Figure or None
    """

    true_values = ensure_1d_array(true_values)
    pred_values = ensure_1d_array(pred_values)

    # Remove NaN values
    mask = ~(np.isnan(true_values) | np.isnan(pred_values))
    true_clean = true_values[mask]
    pred_clean = pred_values[mask]

    if len(true_clean) == 0:
        print(f"Warning: No valid data points for {plot_title}")
        return None

    # Create DataFrame for plotting
    df_plot = pd.DataFrame({"True": true_clean, "Predicted": pred_clean})

    # Add SMILES if provided
    if labels is not None:
        labels_clean = [labels[i] for i in range(len(labels)) if mask[i]]
        df_plot["LABELS"] = labels_clean
        hover_data = ["LABELS"]
    else:
        hover_data = None

    # Create scatter plot with custom colors
    fig = px.scatter(
        df_plot,
        x="True",
        y="Predicted",
        title=plot_title,
        hover_data=hover_data,
        color_discrete_sequence=["#503291"],  # Custom dot color
        template="plotly_white",
    )  # Clean white template

    # Compute OLS trendline with numpy and add it
    coeffs = np.polyfit(true_clean, pred_clean, 1)
    trendline_x = np.array([true_clean.min(), true_clean.max()])
    trendline_y = np.polyval(coeffs, trendline_x)
    fig.add_trace(
        go.Scatter(
            x=trendline_x,
            y=trendline_y,
            mode="lines",
            line=dict(color="#eb3c96", width=2),
            name="OLS Trendline",
            showlegend=True,
        )
    )

    # Get data range for lines
    min_val = min(true_clean.min(), pred_clean.min())
    max_val = max(true_clean.max(), pred_clean.max())

    # Add perfect prediction line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="Perfect Prediction",
            showlegend=True,
        )
    )

    # Add fold error boundary lines
    if is_log10:
        # For log10 data, use log10(2) and log10(3) as offsets
        log2 = np.log10(2)
        log3 = np.log10(3)

        # 2-fold boundaries
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val + log2, max_val + log2],
                mode="lines",
                line=dict(color="orange", dash="dot", width=1),
                name="2-fold boundary",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val - log2, max_val - log2],
                mode="lines",
                line=dict(color="orange", dash="dot", width=1),
                showlegend=False,  # Don't show duplicate legend entry
            )
        )

        # 3-fold boundaries
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val + log3, max_val + log3],
                mode="lines",
                line=dict(color="red", dash="dot", width=1),
                name="3-fold boundary",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val - log3, max_val - log3],
                mode="lines",
                line=dict(color="red", dash="dot", width=1),
                showlegend=False,  # Don't show duplicate legend entry
            )
        )
    else:
        # For regular data, create boundaries based on ratios
        # 2-fold: y = 2*x and y = x/2
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[2 * min_val, 2 * max_val],
                mode="lines",
                line=dict(color="orange", dash="dot", width=1),
                name="2-fold boundary",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val / 2, max_val / 2],
                mode="lines",
                line=dict(color="orange", dash="dot", width=1),
                showlegend=False,  # Don't show duplicate legend entry
            )
        )

        # 3-fold: y = 3*x and y = x/3
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[3 * min_val, 3 * max_val],
                mode="lines",
                line=dict(color="red", dash="dot", width=1),
                name="3-fold boundary",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val / 3, max_val / 3],
                mode="lines",
                line=dict(color="red", dash="dot", width=1),
                showlegend=False,  # Don't show duplicate legend entry
            )
        )

    # Update layout
    xlabel = "True Values (log10)" if is_log10 else "True Values"
    ylabel = "Predicted Values (log10)" if is_log10 else "Predicted Values"

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title={
            "text": plot_title,
            "x": 0.5,  # Center the title
            "xanchor": "center",
        },
        width=600,
        height=600,
        showlegend=True,
        template="plotly_white",  # Clean white template
    )

    return fig


def plot_classification(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    prob_values: np.ndarray,
    plot_title: str = "Classification performance",
    model: BaseScikitLearnModel | None = None,
):
    """Create a three-panel classification plot with ROC, PR, and probability histograms.

    Generates an interactive Plotly figure with ROC-AUC curve,
    Precision-Recall curve, and predicted probability distribution histograms
    for each class.

    :param numpy.ndarray true_values: Array of true binary labels (0 or 1).
    :param numpy.ndarray pred_values: Array of predicted binary labels (0 or 1).
    :param numpy.ndarray prob_values: Array of predicted probabilities for the
        positive class.
    :param str plot_title: Title for the plot.
    :param model: Optional model instance used to encode labels via its
        internal label encoder.
    :type model: BaseScikitLearnModel or None
    :returns: A Plotly figure object, or ``None`` if no valid data points exist.
    :rtype: plotly.graph_objects.Figure or None
    """

    true_values = ensure_1d_array(true_values)
    pred_values = ensure_1d_array(pred_values)
    prob_values = ensure_1d_array(prob_values)

    # Remove NaN values
    mask = ~(np.isnan(true_values) | np.isnan(pred_values) | np.isnan(prob_values))
    true_clean = true_values[mask]
    prob_clean = prob_values[mask]

    if model is not None:
        true_clean = model._datamodule._label_encoder._all_to_categorical(true_clean)[
            :, 0
        ]

    if len(true_clean) == 0:
        print(f"Warning: No valid data points for {plot_title}")
        return None

    # Create subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "ROC Curve",
            "Precision-Recall Curve",
            "Probability Distribution",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}]],
    )

    # 1. ROC-AUC curve
    fpr, tpr, _ = roc_curve(true_clean, prob_clean)
    roc_auc = roc_auc_score(true_clean, prob_clean)

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line=dict(color="#0f69af", width=2),
        ),
        row=1,
        col=1,
    )

    # Add diagonal line for ROC
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_clean, prob_clean)
    pr_auc = average_precision_score(true_clean, prob_clean)

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AUC = {pr_auc:.3f})",
            line=dict(color="#0f69af", width=2),
        ),
        row=1,
        col=2,
    )

    # Add baseline for PR curve (proportion of positive class)
    baseline = np.mean(true_clean)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Histogram of probabilities for each class
    prob_class0 = prob_clean[true_clean == 0]
    prob_class1 = prob_clean[true_clean == 1]

    fig.add_trace(
        go.Histogram(
            x=prob_class0,
            name="Class 0",
            marker_color="#503291",
            opacity=0.7,
            nbinsx=30,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Histogram(
            x=prob_class1,
            name="Class 1",
            marker_color="#eb3c96",
            opacity=0.7,
            nbinsx=30,
        ),
        row=1,
        col=3,
    )

    # Update layout for each subplot
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)

    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    fig.update_xaxes(title_text="Predicted Probability", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=3)

    # Update overall layout
    fig.update_layout(
        title={
            "text": plot_title,
            "x": 0.5,
            "xanchor": "center",
        },
        width=1800,  # Wider for three subplots
        height=500,
        showlegend=True,
        template="plotly_white",
        barmode="overlay",  # For overlapping histograms
    )

    return fig


def save_plot(fig, file_path):
    """Save a Plotly figure as an HTML file.

    Creates parent directories if they do not exist.

    :param plotly.graph_objects.Figure fig: The Plotly figure to save.
    :param str file_path: Destination file path (should end in ``.html``).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fig.write_html(file_path, include_plotlyjs="cdn")
