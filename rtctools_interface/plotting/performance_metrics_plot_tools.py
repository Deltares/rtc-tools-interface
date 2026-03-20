"""Interactive Plotly dashboard for performance metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=350)
    return fig


def _flatten_performance_metrics(
    performance_metrics: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    goal_order: list[str] = []
    priority_order: list[str] = []
    metric_order: list[str] = []

    for goal_id, table in performance_metrics.items():
        goal_id = str(goal_id)
        if goal_id not in goal_order:
            goal_order.append(goal_id)

        if table is None or table.empty:
            continue

        for priority_key in table.index:
            priority_label = str(priority_key)
            if priority_label not in priority_order:
                priority_order.append(priority_label)

            row = table.loc[priority_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            for metric_name, value in row.items():
                if metric_name not in metric_order:
                    metric_order.append(metric_name)

                if pd.isna(value):
                    continue

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                records.append(
                    {
                        "goal_id": goal_id,
                        "priority": priority_label,
                        "metric": metric_name,
                        "value": numeric_value,
                    }
                )

    long_df = pd.DataFrame.from_records(records)
    return long_df, goal_order, priority_order, metric_order


def _metric_by_goal_figure(
    long_df: pd.DataFrame,
    goal_order: list[str],
    priority_order: list[str],
    metric_order: list[str],
) -> go.Figure:
    if long_df.empty:
        return _empty_figure("No performance metrics available to plot.")

    fig = go.Figure()
    n_goals = len(goal_order)

    for metric_idx, metric_name in enumerate(metric_order):
        metric_df = long_df[long_df["metric"] == metric_name]

        for goal_id in goal_order:
            goal_df = (
                metric_df[metric_df["goal_id"] == goal_id]
                .set_index("priority")
                .reindex(priority_order)
            )

            y_values = (
                goal_df["value"].tolist()
                if "value" in goal_df.columns
                else [None] * len(priority_order)
            )

            fig.add_bar(
                x=priority_order,
                y=y_values,
                name=goal_id,
                visible=(metric_idx == 0),
                hovertemplate=(
                    "goal=%{fullData.name}<br>"
                    "priority=%{x}<br>"
                    "value=%{y}<extra></extra>"
                ),
            )

    buttons = []
    for metric_idx, metric_name in enumerate(metric_order):
        visible = [False] * (len(metric_order) * n_goals)
        start = metric_idx * n_goals
        end = start + n_goals
        visible[start:end] = [True] * n_goals

        buttons.append(
            {
                "label": metric_name,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Metric-focused view — {metric_name} across goals"},
                ],
            }
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Metric-focused view — {metric_order[0]} across goals",
        barmode="group",
        xaxis_title="Priority",
        yaxis_title="Metric value",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            }
        ],
        height=500,
    )
    return fig


def _goal_by_metric_figure(
    long_df: pd.DataFrame,
    goal_order: list[str],
    priority_order: list[str],
    metric_order: list[str],
) -> go.Figure:
    if long_df.empty:
        return _empty_figure("No performance metrics available to plot.")

    fig = go.Figure()
    n_metrics = len(metric_order)

    for goal_idx, goal_id in enumerate(goal_order):
        goal_df = long_df[long_df["goal_id"] == goal_id]

        for metric_name in metric_order:
            metric_df = (
                goal_df[goal_df["metric"] == metric_name]
                .set_index("priority")
                .reindex(priority_order)
            )

            y_values = (
                metric_df["value"].tolist()
                if "value" in metric_df.columns
                else [None] * len(priority_order)
            )

            fig.add_trace(
                go.Scatter(
                    x=priority_order,
                    y=y_values,
                    mode="lines+markers",
                    name=metric_name,
                    visible=(goal_idx == 0),
                    hovertemplate=(
                        "metric=%{fullData.name}<br>"
                        "priority=%{x}<br>"
                        "value=%{y}<extra></extra>"
                    ),
                )
            )

    buttons = []
    for goal_idx, goal_id in enumerate(goal_order):
        visible = [False] * (len(goal_order) * n_metrics)
        start = goal_idx * n_metrics
        end = start + n_metrics
        visible[start:end] = [True] * n_metrics

        buttons.append(
            {
                "label": goal_id,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Goal-focused view — all metrics for {goal_id}"},
                ],
            }
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Goal-focused view — all metrics for {goal_order[0]}",
        xaxis_title="Priority",
        yaxis_title="Metric value",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            }
        ],
        height=500,
    )
    return fig


def _metric_heatmap_figure(
    long_df: pd.DataFrame,
    goal_order: list[str],
    priority_order: list[str],
    metric_order: list[str],
) -> go.Figure:
    if long_df.empty:
        return _empty_figure("No performance metrics available to plot.")

    fig = go.Figure()

    for metric_idx, metric_name in enumerate(metric_order):
        metric_df = long_df[long_df["metric"] == metric_name]
        pivot = (
            metric_df.pivot_table(
                index="goal_id",
                columns="priority",
                values="value",
                aggfunc="first",
            )
            .reindex(index=goal_order, columns=priority_order)
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=priority_order,
                y=goal_order,
                colorbar={"title": "value"},
                visible=(metric_idx == 0),
                hovertemplate=(
                    "goal=%{y}<br>"
                    "priority=%{x}<br>"
                    "value=%{z}<extra></extra>"
                ),
            )
        )

    buttons = []
    for metric_idx, metric_name in enumerate(metric_order):
        visible = [False] * len(metric_order)
        visible[metric_idx] = True
        buttons.append(
            {
                "label": metric_name,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Heatmap view — {metric_name}"},
                ],
            }
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Heatmap view — {metric_order[0]}",
        xaxis_title="Priority",
        yaxis_title="Goal",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
            }
        ],
        height=max(400, 80 + 35 * len(goal_order)),
    )
    return fig


def create_performance_metrics_dashboard(
    performance_metrics: dict[str, pd.DataFrame],
    output_folder: str | Path,
    file_name: str = "performance_metrics_dashboard.html",
) -> tuple[dict[str, go.Figure], Path]:
    """
    Create an HTML dashboard for performance metrics.

    Returns
    -------
    figures, html_path
        figures is a dict with three plotly figures.
        html_path is the written dashboard file.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    long_df, goal_order, priority_order, metric_order = _flatten_performance_metrics(
        performance_metrics
    )

    metric_by_goal = _metric_by_goal_figure(
        long_df, goal_order, priority_order, metric_order
    )
    goal_by_metric = _goal_by_metric_figure(
        long_df, goal_order, priority_order, metric_order
    )
    heatmap = _metric_heatmap_figure(
        long_df, goal_order, priority_order, metric_order
    )

    figures = {
        "metric_by_goal": metric_by_goal,
        "goal_by_metric": goal_by_metric,
        "heatmap": heatmap,
    }

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Performance Metrics Dashboard</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; }",
        "h1, h2 { margin-bottom: 0.4rem; }",
        "p { color: #444; }",
        ".section { margin-bottom: 40px; }",
        "</style></head><body>",
        "<h1>Performance Metrics Dashboard</h1>",
        "<p>"
        "Use the dropdowns in each chart to switch between metric-centric, "
        "goal-centric, and heatmap views."
        "</p>",
        "<div class='section'><h2>Same metric across goals</h2>",
        pio.to_html(metric_by_goal, include_plotlyjs="cdn", full_html=False),
        "</div>",
        "<div class='section'><h2>All metrics for one goal over priorities</h2>",
        pio.to_html(goal_by_metric, include_plotlyjs=False, full_html=False),
        "</div>",
        "<div class='section'><h2>Heatmap view</h2>",
        pio.to_html(heatmap, include_plotlyjs=False, full_html=False),
        "</div>",
        "</body></html>",
    ]

    html_path = output_folder / file_name
    html_path.write_text("".join(html_parts), encoding="utf-8")

    return figures, html_path
