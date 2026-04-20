"""Interactive Plotly dashboard for performance metrics."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ACTIVE_CONSTRAINT_LABELS = {
    "active_hard_constraints": "Active hard constraints",
    "active_hard_constraints_fraction": "Fraction of active hard constraints",
    "active_previous_priority_constraints": "Active constraints from earlier priorities",
}

SHADOW_PRICE_LABEL = "Sum of absolute shadow prices"

GOAL_METRIC_LABELS = {
    "timeseries_sum": "Timeseries Sum",
    "timeseries_min": "Timeseries Minimum",
    "timeseries_max": "Timeseries Maximum",
    "timeseries_avg": "Timeseries Average",
    "mean_absolute_percentual_difference": "Mean Absolute Percentual Difference",
    "mean_absolute_difference": "Mean Absolute Difference",
    "max_difference": "Maximum Difference",
    "perc_below_target": "Percentage Below Target",
    "perc_above_target": "Percentage Above Target",
    "sum_below_target": "Sum Below Target",
    "sum_above_target": "Sum Above Target",
}


def _metric_display_label(metric_name: str) -> str:
    """Return a user-friendly label for a metric key."""
    if metric_name in ACTIVE_CONSTRAINT_LABELS:
        return ACTIVE_CONSTRAINT_LABELS[metric_name]
    if metric_name in GOAL_METRIC_LABELS:
        return GOAL_METRIC_LABELS[metric_name]
    return metric_name.replace("_", " ").title()


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
                meta=metric_name,
                visible=(metric_idx == 0),
                hovertemplate=(
                    "goal=%{fullData.name}<br>priority=%{x}<br>value=%{y}<extra></extra>"
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
                "label": _metric_display_label(metric_name),
                "method": "update",
                "args": [
                    {"visible": visible},
                    {},
                ],
            }
        )

    fig.update_layout(
        template="plotly_white",
        margin={"t": 120, "r": 40},
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


def _format_metric_value(value: Any) -> str:
    """Format metric values for table output."""
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _build_goal_tables_html(
    performance_metrics: dict[str, pd.DataFrame], goal_order: list[str]
) -> str:
    """Create the HTML for the table-based metrics view."""
    if not goal_order:
        return "<p>No performance metrics available.</p>"

    # Normalize performance metric keys to strings so they match goal_order,
    # which is built from str(goal_id) in _flatten_performance_metrics.
    normalized_metrics: dict[str, pd.DataFrame] = {
        str(goal_key): table for goal_key, table in performance_metrics.items()
    }

    options = ['<option value="__all_goals__">All goals</option>']
    panels: list[str] = []

    for goal_idx, goal_id in enumerate(goal_order):
        table = normalized_metrics.get(goal_id)
        if table is None or table.empty:
            continue

        selected = " selected" if goal_idx == 0 else ""
        escaped_goal_id = html.escape(goal_id, quote=True)
        options.append(f'<option value="{escaped_goal_id}"{selected}>{escaped_goal_id}</option>')

        headers = "".join(
            f"<th>{html.escape(_metric_display_label(str(column)))}</th>"
            for column in table.columns
        )
        rows: list[str] = []
        for priority, row in table.iterrows():
            cells = "".join(f"<td>{_format_metric_value(value)}</td>" for value in row.tolist())
            rows.append(f"<tr><th scope='row'>{html.escape(str(priority))}</th>{cells}</tr>")

        active_class = " active" if goal_idx == 0 else ""
        panels.append(
            '<div class="goal-table-panel'
            f'{active_class}" data-goal-table="{escaped_goal_id}">'
            f"<h3 class='goal-table-title'>{escaped_goal_id}</h3>"
            "<table class='metric-table'>"
            f"<thead><tr><th>Priority</th>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</div>"
        )

    return (
        "<div class='goal-table-controls'>"
        "<label for='goal-table-select'><strong>Goal:</strong></label> "
        f"<select id='goal-table-select' class='goal-table-select'>{''.join(options)}</select>"
        "</div>"
        f"{''.join(panels)}"
    )


def _build_active_constraint_tables_html(active_constraint_metrics: pd.DataFrame) -> str:
    """Create the HTML for the priority-level active-constraint summary."""
    if active_constraint_metrics is None or active_constraint_metrics.empty:
        return "<p>No active constraint summary available.</p>"

    headers = "".join(
        (f"<th>{html.escape(_metric_display_label(str(column)))}</th>")
        for column in active_constraint_metrics.columns
    )
    rows: list[str] = []
    for priority, row in active_constraint_metrics.iterrows():
        cells = "".join(f"<td>{_format_metric_value(value)}</td>" for value in row.tolist())
        rows.append(f"<tr><th scope='row'>{html.escape(str(priority))}</th>{cells}</tr>")

    return (
        "<div class='active-constraint-table-panel active'>"
        "<h3 class='goal-table-title'>Constraint activity by priority</h3>"
        "<table class='metric-table'>"
        f"<thead><tr><th>Priority</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _build_shadow_price_tables_html(shadow_price_metrics: pd.DataFrame | None) -> str:
    """Create the HTML for the lower-triangular shadow-price summary."""
    if shadow_price_metrics is None or shadow_price_metrics.empty:
        return "<p>No shadow price summary available.</p>"

    formatted_columns = [f"From priority {column}" for column in shadow_price_metrics.columns]
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in formatted_columns)
    rows: list[str] = []
    for priority, row in shadow_price_metrics.iterrows():
        cells = "".join(f"<td>{_format_metric_value(value)}</td>" for value in row.tolist())
        rows.append(f"<tr><th scope='row'>{html.escape(str(priority))}</th>{cells}</tr>")

    return (
        "<div class='active-constraint-table-panel active'>"
        "<h3 class='goal-table-title'>Shadow prices from earlier priorities</h3>"
        f"<p>{html.escape(SHADOW_PRICE_LABEL)} aggregated by source priority.</p>"
        "<table class='metric-table'>"
        f"<thead><tr><th>Priority</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def create_performance_metrics_dashboard(
    performance_metrics: dict[str, pd.DataFrame],
    active_constraint_metrics: pd.DataFrame | None,
    shadow_price_metrics: pd.DataFrame | None,
    output_folder: str | Path,
    file_name: str = "performance_metrics_dashboard.html",
) -> tuple[dict[str, Any], Path]:
    """
    Create an HTML dashboard for performance metrics.

    Returns
    -------
    figures, html_path
        figures is a dict with the created chart figure and table metadata.
        html_path is the written dashboard file.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    long_df, goal_order, priority_order, metric_order = _flatten_performance_metrics(
        performance_metrics
    )

    metric_by_goal = _metric_by_goal_figure(long_df, goal_order, priority_order, metric_order)
    figures = {
        "metric_by_goal": metric_by_goal,
        "goal_order": goal_order,
        "metric_order": metric_order,
        "active_constraint_metrics": active_constraint_metrics,
        "shadow_price_metrics": shadow_price_metrics,
    }

    goal_tables_html = _build_goal_tables_html(performance_metrics, goal_order)
    active_constraint_tables_html = _build_active_constraint_tables_html(active_constraint_metrics)
    shadow_price_tables_html = _build_shadow_price_tables_html(shadow_price_metrics)
    default_metric = metric_order[0] if metric_order else ""
    metric_label_to_key = {_metric_display_label(metric): metric for metric in metric_order}

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Performance Metrics Dashboard</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; }",
        "h1, h2 { margin-bottom: 0.4rem; }",
        ".tab-buttons { display: flex; gap: 12px; margin: 24px 0 16px; }",
        (
            ".tab-button { background: #f3f4f6; border: 1px solid #cbd5e1; "
            "border-radius: 6px; cursor: pointer; font-size: 14px; "
            "padding: 10px 16px; }"
        ),
        ".tab-button.active { background: #2563eb; border-color: #2563eb; color: white; }",
        ".tab-panel { display: none; }",
        ".tab-panel.active { display: block; }",
        ".metric-chart-wrapper { position: relative; }",
        (
            ".chart-controls { position: absolute; top: 58px; right: 12px; "
            "z-index: 10; display: flex; flex-direction: row; gap: 8px; "
            "align-items: center; flex-wrap: nowrap; }"
        ),
        (
            ".chart-control-button { background: #f8fafc; border: 1px solid #cbd5e1; "
            "border-radius: 6px; cursor: pointer; font-size: 14px; "
            "padding: 8px 14px; }"
        ),
        ".chart-control-button:hover { background: #eef2ff; }",
        ".goal-table-controls { margin-bottom: 16px; }",
        ".goal-table-select { min-width: 320px; padding: 6px 8px; }",
        ".goal-table-panel { display: none; }",
        ".goal-table-panel.active { display: block; }",
        ".goal-table-panel + .goal-table-panel.active { margin-top: 24px; }",
        ".goal-table-title { margin: 0 0 12px; font-size: 16px; }",
        ".metric-table { border-collapse: collapse; width: 100%; }",
        (
            ".metric-table th, .metric-table td { border: 1px solid #d1d5db; "
            "padding: 8px 10px; text-align: left; vertical-align: top; }"
        ),
        ".metric-table thead th { background: #f8fafc; position: sticky; top: 0; }",
        ".metric-table tbody tr:nth-child(even) { background: #f9fafb; }",
        "</style></head><body>",
        "<h1>Performance Metrics Dashboard</h1>",
        (
            "<div class='tab-buttons'>"
            "<button class='tab-button active' type='button' "
            "data-target='metric-by-goal'>Bar Charts</button>"
            "<button class='tab-button' type='button' "
            "data-target='goal-by-metric'>Tables</button>"
            "<button class='tab-button' type='button' "
            "data-target='active-constraints'>Constraint Activity</button>"
            "<button class='tab-button' type='button' "
            "data-target='shadow-prices'>Shadow Prices</button>"
            "</div>"
        ),
        "<div id='metric-by-goal' class='tab-panel active'><h2>Performance Metrics Bar Chart</h2>",
        "<div class='metric-chart-wrapper'>",
        pio.to_html(metric_by_goal, include_plotlyjs=True, full_html=False),
        (
            "<div class='chart-controls'>"
            "<button id='select-all-goals-button' class='chart-control-button' "
            "type='button'>Select all goals</button>"
            "<button id='unselect-all-goals-button' class='chart-control-button' "
            "type='button'>Unselect all goals</button>"
            "</div>"
        ),
        "</div>",
        "</div>",
        "<div id='goal-by-metric' class='tab-panel'><h2>Performance Metrics Tables</h2>",
        goal_tables_html,
        "</div>",
        "<div id='active-constraints' class='tab-panel'><h2>Constraint Activity</h2>",
        active_constraint_tables_html,
        "</div>",
        "<div id='shadow-prices' class='tab-panel'><h2>Shadow Prices</h2>",
        shadow_price_tables_html,
        "</div>",
        "<script>(function () {",
        "  const buttons = document.querySelectorAll('.tab-button');",
        "  const panels = document.querySelectorAll('.tab-panel');",
        "  function activateTab(targetId) {",
        "    buttons.forEach((button) => {",
        "      button.classList.toggle('active', button.getAttribute('data-target') === targetId);",
        "    });",
        "    panels.forEach((panel) => {",
        "      panel.classList.toggle('active', panel.id === targetId);",
        "    });",
        "  }",
        "  buttons.forEach((button) => {",
        "    button.addEventListener('click', function () {",
        "      activateTab(this.getAttribute('data-target'));",
        "    });",
        "  });",
        "  const metricChart = document.querySelector('#metric-by-goal .plotly-graph-div');",
        "  const selectAllGoalsButton = document.getElementById('select-all-goals-button');",
        "  const unselectAllGoalsButton = document.getElementById('unselect-all-goals-button');",
        f"  let currentMetric = {json.dumps(default_metric)};",
        f"  const metricLabelToKey = {json.dumps(metric_label_to_key)};",
        "  function getTraceIndexesForMetric(metricName) {",
        (
            "    if (!metricChart || !Array.isArray(metricChart.data) || "
            "metricChart.data.length === 0) {"
        ),
        "      return [];",
        "    }",
        "    const indexes = [];",
        "    metricChart.data.forEach((trace, index) => {",
        "      if (trace.meta === metricName) {",
        "        indexes.push(index);",
        "      }",
        "    });",
        "    return indexes;",
        "  }",
        "  function setAllGoalsForActiveMetric(visibleState) {",
        "    const traceIndexes = getTraceIndexesForMetric(currentMetric);",
        "    if (!metricChart || typeof Plotly === 'undefined' || traceIndexes.length === 0) {",
        "      return;",
        "    }",
        (
            "    Plotly.restyle(metricChart, { visible: traceIndexes.map(() => visibleState) }, "
            "traceIndexes);"
        ),
        "  }",
        "  if (metricChart) {",
        "    metricChart.on('plotly_buttonclicked', function (eventData) {",
        "      if (eventData && eventData.button && eventData.button.label) {",
        (
            "        currentMetric = metricLabelToKey[eventData.button.label] || "
            "eventData.button.label;"
        ),
        "      }",
        "    });",
        "  }",
        "  if (selectAllGoalsButton) {",
        "    selectAllGoalsButton.addEventListener('click', function () {",
        "      setAllGoalsForActiveMetric(true);",
        "    });",
        "  }",
        "  if (unselectAllGoalsButton) {",
        "    unselectAllGoalsButton.addEventListener('click', function () {",
        "      setAllGoalsForActiveMetric('legendonly');",
        "    });",
        "  }",
        "  const goalTableSelect = document.getElementById('goal-table-select');",
        "  const goalTablePanels = document.querySelectorAll('.goal-table-panel');",
        "  function activateGoalTable(goalId) {",
        "    goalTablePanels.forEach((panel) => {",
        "      const showAllGoals = goalId === '__all_goals__';",
        (
            "      panel.classList.toggle('active', showAllGoals || "
            "panel.getAttribute('data-goal-table') === goalId);"
        ),
        "    });",
        "  }",
        "  if (goalTableSelect) {",
        "    activateGoalTable(goalTableSelect.value);",
        "    goalTableSelect.addEventListener('change', function () {",
        "      activateGoalTable(this.value);",
        "    });",
        "  }",
        "})();</script>",
        "</body></html>",
    ]

    html_path = output_folder / file_name
    html_path.write_text("".join(html_parts), encoding="utf-8")

    return figures, html_path
