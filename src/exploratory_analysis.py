"""Exploratory data analysis for workflow scheduling and delay behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_directories, format_percentage, set_plot_style


def _save_figure(figure: plt.Figure, file_path: Path) -> None:
    """Save a matplotlib figure and close it to free memory."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def compute_kpis(dataframe: pd.DataFrame) -> Dict[str, float]:
    """Compute Industrial Engineering style KPIs for the workflow dataset."""
    average_workload_by_department = dataframe.groupby("Department")["Employee_Workload"].mean()
    utilization_proxy_by_department = (
        average_workload_by_department / average_workload_by_department.max()
    )

    kpis = {
        "average_cycle_time_minutes": float(dataframe["Actual_Time_Minutes"].mean()),
        "average_delay_minutes": float(dataframe["Calculated_Delay_Minutes"].mean()),
        "average_task_efficiency": float(dataframe["Task_Efficiency"].mean()),
        "average_department_utilization_proxy": float(utilization_proxy_by_department.mean()),
        "total_operational_cost": float(dataframe["Cost_Per_Task"].sum()),
        "percent_delayed_tasks": float(dataframe["Is_Delayed_Binary"].mean()),
    }
    return kpis


def perform_exploratory_analysis(
    dataframe: pd.DataFrame,
    figures_dir: str | Path,
    reports_dir: str | Path,
) -> Dict[str, pd.DataFrame | Dict[str, float]]:
    """Run EDA, save charts, and return summary tables for later use."""
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    ensure_directories([figures_dir, reports_dir])
    set_plot_style()

    kpis = compute_kpis(dataframe)
    kpi_series = pd.Series(kpis, name="value")
    kpi_series.to_csv(reports_dir / "eda_kpis.csv", header=True)

    department_summary = dataframe.groupby("Department").agg(
        task_count=("Task_ID", "count"),
        delay_rate=("Is_Delayed_Binary", "mean"),
        average_actual_time=("Actual_Time_Minutes", "mean"),
        total_cost=("Cost_Per_Task", "sum"),
        average_workload=("Employee_Workload", "mean"),
    )
    department_summary.to_csv(reports_dir / "department_summary.csv")

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = dataframe[numeric_columns].corr(numeric_only=True)
    correlation_matrix.to_csv(reports_dir / "numeric_correlation_matrix.csv")

    report_lines = [
        "Workflow Scheduling Optimization and Delay Prediction System - EDA Summary",
        "=" * 75,
        f"Average cycle time: {kpis['average_cycle_time_minutes']:.2f} minutes",
        f"Average delay minutes: {kpis['average_delay_minutes']:.2f} minutes",
        f"Average task efficiency: {kpis['average_task_efficiency']:.2f}",
        "Average department utilization proxy: "
        f"{format_percentage(kpis['average_department_utilization_proxy'])}",
        f"Total operational cost: ${kpis['total_operational_cost']:.2f}",
        f"Percent of delayed tasks: {format_percentage(kpis['percent_delayed_tasks'])}",
    ]
    (reports_dir / "eda_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")

    delay_rate_overall = dataframe["Is_Delayed_Binary"].mean()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["On Time", "Delayed"], [1 - delay_rate_overall, delay_rate_overall], color=["#4C78A8", "#E45756"])
    axis.set_title("Overall Task Delay Rate")
    axis.set_ylabel("Share of Tasks")
    _save_figure(figure, figures_dir / "overall_delay_rate.png")

    average_time_df = pd.DataFrame(
        {
            "Metric": ["Estimated Time", "Actual Time"],
            "Minutes": [
                dataframe["Estimated_Time_Minutes"].mean(),
                dataframe["Actual_Time_Minutes"].mean(),
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(average_time_df["Metric"], average_time_df["Minutes"], color=["#72B7B2", "#F58518"])
    axis.set_title("Average Estimated vs Actual Task Time")
    axis.set_ylabel("Minutes")
    _save_figure(figure, figures_dir / "average_actual_vs_estimated_time.png")

    delay_by_department = dataframe.groupby("Department")["Is_Delayed_Binary"].mean().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    delay_by_department.plot(kind="bar", ax=axis, color="#E45756")
    axis.set_title("Delay Rate by Department")
    axis.set_xlabel("Department")
    axis.set_ylabel("Delay Rate")
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figures_dir / "delay_rate_by_department.png")

    delay_by_priority = dataframe.groupby("Priority_Level")["Is_Delayed_Binary"].mean().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    delay_by_priority.plot(kind="bar", ax=axis, color="#4C78A8")
    axis.set_title("Delay Rate by Priority Level")
    axis.set_xlabel("Priority Level")
    axis.set_ylabel("Delay Rate")
    axis.tick_params(axis="x", rotation=0)
    _save_figure(figure, figures_dir / "delay_rate_by_priority_level.png")

    delay_by_task_type = dataframe.groupby("Task_Type")["Is_Delayed_Binary"].mean().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(8, 4))
    delay_by_task_type.plot(kind="bar", ax=axis, color="#54A24B")
    axis.set_title("Delay Rate by Task Type")
    axis.set_xlabel("Task Type")
    axis.set_ylabel("Delay Rate")
    axis.tick_params(axis="x", rotation=20)
    _save_figure(figure, figures_dir / "delay_rate_by_task_type.png")

    workload_by_employee = (
        dataframe.groupby("Assigned_Employee_ID")["Employee_Workload"].mean().sort_values(ascending=False).head(15)
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    workload_by_employee.plot(kind="bar", ax=axis, color="#72B7B2")
    axis.set_title("Average Workload by Employee (Top 15)")
    axis.set_xlabel("Employee ID")
    axis.set_ylabel("Average Workload")
    axis.tick_params(axis="x", rotation=45)
    _save_figure(figure, figures_dir / "average_workload_by_employee.png")

    top_delayed_employees = (
        dataframe[dataframe["Is_Delayed_Binary"] == 1]
        .groupby("Assigned_Employee_ID")["Task_ID"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    top_delayed_employees.plot(kind="bar", ax=axis, color="#E45756")
    axis.set_title("Employees with the Highest Number of Delayed Tasks")
    axis.set_xlabel("Employee ID")
    axis.set_ylabel("Number of Delayed Tasks")
    axis.tick_params(axis="x", rotation=45)
    _save_figure(figure, figures_dir / "top_employees_by_delayed_tasks.png")

    cost_by_department = dataframe.groupby("Department")["Cost_Per_Task"].sum().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    cost_by_department.plot(kind="bar", ax=axis, color="#F58518")
    axis.set_title("Total Operational Cost by Department")
    axis.set_xlabel("Department")
    axis.set_ylabel("Total Cost")
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figures_dir / "cost_by_department.png")

    workload_delay_relationship = dataframe.groupby("Employee_Workload")["Is_Delayed_Binary"].mean()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        workload_delay_relationship.index,
        workload_delay_relationship.values,
        marker="o",
        color="#B279A2",
    )
    axis.set_title("Delay Rate by Employee Workload Level")
    axis.set_xlabel("Employee Workload")
    axis.set_ylabel("Delay Rate")
    _save_figure(figure, figures_dir / "workload_vs_delay_relationship.png")

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        dataframe["Estimated_Time_Minutes"],
        dataframe["Actual_Time_Minutes"],
        alpha=0.45,
        color="#4C78A8",
        edgecolors="none",
    )
    reference_line_max = max(
        dataframe["Estimated_Time_Minutes"].max(),
        dataframe["Actual_Time_Minutes"].max(),
    )
    axis.plot([0, reference_line_max], [0, reference_line_max], linestyle="--", color="black")
    axis.set_title("Actual Time vs Estimated Time")
    axis.set_xlabel("Estimated Time (Minutes)")
    axis.set_ylabel("Actual Time (Minutes)")
    _save_figure(figure, figures_dir / "actual_vs_estimated_time_scatter.png")

    task_duration_values = dataframe["Task_Duration_From_Timestamps"].fillna(dataframe["Actual_Time_Minutes"])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(task_duration_values, bins=25, color="#54A24B", edgecolor="white")
    axis.set_title("Distribution of Task Durations")
    axis.set_xlabel("Duration (Minutes)")
    axis.set_ylabel("Number of Tasks")
    _save_figure(figure, figures_dir / "task_duration_distribution.png")

    departments = list(dataframe["Department"].dropna().unique())
    boxplot_data = [
        dataframe.loc[dataframe["Department"] == department, "Actual_Time_Minutes"].values
        for department in departments
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(boxplot_data, tick_labels=departments, patch_artist=True)
    axis.set_title("Actual Task Time Distribution by Department")
    axis.set_xlabel("Department")
    axis.set_ylabel("Actual Time (Minutes)")
    axis.tick_params(axis="x", rotation=35)
    _save_figure(figure, figures_dir / "actual_time_boxplot_by_department.png")

    figure, axis = plt.subplots(figsize=(8, 6))
    heatmap = axis.imshow(correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation_matrix.columns)))
    axis.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation_matrix.index)))
    axis.set_yticklabels(correlation_matrix.index)
    axis.set_title("Correlation Heatmap for Numeric Variables")
    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    _save_figure(figure, figures_dir / "numeric_correlation_heatmap.png")

    return {
        "kpis": kpis,
        "department_summary": department_summary,
        "correlation_matrix": correlation_matrix,
    }
