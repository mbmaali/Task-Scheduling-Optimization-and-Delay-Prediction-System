"""Evaluation utilities for comparing original and optimized task assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_directories, set_plot_style


METRIC_LABELS = {
    "average_workload": "Average Workload",
    "workload_standard_deviation": "Workload Standard Deviation",
    "average_predicted_delay_risk": "Average Predicted Delay Risk",
    "total_predicted_delay_risk": "Total Predicted Delay Risk",
    "total_cost": "Total Estimated Cost",
    "number_of_overloaded_employees": "Overloaded Employees",
}


def _save_figure(figure: plt.Figure, file_path: Path) -> None:
    """Save a matplotlib figure and close it afterwards."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_optimization_results(
    optimization_results: Dict[str, object],
    figures_dir: str | Path,
    reports_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    """Compare the original and optimized assignments and save visual reports."""
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    ensure_directories([figures_dir, reports_dir])
    set_plot_style()

    before_metrics = optimization_results["before_metrics"]
    after_metrics = optimization_results["after_metrics"]
    employee_comparison = optimization_results["employee_comparison"].copy()

    comparison_df = pd.DataFrame(
        {
            "metric": list(before_metrics.keys()),
            "before": [before_metrics[key] for key in before_metrics],
            "after": [after_metrics[key] for key in before_metrics],
        }
    )
    comparison_df["improvement"] = comparison_df["before"] - comparison_df["after"]
    comparison_df["metric_label"] = comparison_df["metric"].map(METRIC_LABELS)
    comparison_df.to_csv(reports_dir / "before_vs_after_metrics.csv", index=False)

    report_lines = [
        "Before vs After Optimization Summary",
        "=" * 60,
    ]
    for row in comparison_df.itertuples(index=False):
        report_lines.append(
            f"{row.metric_label}: before={row.before:.4f}, after={row.after:.4f}, improvement={row.improvement:.4f}"
        )
    (reports_dir / "before_vs_after_summary.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    employee_plot_df = employee_comparison.sort_values(
        by="After_Total_Workload", ascending=False
    ).head(15)
    x_positions = range(len(employee_plot_df))
    bar_width = 0.4

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        employee_plot_df["Before_Total_Workload"],
        width=bar_width,
        label="Before Optimization",
        color="#E45756",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        employee_plot_df["After_Total_Workload"],
        width=bar_width,
        label="After Optimization",
        color="#4C78A8",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(employee_plot_df["Assigned_Employee_ID"], rotation=45)
    axis.set_title("Employee Workload Distribution Before and After Optimization")
    axis.set_xlabel("Employee ID")
    axis.set_ylabel("Total Workload Proxy (Minutes)")
    axis.legend()
    _save_figure(figure, figures_dir / "before_after_workload_distribution.png")

    delay_risk_df = pd.DataFrame(
        {
            "Scenario": ["Before", "After"],
            "Total Predicted Delay Risk": [
                before_metrics["total_predicted_delay_risk"],
                after_metrics["total_predicted_delay_risk"],
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        delay_risk_df["Scenario"],
        delay_risk_df["Total Predicted Delay Risk"],
        color=["#E45756", "#4C78A8"],
    )
    axis.set_title("Total Predicted Delay Risk Before and After Optimization")
    axis.set_ylabel("Risk Score")
    _save_figure(figure, figures_dir / "before_after_delay_risk.png")

    cost_df = pd.DataFrame(
        {
            "Scenario": ["Before", "After"],
            "Total Estimated Cost": [before_metrics["total_cost"], after_metrics["total_cost"]],
        }
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(cost_df["Scenario"], cost_df["Total Estimated Cost"], color=["#F58518", "#72B7B2"])
    axis.set_title("Total Estimated Cost Before and After Optimization")
    axis.set_ylabel("Cost")
    _save_figure(figure, figures_dir / "before_after_cost_comparison.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        [
            employee_comparison["Before_Total_Workload"],
            employee_comparison["After_Total_Workload"],
        ],
        tick_labels=["Before", "After"],
        patch_artist=True,
    )
    axis.set_title("Workload Distribution Spread Before and After Optimization")
    axis.set_ylabel("Total Workload Proxy (Minutes)")
    _save_figure(figure, figures_dir / "before_after_workload_boxplot.png")

    return {
        "comparison_df": comparison_df,
        "employee_comparison": employee_comparison,
    }
