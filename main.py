"""Run the full workflow optimization and delay prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_preprocessing import (
    clean_workflow_data,
    inspect_data_types,
    load_workflow_data,
    print_data_summary,
)
from src.delay_prediction import train_delay_prediction_models
from src.evaluation import evaluate_optimization_results
from src.exploratory_analysis import perform_exploratory_analysis
from src.feature_engineering import create_derived_features
from src.optimization_model import optimize_task_assignments
from src.utils import PROJECT_ROOT, ensure_directories, resolve_data_path


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
MODELS_DIR = OUTPUT_ROOT / "models"
REPORTS_DIR = OUTPUT_ROOT / "reports"


def run_pipeline(data_path: str | None = None, max_tasks: int = 25, max_employees: int = 12) -> dict:
    """Run the complete project pipeline from raw data to optimization results."""
    ensure_directories([OUTPUT_ROOT, FIGURES_DIR, MODELS_DIR, REPORTS_DIR])

    resolved_data_path = resolve_data_path(data_path)
    print(f"Using dataset: {resolved_data_path}")

    raw_dataframe = load_workflow_data(str(resolved_data_path))
    print("\nOriginal data types:")
    print(inspect_data_types(raw_dataframe))

    cleaned_dataframe = clean_workflow_data(raw_dataframe)
    engineered_dataframe = create_derived_features(cleaned_dataframe)
    print_data_summary(engineered_dataframe)

    engineered_dataframe.to_csv(REPORTS_DIR / "cleaned_engineered_workflow_data.csv", index=False)

    eda_results = perform_exploratory_analysis(
        engineered_dataframe,
        figures_dir=FIGURES_DIR,
        reports_dir=REPORTS_DIR,
    )
    print("\nEDA complete. KPI report saved to outputs/reports/eda_kpis.csv")

    model_results = train_delay_prediction_models(
        engineered_dataframe,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        figures_dir=FIGURES_DIR,
    )
    print(
        f"Best delay prediction model: {model_results['best_model_name']} "
        f"(saved to {MODELS_DIR / 'best_delay_prediction_model.joblib'})"
    )

    optimization_results = optimize_task_assignments(
        engineered_dataframe,
        model_bundle=model_results["best_model_bundle"],
        output_root=OUTPUT_ROOT,
        max_tasks=max_tasks,
        max_employees=max_employees,
        enforce_department_compatibility=False,
        max_tasks_per_employee=4,
    )
    print("Optimization complete. Assignments saved to outputs/optimized_assignments.csv")

    evaluation_results = evaluate_optimization_results(
        optimization_results,
        figures_dir=FIGURES_DIR,
        reports_dir=REPORTS_DIR,
    )
    print("Before-vs-after evaluation saved to outputs/reports/before_vs_after_metrics.csv")

    summary_path = REPORTS_DIR / "pipeline_summary.txt"
    summary_lines = [
        "Workflow Task Scheduling Optimization and Delay Prediction System",
        "=" * 75,
        f"Rows analyzed: {len(engineered_dataframe)}",
        f"Best prediction model: {model_results['best_model_name']}",
        f"Total predicted delay risk before optimization: {optimization_results['before_metrics']['total_predicted_delay_risk']:.4f}",
        f"Total predicted delay risk after optimization: {optimization_results['after_metrics']['total_predicted_delay_risk']:.4f}",
        f"Total cost before optimization: ${optimization_results['before_metrics']['total_cost']:.2f}",
        f"Total cost after optimization: ${optimization_results['after_metrics']['total_cost']:.2f}",
        "Generated files: charts, reports, trained model, and optimized assignment CSV.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "data": engineered_dataframe,
        "eda_results": eda_results,
        "model_results": model_results,
        "optimization_results": optimization_results,
        "evaluation_results": evaluation_results,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the workflow task scheduling optimization pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to the workflow CSV file.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=25,
        help="Number of tasks to include in the optimization what-if study.",
    )
    parser.add_argument(
        "--max-employees",
        type=int,
        default=12,
        help="Number of candidate employees to include in the optimization pool.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_pipeline(
        data_path=arguments.data_path,
        max_tasks=arguments.max_tasks,
        max_employees=arguments.max_employees,
    )
