"""Linear optimization model for task assignment and workload balancing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.delay_prediction import predict_delay_probabilities
from src.feature_engineering import get_model_feature_columns
from src.utils import DEFAULT_RANDOM_STATE, ensure_directories

try:
    import pulp
except ImportError as exc:  # pragma: no cover - handled by requirements.txt
    raise ImportError(
        "PuLP is required for the optimization module. Install it from requirements.txt."
    ) from exc


def _mode_or_unknown(series: pd.Series) -> str:
    """Return the most common value in a series, or 'Unknown' when unavailable."""
    non_null_values = series.dropna()
    if non_null_values.empty:
        return "Unknown"
    modes = non_null_values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(non_null_values.iloc[0])


def build_employee_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate employee-level performance and workload proxies from history."""
    employee_summary = dataframe.groupby("Assigned_Employee_ID").agg(
        Average_Workload=("Employee_Workload", "mean"),
        Average_Actual_Task_Time=("Actual_Time_Minutes", "mean"),
        Employee_Delay_Rate=("Is_Delayed_Binary", "mean"),
        Average_Cost_Per_Task=("Cost_Per_Task", "mean"),
        Historical_Task_Count=("Task_ID", "count"),
    )
    employee_summary = employee_summary.reset_index()

    department_map = (
        dataframe.groupby("Assigned_Employee_ID")["Department"].agg(_mode_or_unknown).reset_index()
    )
    employee_summary = employee_summary.merge(
        department_map,
        on="Assigned_Employee_ID",
        how="left",
    )
    employee_summary = employee_summary.rename(columns={"Department": "Primary_Department"})

    employee_summary["Average_Cost_Per_Minute"] = (
        employee_summary["Average_Cost_Per_Task"]
        / employee_summary["Average_Actual_Task_Time"].replace(0, np.nan)
    )
    employee_summary["Average_Cost_Per_Minute"] = employee_summary[
        "Average_Cost_Per_Minute"
    ].fillna(employee_summary["Average_Cost_Per_Task"].median() / max(dataframe["Actual_Time_Minutes"].median(), 1))

    # This proxy turns workload counts into comparable minute-based load.
    employee_summary["Current_Load_Minutes"] = (
        employee_summary["Average_Workload"] * employee_summary["Average_Actual_Task_Time"]
    )

    return employee_summary.sort_values(
        by=["Historical_Task_Count", "Employee_Delay_Rate"],
        ascending=[False, True],
    ).reset_index(drop=True)


def select_tasks_for_optimization(
    dataframe: pd.DataFrame,
    max_tasks: int = 25,
) -> pd.DataFrame:
    """Select a recent subset of tasks for the what-if assignment study."""
    working_df = dataframe.copy()

    if working_df["Task_Start_Time"].notna().any():
        working_df = working_df.sort_values("Task_Start_Time", ascending=False)
    else:
        working_df = working_df.sort_index(ascending=False)

    selected_tasks = working_df.head(min(max_tasks, len(working_df))).copy()
    selected_tasks = selected_tasks.reset_index(drop=True)
    return selected_tasks


def select_candidate_employees(
    employee_summary: pd.DataFrame,
    tasks_to_assign: pd.DataFrame,
    max_employees: int = 12,
) -> pd.DataFrame:
    """Create a candidate employee pool while preserving the original assignees."""
    relevant_departments = tasks_to_assign["Department"].dropna().unique().tolist()
    original_employees = tasks_to_assign["Assigned_Employee_ID"].dropna().unique().tolist()

    department_pool = employee_summary[
        employee_summary["Primary_Department"].isin(relevant_departments)
    ].copy()

    base_pool = pd.concat(
        [
            department_pool,
            employee_summary[employee_summary["Assigned_Employee_ID"].isin(original_employees)],
            employee_summary.head(max_employees),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["Assigned_Employee_ID"])

    prioritized_pool = base_pool.sort_values(
        by=["Employee_Delay_Rate", "Average_Workload", "Historical_Task_Count"],
        ascending=[True, True, False],
    )

    required_count = max(len(original_employees), max_employees)
    candidate_employees = prioritized_pool.head(required_count).copy()

    missing_original_employees = [
        employee_id
        for employee_id in original_employees
        if employee_id not in candidate_employees["Assigned_Employee_ID"].tolist()
    ]
    if missing_original_employees:
        supplemental_rows = employee_summary[
            employee_summary["Assigned_Employee_ID"].isin(missing_original_employees)
        ]
        candidate_employees = pd.concat(
            [candidate_employees, supplemental_rows], ignore_index=True
        ).drop_duplicates(subset=["Assigned_Employee_ID"])

    return candidate_employees.reset_index(drop=True)


def _build_assignment_feature_frame(
    tasks_to_assign: pd.DataFrame,
    candidate_employees: pd.DataFrame,
    model_bundle: Dict[str, object],
    enforce_department_compatibility: bool,
) -> pd.DataFrame:
    """Build one feature record for every allowed task-employee pairing."""
    feature_columns = get_model_feature_columns()
    assignment_records: List[Dict[str, object]] = []

    max_current_load = max(candidate_employees["Current_Load_Minutes"].max(), 1)

    for task_row in tasks_to_assign.itertuples(index=False):
        for employee_row in candidate_employees.itertuples(index=False):
            department_match = task_row.Department == employee_row.Primary_Department
            if enforce_department_compatibility and not department_match:
                continue

            estimated_assignment_cost = (
                employee_row.Average_Cost_Per_Minute * task_row.Estimated_Time_Minutes
            )

            record = {
                "Task_ID": task_row.Task_ID,
                "Assigned_Employee_ID": employee_row.Assigned_Employee_ID,
                "Department_Match": department_match,
                "Employee_Delay_Rate": employee_row.Employee_Delay_Rate,
                "Current_Load_Minutes": employee_row.Current_Load_Minutes,
                "Current_Load_Normalized": employee_row.Current_Load_Minutes / max_current_load,
                "Estimated_Assignment_Cost": estimated_assignment_cost,
            }

            for column in feature_columns:
                if column == "Employee_Workload":
                    record[column] = employee_row.Average_Workload
                elif column == "Cost_Per_Task":
                    record[column] = estimated_assignment_cost
                else:
                    record[column] = getattr(task_row, column)

            assignment_records.append(record)

    assignment_feature_frame = pd.DataFrame(assignment_records)
    if assignment_feature_frame.empty:
        raise ValueError(
            "No feasible task-employee pairs were generated. Try disabling department compatibility."
        )

    model_probabilities = predict_delay_probabilities(
        model_bundle,
        assignment_feature_frame[feature_columns],
    )
    assignment_feature_frame["Base_Predicted_Delay_Risk"] = model_probabilities

    mismatch_penalty = np.where(assignment_feature_frame["Department_Match"], 0.0, 0.05)
    assignment_feature_frame["Adjusted_Predicted_Delay_Risk"] = np.clip(
        assignment_feature_frame["Base_Predicted_Delay_Risk"]
        + 0.20 * assignment_feature_frame["Employee_Delay_Rate"]
        + 0.05 * assignment_feature_frame["Current_Load_Normalized"]
        + mismatch_penalty,
        0,
        1,
    )

    return assignment_feature_frame


def _planning_capacity_minutes(
    tasks_to_assign: pd.DataFrame,
    candidate_employees: pd.DataFrame,
) -> Dict[str, float]:
    """Estimate a simple planning capacity for each candidate employee."""
    total_task_minutes = tasks_to_assign["Estimated_Time_Minutes"].sum()
    employee_count = max(len(candidate_employees), 1)
    average_new_load = total_task_minutes / employee_count
    median_task_time = max(tasks_to_assign["Estimated_Time_Minutes"].median(), 1)
    base_capacity = max(average_new_load * 1.6, median_task_time * 2.5)

    capacities = {}
    for employee_row in candidate_employees.itertuples(index=False):
        efficiency_bonus = 1.0 + max(0.0, 0.15 - 0.10 * employee_row.Employee_Delay_Rate)
        capacities[employee_row.Assigned_Employee_ID] = base_capacity * efficiency_bonus

    return capacities


def optimize_task_assignments(
    dataframe: pd.DataFrame,
    model_bundle: Dict[str, object],
    output_root: str | Path,
    max_tasks: int = 25,
    max_employees: int = 12,
    enforce_department_compatibility: bool = False,
    max_tasks_per_employee: int | None = 4,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    """Solve a beginner-friendly linear task assignment model using PuLP."""
    _ = random_state  # Reserved for future extensions and reproducibility notes.

    output_root = Path(output_root)
    reports_dir = output_root / "reports"
    ensure_directories([output_root, reports_dir])

    employee_summary = build_employee_summary(dataframe)
    tasks_to_assign = select_tasks_for_optimization(dataframe, max_tasks=max_tasks)
    candidate_employees = select_candidate_employees(
        employee_summary, tasks_to_assign, max_employees=max_employees
    )

    try:
        assignment_options = _build_assignment_feature_frame(
            tasks_to_assign=tasks_to_assign,
            candidate_employees=candidate_employees,
            model_bundle=model_bundle,
            enforce_department_compatibility=enforce_department_compatibility,
        )
    except ValueError:
        assignment_options = _build_assignment_feature_frame(
            tasks_to_assign=tasks_to_assign,
            candidate_employees=candidate_employees,
            model_bundle=model_bundle,
            enforce_department_compatibility=False,
        )
        enforce_department_compatibility = False

    tasks = tasks_to_assign["Task_ID"].tolist()
    employees = candidate_employees["Assigned_Employee_ID"].tolist()
    estimated_time_map = tasks_to_assign.set_index("Task_ID")["Estimated_Time_Minutes"].to_dict()

    risk_map = assignment_options.set_index(["Task_ID", "Assigned_Employee_ID"])[
        "Adjusted_Predicted_Delay_Risk"
    ].to_dict()
    cost_map = assignment_options.set_index(["Task_ID", "Assigned_Employee_ID"])[
        "Estimated_Assignment_Cost"
    ].to_dict()
    allowed_pairs = list(risk_map.keys())

    planning_capacity = _planning_capacity_minutes(tasks_to_assign, candidate_employees)
    current_load_map = candidate_employees.set_index("Assigned_Employee_ID")[
        "Current_Load_Minutes"
    ].to_dict()

    total_task_minutes = sum(estimated_time_map.values())
    target_total_workload = np.mean(list(current_load_map.values())) + (
        total_task_minutes / max(len(employees), 1)
    )
    balance_scale = max(target_total_workload, 1)
    max_cost_value = max(cost_map.values()) if cost_map else 1

    if max_tasks_per_employee is not None:
        minimum_required_slots = math.ceil(len(tasks) / max(len(employees), 1))
        max_tasks_per_employee = max(max_tasks_per_employee, minimum_required_slots)

    optimization_problem = pulp.LpProblem(
        "Workflow_Task_Assignment_Optimization", pulp.LpMinimize
    )

    assignment_variables = pulp.LpVariable.dicts(
        "assign",
        allowed_pairs,
        lowBound=0,
        upBound=1,
        cat="Binary",
    )
    deviation_positive = pulp.LpVariable.dicts(
        "dev_pos", employees, lowBound=0, cat="Continuous"
    )
    deviation_negative = pulp.LpVariable.dicts(
        "dev_neg", employees, lowBound=0, cat="Continuous"
    )

    delay_weight = 0.60
    balance_weight = 0.25
    cost_weight = 0.15

    optimization_problem += (
        delay_weight
        * pulp.lpSum(risk_map[pair] * assignment_variables[pair] for pair in allowed_pairs)
        + cost_weight
        * pulp.lpSum(
            (cost_map[pair] / max_cost_value) * assignment_variables[pair]
            for pair in allowed_pairs
        )
        + balance_weight
        * (1 / balance_scale)
        * pulp.lpSum(
            deviation_positive[employee_id] + deviation_negative[employee_id]
            for employee_id in employees
        )
    )

    for task_id in tasks:
        valid_employee_pairs = [
            (task_id, employee_id)
            for employee_id in employees
            if (task_id, employee_id) in assignment_variables
        ]
        optimization_problem += (
            pulp.lpSum(assignment_variables[pair] for pair in valid_employee_pairs) == 1,
            f"assign_each_task_once_{task_id}",
        )

    for employee_id in employees:
        assigned_minutes = pulp.lpSum(
            estimated_time_map[task_id] * assignment_variables[(task_id, employee_id)]
            for task_id in tasks
            if (task_id, employee_id) in assignment_variables
        )

        optimization_problem += (
            assigned_minutes <= planning_capacity[employee_id],
            f"capacity_limit_{employee_id}",
        )

        if max_tasks_per_employee is not None:
            optimization_problem += (
                pulp.lpSum(
                    assignment_variables[(task_id, employee_id)]
                    for task_id in tasks
                    if (task_id, employee_id) in assignment_variables
                )
                <= max_tasks_per_employee,
                f"max_tasks_limit_{employee_id}",
            )

        optimization_problem += (
            current_load_map[employee_id]
            + assigned_minutes
            - target_total_workload
            == deviation_positive[employee_id] - deviation_negative[employee_id],
            f"workload_balance_{employee_id}",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    optimization_problem.solve(solver)

    status = pulp.LpStatus.get(optimization_problem.status, "Unknown")
    if status != "Optimal":
        raise RuntimeError(
            f"Optimization did not find an optimal solution. Solver status: {status}"
        )

    optimized_assignment_map = {}
    for task_id in tasks:
        for employee_id in employees:
            pair = (task_id, employee_id)
            if pair in assignment_variables and pulp.value(assignment_variables[pair]) > 0.5:
                optimized_assignment_map[task_id] = employee_id
                break

    original_assignment_map = tasks_to_assign.set_index("Task_ID")["Assigned_Employee_ID"].to_dict()

    original_rows = []
    optimized_rows = []
    for task_id in tasks:
        original_employee = original_assignment_map[task_id]
        optimized_employee = optimized_assignment_map[task_id]

        original_pair = (task_id, original_employee)
        optimized_pair = (task_id, optimized_employee)

        original_rows.append(
            {
                "Task_ID": task_id,
                "Employee_ID": original_employee,
                "Assigned_Minutes": estimated_time_map[task_id],
                "Predicted_Delay_Risk": risk_map.get(original_pair, np.nan),
                "Estimated_Cost": cost_map.get(original_pair, np.nan),
            }
        )
        optimized_rows.append(
            {
                "Task_ID": task_id,
                "Employee_ID": optimized_employee,
                "Assigned_Minutes": estimated_time_map[task_id],
                "Predicted_Delay_Risk": risk_map.get(optimized_pair, np.nan),
                "Estimated_Cost": cost_map.get(optimized_pair, np.nan),
            }
        )

    before_assignment_df = pd.DataFrame(original_rows)
    after_assignment_df = pd.DataFrame(optimized_rows)

    employee_comparison = candidate_employees[["Assigned_Employee_ID", "Current_Load_Minutes"]].copy()
    before_employee_minutes = before_assignment_df.groupby("Employee_ID")["Assigned_Minutes"].sum()
    after_employee_minutes = after_assignment_df.groupby("Employee_ID")["Assigned_Minutes"].sum()

    employee_comparison["Before_Assigned_Minutes"] = employee_comparison["Assigned_Employee_ID"].map(before_employee_minutes).fillna(0)
    employee_comparison["After_Assigned_Minutes"] = employee_comparison["Assigned_Employee_ID"].map(after_employee_minutes).fillna(0)
    employee_comparison["Before_Total_Workload"] = (
        employee_comparison["Current_Load_Minutes"]
        + employee_comparison["Before_Assigned_Minutes"]
    )
    employee_comparison["After_Total_Workload"] = (
        employee_comparison["Current_Load_Minutes"]
        + employee_comparison["After_Assigned_Minutes"]
    )
    employee_comparison["Planning_Capacity_Minutes"] = employee_comparison[
        "Assigned_Employee_ID"
    ].map(planning_capacity)

    overload_threshold = target_total_workload * 1.15
    before_metrics = {
        "average_workload": float(employee_comparison["Before_Total_Workload"].mean()),
        "workload_standard_deviation": float(
            employee_comparison["Before_Total_Workload"].std(ddof=0)
        ),
        "average_predicted_delay_risk": float(
            before_assignment_df["Predicted_Delay_Risk"].mean()
        ),
        "total_predicted_delay_risk": float(
            before_assignment_df["Predicted_Delay_Risk"].sum()
        ),
        "total_cost": float(before_assignment_df["Estimated_Cost"].sum()),
        "number_of_overloaded_employees": int(
            (employee_comparison["Before_Total_Workload"] > overload_threshold).sum()
        ),
    }
    after_metrics = {
        "average_workload": float(employee_comparison["After_Total_Workload"].mean()),
        "workload_standard_deviation": float(
            employee_comparison["After_Total_Workload"].std(ddof=0)
        ),
        "average_predicted_delay_risk": float(
            after_assignment_df["Predicted_Delay_Risk"].mean()
        ),
        "total_predicted_delay_risk": float(
            after_assignment_df["Predicted_Delay_Risk"].sum()
        ),
        "total_cost": float(after_assignment_df["Estimated_Cost"].sum()),
        "number_of_overloaded_employees": int(
            (employee_comparison["After_Total_Workload"] > overload_threshold).sum()
        ),
    }

    optimized_assignments = tasks_to_assign[
        [
            "Task_ID",
            "Department",
            "Task_Type",
            "Priority_Level",
            "Estimated_Time_Minutes",
            "Assigned_Employee_ID",
        ]
    ].copy()
    optimized_assignments = optimized_assignments.rename(
        columns={"Assigned_Employee_ID": "Original_Employee_ID"}
    )
    optimized_assignments["Optimized_Employee_ID"] = optimized_assignments["Task_ID"].map(
        optimized_assignment_map
    )
    optimized_assignments["Original_Predicted_Delay_Risk"] = optimized_assignments.apply(
        lambda row: risk_map.get((row["Task_ID"], row["Original_Employee_ID"]), np.nan),
        axis=1,
    )
    optimized_assignments["Optimized_Predicted_Delay_Risk"] = optimized_assignments.apply(
        lambda row: risk_map.get((row["Task_ID"], row["Optimized_Employee_ID"]), np.nan),
        axis=1,
    )
    optimized_assignments["Original_Estimated_Cost"] = optimized_assignments.apply(
        lambda row: cost_map.get((row["Task_ID"], row["Original_Employee_ID"]), np.nan),
        axis=1,
    )
    optimized_assignments["Optimized_Estimated_Cost"] = optimized_assignments.apply(
        lambda row: cost_map.get((row["Task_ID"], row["Optimized_Employee_ID"]), np.nan),
        axis=1,
    )
    optimized_assignments["Risk_Improvement"] = (
        optimized_assignments["Original_Predicted_Delay_Risk"]
        - optimized_assignments["Optimized_Predicted_Delay_Risk"]
    )
    optimized_assignments["Cost_Change"] = (
        optimized_assignments["Optimized_Estimated_Cost"]
        - optimized_assignments["Original_Estimated_Cost"]
    )

    optimized_assignments.to_csv(output_root / "optimized_assignments.csv", index=False)
    employee_comparison.to_csv(reports_dir / "employee_workload_comparison.csv", index=False)

    summary_lines = [
        "Optimization Model Summary",
        "=" * 60,
        f"Tasks optimized: {len(tasks)}",
        f"Candidate employees: {len(employees)}",
        f"Department compatibility enforced: {enforce_department_compatibility}",
        f"Total predicted delay risk before: {before_metrics['total_predicted_delay_risk']:.4f}",
        f"Total predicted delay risk after: {after_metrics['total_predicted_delay_risk']:.4f}",
        f"Total cost before: ${before_metrics['total_cost']:.2f}",
        f"Total cost after: ${after_metrics['total_cost']:.2f}",
        "Model objective: minimize predicted delay risk, workload imbalance, and cost.",
    ]
    (reports_dir / "optimization_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    return {
        "tasks_to_assign": tasks_to_assign,
        "candidate_employees": candidate_employees,
        "assignment_options": assignment_options,
        "optimized_assignments": optimized_assignments,
        "employee_comparison": employee_comparison,
        "before_assignment_df": before_assignment_df,
        "after_assignment_df": after_assignment_df,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "solver_status": status,
        "target_total_workload": target_total_workload,
    }
