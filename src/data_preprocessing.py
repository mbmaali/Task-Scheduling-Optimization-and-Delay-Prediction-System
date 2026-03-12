"""Data loading and cleaning logic for workflow scheduling analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import resolve_data_path


EXPECTED_COLUMNS: List[str] = [
    "Workflow_ID",
    "Process_Name",
    "Task_ID",
    "Task_Type",
    "Priority_Level",
    "Department",
    "Assigned_Employee_ID",
    "Task_Start_Time",
    "Task_End_Time",
    "Estimated_Time_Minutes",
    "Actual_Time_Minutes",
    "Delay_Flag",
    "Approval_Level",
    "Employee_Workload",
    "Cost_Per_Task",
]

NUMERIC_COLUMNS: List[str] = [
    "Estimated_Time_Minutes",
    "Actual_Time_Minutes",
    "Employee_Workload",
    "Cost_Per_Task",
]

CATEGORICAL_COLUMNS: List[str] = [
    "Workflow_ID",
    "Process_Name",
    "Task_ID",
    "Task_Type",
    "Priority_Level",
    "Department",
    "Assigned_Employee_ID",
    "Approval_Level",
]

TIME_COLUMNS: List[str] = ["Task_Start_Time", "Task_End_Time"]


def normalize_delay_flag(value: object) -> float:
    """Convert different delay flag formats into 0 or 1."""
    if pd.isna(value):
        return np.nan

    value_as_text = str(value).strip().lower()
    positive_values = {"1", "yes", "true", "y", "delayed"}
    negative_values = {"0", "no", "false", "n", "on time", "ontime"}

    if value_as_text in positive_values:
        return 1.0
    if value_as_text in negative_values:
        return 0.0

    try:
        numeric_value = float(value)
        return 1.0 if numeric_value > 0 else 0.0
    except (TypeError, ValueError):
        return np.nan


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize incoming column names to the project's expected schema."""
    normalized_expected = {
        column.lower().replace(" ", "_"): column for column in EXPECTED_COLUMNS
    }

    renamed_columns = {}
    for column in dataframe.columns:
        normalized_column = column.strip().lower().replace(" ", "_")
        renamed_columns[column] = normalized_expected.get(normalized_column, column.strip())

    return dataframe.rename(columns=renamed_columns)


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise an error if mandatory dataset columns are missing."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns)
        )


def load_workflow_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the workflow dataset from CSV into a pandas DataFrame."""
    resolved_path = resolve_data_path(str(file_path) if file_path else None)
    dataframe = pd.read_csv(resolved_path)
    dataframe = standardize_column_names(dataframe)
    validate_required_columns(dataframe)
    return dataframe


def inspect_data_types(dataframe: pd.DataFrame) -> pd.Series:
    """Return the data types of each column for quick inspection."""
    return dataframe.dtypes


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple, beginner-friendly rules."""
    cleaned_df = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        if cleaned_df[column].isna().all():
            cleaned_df[column] = cleaned_df[column].fillna(0)
        else:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    cleaned_df["Delay_Flag"] = cleaned_df["Delay_Flag"].apply(normalize_delay_flag)

    # When the original delay flag is missing, estimate it from actual vs estimated time.
    derived_delay_flag = (
        cleaned_df["Actual_Time_Minutes"] > cleaned_df["Estimated_Time_Minutes"]
    ).astype(float)
    cleaned_df["Delay_Flag"] = cleaned_df["Delay_Flag"].fillna(derived_delay_flag)

    for column in CATEGORICAL_COLUMNS:
        cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    return cleaned_df


def convert_time_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert workflow timestamps into pandas datetime objects."""
    converted_df = dataframe.copy()
    for column in TIME_COLUMNS:
        converted_df[column] = pd.to_datetime(converted_df[column], errors="coerce")
    return converted_df


def clean_workflow_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to the raw workflow dataset."""
    cleaned_df = standardize_column_names(dataframe)
    validate_required_columns(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates().copy()
    cleaned_df = handle_missing_values(cleaned_df)
    cleaned_df = convert_time_columns(cleaned_df)

    # Remove rows that are missing the minimum information required for modeling.
    cleaned_df = cleaned_df.dropna(
        subset=["Task_ID", "Estimated_Time_Minutes", "Actual_Time_Minutes"]
    ).reset_index(drop=True)

    return cleaned_df


def generate_summary_statistics(dataframe: pd.DataFrame) -> Dict[str, object]:
    """Generate a compact summary used in the terminal and reports."""
    summary = {
        "number_of_rows": int(len(dataframe)),
        "number_of_unique_workflows": int(dataframe["Workflow_ID"].nunique()),
        "number_of_employees": int(dataframe["Assigned_Employee_ID"].nunique()),
        "missing_value_summary": dataframe.isna().sum().to_dict(),
        "delay_distribution": dataframe["Delay_Flag"].value_counts(dropna=False).to_dict(),
    }
    return summary


def print_data_summary(dataframe: pd.DataFrame) -> Dict[str, object]:
    """Print a beginner-friendly data summary and return it as a dictionary."""
    summary = generate_summary_statistics(dataframe)

    print("\nDATASET SUMMARY")
    print("-" * 60)
    print(f"Number of rows: {summary['number_of_rows']}")
    print(f"Unique workflows: {summary['number_of_unique_workflows']}")
    print(f"Unique employees: {summary['number_of_employees']}")
    print("\nMissing value summary:")
    print(pd.Series(summary["missing_value_summary"]).sort_values(ascending=False))
    print("\nDelayed vs non-delayed tasks:")
    print(pd.Series(summary["delay_distribution"]))

    return summary
