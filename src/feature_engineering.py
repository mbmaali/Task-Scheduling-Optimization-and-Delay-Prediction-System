"""Feature engineering helpers for workflow analytics and modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import safe_divide


DAY_NAME_MAPPING = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def create_derived_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns used in EDA, prediction, and optimization."""
    engineered_df = dataframe.copy()

    engineered_df["Calculated_Delay_Minutes"] = (
        engineered_df["Actual_Time_Minutes"] - engineered_df["Estimated_Time_Minutes"]
    )

    engineered_df["Is_Delayed_Binary"] = engineered_df["Delay_Flag"].apply(
        lambda value: 1 if float(value) > 0 else 0
    )

    duration_minutes = (
        engineered_df["Task_End_Time"] - engineered_df["Task_Start_Time"]
    ).dt.total_seconds() / 60.0
    engineered_df["Task_Duration_From_Timestamps"] = duration_minutes.where(
        duration_minutes >= 0, np.nan
    )

    engineered_df["Start_Hour"] = engineered_df["Task_Start_Time"].dt.hour
    engineered_df["Start_DayOfWeek"] = engineered_df["Task_Start_Time"].dt.dayofweek
    engineered_df["Start_Day_Name"] = engineered_df["Start_DayOfWeek"].map(DAY_NAME_MAPPING)

    engineered_df["Task_Efficiency"] = engineered_df.apply(
        lambda row: safe_divide(
            row["Estimated_Time_Minutes"], row["Actual_Time_Minutes"]
        ),
        axis=1,
    )

    engineered_df["Cost_Per_Minute"] = engineered_df.apply(
        lambda row: safe_divide(row["Cost_Per_Task"], row["Actual_Time_Minutes"]),
        axis=1,
    )

    engineered_df["Positive_Delay_Minutes"] = engineered_df["Calculated_Delay_Minutes"].clip(
        lower=0
    )

    return engineered_df


def get_model_feature_columns() -> list[str]:
    """Return the feature columns used by the delay prediction model."""
    return [
        "Task_Type",
        "Priority_Level",
        "Department",
        "Approval_Level",
        "Employee_Workload",
        "Estimated_Time_Minutes",
        "Cost_Per_Task",
        "Start_Hour",
        "Start_DayOfWeek",
    ]
