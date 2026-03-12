"""Utility helpers shared across the workflow optimization project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "workflow_data.csv"
DEFAULT_RANDOM_STATE = 42


def ensure_directories(paths: Iterable[Path | str]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers and return NaN when division is not possible."""
    if denominator in (0, None) or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def resolve_data_path(custom_path: str | None = None) -> Path:
    """Resolve the dataset path, with a fallback to the shipped CSV file name."""
    if custom_path:
        candidate = Path(custom_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found at: {candidate}")

    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH

    fallback_path = PROJECT_ROOT / "data" / "AI_Workflow_Optimization_Dataset_2500_Rows_v1.csv"
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "No dataset was found. Place the CSV file at data/workflow_data.csv."
    )


def save_text_report(report_text: str, output_path: Path | str) -> None:
    """Save a plain text report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def set_plot_style() -> None:
    """Apply a clean matplotlib style used across the project."""
    try:
        import matplotlib.pyplot as plt

        plt.style.use("tableau-colorblind10")
    except OSError:
        import matplotlib.pyplot as plt

        plt.style.use("ggplot")


def format_percentage(value: float) -> str:
    """Format a decimal value as a percentage string."""
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"
