"""Delay prediction models for workflow tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import get_model_feature_columns
from src.utils import DEFAULT_RANDOM_STATE, ensure_directories, set_plot_style


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create an OneHotEncoder compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_preprocessor(
    categorical_features: List[str], numeric_features: List[str]
) -> ColumnTransformer:
    """Build a preprocessing pipeline for categorical and numeric variables."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _build_one_hot_encoder()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def get_model_candidates(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, object]:
    """Return the classification models used in the project."""
    candidates: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        pass

    return candidates


def prepare_model_data(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare the feature matrix and target vector for model training."""
    feature_columns = get_model_feature_columns()
    modeling_df = dataframe[feature_columns + ["Is_Delayed_Binary"]].copy()

    if modeling_df["Is_Delayed_Binary"].nunique() < 2:
        raise ValueError(
            "The delay target contains only one class. A classification model cannot be trained."
        )

    numeric_features = [
        "Employee_Workload",
        "Estimated_Time_Minutes",
        "Cost_Per_Task",
        "Start_Hour",
        "Start_DayOfWeek",
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    X = modeling_df[feature_columns]
    y = modeling_df["Is_Delayed_Binary"].astype(int)
    return X, y, categorical_features, numeric_features


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """Evaluate a trained classification pipeline on the test set."""
    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
        if probabilities is not None and len(np.unique(y_test)) > 1
        else np.nan,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }
    return metrics


def _save_confusion_matrix(
    confusion_values: np.ndarray,
    labels: List[str],
    model_name: str,
    figure_path: Path,
) -> None:
    """Save a confusion matrix chart for a trained model."""
    set_plot_style()
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_values, display_labels=labels)
    display.plot(ax=axis, colorbar=False)
    axis.set_title(f"Confusion Matrix - {model_name}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _create_feature_importance_report(
    trained_pipeline: Pipeline,
    model_name: str,
    figures_dir: Path,
    reports_dir: Path,
) -> pd.DataFrame:
    """Create a simple model interpretability report for the best model."""
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "Logistic Regression":
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.coef_[0],
            }
        ).sort_values("importance", ascending=False)
        plot_title = "Top Positive Logistic Regression Coefficients"
    else:
        importance_values = getattr(model, "feature_importances_", None)
        if importance_values is None:
            return pd.DataFrame(columns=["feature", "importance"])
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        ).sort_values("importance", ascending=False)
        plot_title = f"Feature Importance - {model_name}"

    importance_df.to_csv(reports_dir / "best_model_feature_importance.csv", index=False)

    top_features = importance_df.head(15).iloc[::-1]
    set_plot_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top_features["feature"], top_features["importance"], color="#4C78A8")
    axis.set_title(plot_title)
    axis.set_xlabel("Importance Value")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(figures_dir / "best_model_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    return importance_df


def train_delay_prediction_models(
    dataframe: pd.DataFrame,
    models_dir: str | Path,
    reports_dir: str | Path,
    figures_dir: str | Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    """Train multiple models, compare them, and save the best one."""
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    ensure_directories([models_dir, reports_dir, figures_dir])

    X, y, categorical_features, numeric_features = prepare_model_data(dataframe)
    preprocessor = get_preprocessor(categorical_features, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    metrics_records = []
    trained_pipelines: Dict[str, Pipeline] = {}
    evaluation_details: Dict[str, Dict[str, object]] = {}

    for model_name, model in get_model_candidates(random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        evaluation = evaluate_model(pipeline, X_test, y_test)

        metrics_records.append(
            {
                "model_name": model_name,
                "accuracy": evaluation["accuracy"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1_score": evaluation["f1_score"],
                "roc_auc": evaluation["roc_auc"],
            }
        )
        trained_pipelines[model_name] = pipeline
        evaluation_details[model_name] = evaluation

        safe_model_name = model_name.lower().replace(" ", "_")
        _save_confusion_matrix(
            evaluation["confusion_matrix"],
            ["On Time", "Delayed"],
            model_name,
            figures_dir / f"confusion_matrix_{safe_model_name}.png",
        )
        (reports_dir / f"classification_report_{safe_model_name}.txt").write_text(
            evaluation["classification_report"], encoding="utf-8"
        )

    metrics_df = pd.DataFrame(metrics_records).sort_values(
        by=["f1_score", "recall", "roc_auc"], ascending=False
    )
    metrics_df.to_csv(reports_dir / "model_comparison_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    model_bundle = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_columns": get_model_feature_columns(),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }
    joblib.dump(model_bundle, models_dir / "best_delay_prediction_model.joblib")

    _create_feature_importance_report(
        trained_pipeline=best_pipeline,
        model_name=best_model_name,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
    )

    summary_lines = [
        "Delay Prediction Model Summary",
        "=" * 50,
        f"Best model selected: {best_model_name}",
        f"Accuracy: {metrics_df.iloc[0]['accuracy']:.4f}",
        f"Precision: {metrics_df.iloc[0]['precision']:.4f}",
        f"Recall: {metrics_df.iloc[0]['recall']:.4f}",
        f"F1 Score: {metrics_df.iloc[0]['f1_score']:.4f}",
        f"ROC-AUC: {metrics_df.iloc[0]['roc_auc']:.4f}",
    ]
    (reports_dir / "best_model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "best_model_name": best_model_name,
        "best_model_bundle": model_bundle,
        "metrics_df": metrics_df,
        "evaluation_details": evaluation_details,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_model_bundle(model_path: str | Path) -> Dict[str, object]:
    """Load a saved model bundle from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_delay_probabilities(
    model_bundle: Dict[str, object], input_dataframe: pd.DataFrame
) -> np.ndarray:
    """Predict delay probabilities for one or more task records."""
    feature_columns = model_bundle["feature_columns"]
    prediction_frame = input_dataframe.copy()

    for column in feature_columns:
        if column not in prediction_frame.columns:
            prediction_frame[column] = np.nan

    prediction_frame = prediction_frame[feature_columns]
    probabilities = model_bundle["pipeline"].predict_proba(prediction_frame)[:, 1]
    return probabilities


def predict_delay_for_new_task(
    input_dict: Dict[str, object],
    model_path: str | Path = Path("outputs/models/best_delay_prediction_model.joblib"),
) -> Dict[str, float | int]:
    """Predict whether a new task will be delayed and return its probability."""
    model_bundle = load_model_bundle(model_path)
    input_dataframe = pd.DataFrame([input_dict])
    delay_probability = float(predict_delay_probabilities(model_bundle, input_dataframe)[0])
    predicted_class = int(delay_probability >= 0.5)

    return {
        "predicted_class": predicted_class,
        "delay_probability": delay_probability,
    }
