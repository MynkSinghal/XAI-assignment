"""Core utilities shared across the XAI Heart Disease project.

This module centralises the data acquisition, preprocessing, modelling,
evaluation, explainability, and fairness analysis logic so that notebooks,
Streamlit applications, and batch jobs can rely on a single source of truth.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from fairlearn.metrics import (
        MetricFrame,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )
except ImportError as exc:  # pragma: no cover - handled at runtime within notebooks
    raise ImportError(
        "fairlearn is required for fairness analysis. Install it via `pip install fairlearn`."
    ) from exc

# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORTS_DIR = PROJECT_ROOT / "reports"

ARTIFACT_FILENAMES = {
    "confusion_matrix": "confusion_matrix.png",
    "roc_curves": "roc_curves.png",
    "feature_importance": "feature_importance.png",
    "target_distribution": "target_distribution.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "fairness_selection_rate": "fairness_selection_rate.png",
    "shap_summary": "shap_beeswarm.png",
    "shap_bar": "shap_feature_importance.png",
}

MODEL_FILENAME = "heart_disease_best_model.joblib"
METADATA_FILENAME = "heart_disease_model_metadata.json"
METRICS_FILENAME = "model_performance.csv"
FAIRNESS_FILENAME = "fairness_metrics.csv"
SHAP_FILENAME = "shap_top_features.json"
DATA_FILENAME = "heart_disease.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "disease_present"
SENSITIVE_DEFAULT = "sex"

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------------------
# Directory + path management
# --------------------------------------------------------------------------------------


def get_project_paths() -> Dict[str, Path]:
    """Return the canonical project directories."""

    return {
        "project_root": PROJECT_ROOT,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "visuals": VISUALS_DIR,
        "reports": REPORTS_DIR,
        "model_path": MODELS_DIR / MODEL_FILENAME,
        "metadata_path": MODELS_DIR / METADATA_FILENAME,
        "metrics_csv": REPORTS_DIR / METRICS_FILENAME,
        "fairness_csv": REPORTS_DIR / FAIRNESS_FILENAME,
        "shap_json": REPORTS_DIR / SHAP_FILENAME,
        "raw_data": DATA_DIR / DATA_FILENAME,
    }


def get_default_artifact_paths() -> Dict[str, Path]:
    """Expose the default file paths used across the project."""

    return get_project_paths()



def ensure_project_directories() -> None:
    """Ensure all required project directories exist."""

    for directory in (DATA_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Data acquisition & preparation
# --------------------------------------------------------------------------------------


def load_or_fetch_data(refresh: bool = False) -> pd.DataFrame:
    """Load the cached dataset or download it from the UCI repository.

    Parameters
    ----------
    refresh: bool
        If True the dataset is re-downloaded from the source regardless of local cache.

    Returns
    -------
    pd.DataFrame
        The consolidated dataset including the target column.
    """

    ensure_project_directories()
    data_path = DATA_DIR / DATA_FILENAME

    if refresh or not data_path.exists():
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:  # pragma: no cover - dependency installed at runtime
            raise ImportError(
                "ucimlrepo is required to download the dataset. Install it via `pip install ucimlrepo`."
            ) from exc

        dataset = fetch_ucirepo(id=45)
        features = dataset.data.features
        targets = dataset.data.targets

        df = pd.concat([features, targets], axis=1)
        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
        if "num" in df.columns and TARGET_COLUMN not in df.columns:
            df = df.rename(columns={"num": TARGET_COLUMN})
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float).apply(lambda value: 1 if value > 0 else 0)
        df.replace("?", np.nan, inplace=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns and "num" in df.columns:
        df = df.rename(columns={"num": TARGET_COLUMN})
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float).apply(lambda value: 1 if value > 0 else 0)
    df.replace("?", np.nan, inplace=True)

    return df



def prepare_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and binary target."""

    working_df = df.copy()
    working_df.replace("?", np.nan, inplace=True)

    for column in NUMERIC_FEATURES + ["ca"]:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column in working_df.columns:
            working_df[column] = working_df[column].astype("category")

    if TARGET_COLUMN not in working_df.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' to be present.")

    target = working_df[TARGET_COLUMN].astype(int)

    selected_features = [
        column
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column in working_df.columns
    ]
    features = working_df[selected_features]

    return features, target



def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Perform a stratified train/test split to preserve class balance."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# --------------------------------------------------------------------------------------
# Modelling and evaluation
# --------------------------------------------------------------------------------------


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [col for col in NUMERIC_FEATURES if col in X.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _model_candidates(preprocessor: ColumnTransformer, random_state: int) -> Dict[str, Pipeline]:
    candidates = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        n_estimators=250,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    return candidates


def _compute_classification_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _extract_feature_importances(model: Pipeline) -> pd.Series:
    classifier = model.named_steps["classifier"]
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:  # pragma: no cover - unlikely with chosen estimators
        raise AttributeError("The classifier does not provide feature importances or coefficients.")

    return pd.Series(importances, index=feature_names)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_labels: Iterable[str],
    output_path: Path,
) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels).plot(
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    roc_payload: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, payload in roc_payload.items():
        ax.plot(payload["fpr"], payload["tpr"], label=f"{model_name} (AUC={payload['auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(
    importances: pd.Series,
    output_path: Path,
    top_n: int = 15,
) -> None:
    top_features = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
    ax.set_xlabel("Importance (absolute)")
    ax.set_ylabel("Features")
    ax.set_title("Top Feature Importances")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    visuals_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, str, Pipeline, Dict[str, Any]]:
    """Train candidate models and evaluate them, returning the best model and artefacts."""

    ensure_project_directories()
    visuals_dir = visuals_dir or VISUALS_DIR
    reports_dir = reports_dir or REPORTS_DIR

    preprocessor = _build_preprocessor(X_train)
    candidates = _model_candidates(preprocessor, random_state=random_state)

    metrics_records: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Pipeline] = {}
    roc_payload: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in candidates.items():
        fitted_model = pipeline.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)
        y_proba = fitted_model.predict_proba(X_test)[:, 1]

        metrics_records.append(
            _compute_classification_metrics(model_name, y_test, y_pred, y_proba)
        )
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_payload[model_name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, y_proba)}
        fitted_models[model_name] = fitted_model

    metrics_df = pd.DataFrame(metrics_records).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(reports_dir / METRICS_FILENAME, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    confusion = confusion_matrix(y_test, best_model.predict(X_test))
    _plot_confusion_matrix(
        confusion,
        class_labels=["No Disease", "Disease"],
        output_path=visuals_dir / ARTIFACT_FILENAMES["confusion_matrix"],
    )

    _plot_roc_curves(roc_payload, visuals_dir / ARTIFACT_FILENAMES["roc_curves"])

    importances = _extract_feature_importances(best_model)
    _plot_feature_importance(importances, visuals_dir / ARTIFACT_FILENAMES["feature_importance"])

    evaluation_payload = {
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_png": visuals_dir / ARTIFACT_FILENAMES["confusion_matrix"],
        "roc_curves_png": visuals_dir / ARTIFACT_FILENAMES["roc_curves"],
        "feature_importance_png": visuals_dir / ARTIFACT_FILENAMES["feature_importance"],
        "feature_importances": importances.sort_values(ascending=False),
        "roc_payload": roc_payload,
        "metrics_csv": reports_dir / METRICS_FILENAME,
        "feature_names": list(importances.index),
    }

    return metrics_df, best_model_name, best_model, evaluation_payload


# --------------------------------------------------------------------------------------
# Explainability & fairness
# --------------------------------------------------------------------------------------


def generate_shap_artifacts(
    best_model: Pipeline,
    X_reference: pd.DataFrame,
    visuals_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    sample_size: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate SHAP-based explainability artefacts for the best model."""

    import shap  # Imported lazily to avoid dependency when unused

    visuals_dir = visuals_dir or VISUALS_DIR
    reports_dir = reports_dir or REPORTS_DIR

    if len(X_reference) == 0:
        raise ValueError("X_reference must contain at least one record for SHAP analysis.")

    sample_size = min(sample_size, len(X_reference))
    X_sample = X_reference.sample(sample_size, random_state=random_state)

    preprocessor: ColumnTransformer = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif hasattr(classifier, "predict_proba"):
        explainer = shap.LinearExplainer(classifier, transformed)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    else:  # pragma: no cover - fallback
        explainer = shap.KernelExplainer(classifier.predict, transformed)
        shap_values = explainer.shap_values(transformed)

    shap_values = np.asarray(shap_values)

    shap.summary_plot(
        shap_values,
        features=transformed,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        color_bar=True,
    )
    shap_summary_path = visuals_dir / ARTIFACT_FILENAMES["shap_summary"]
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        features=transformed,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        color="#2d7fb8",
    )
    shap_bar_path = visuals_dir / ARTIFACT_FILENAMES["shap_bar"]
    plt.tight_layout()
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    shap_top = shap_importance.head(20)

    shap_json_path = reports_dir / SHAP_FILENAME
    shap_top.round(6).to_json(shap_json_path, orient="index", indent=2)

    return {
        "shap_values": shap_values,
        "feature_names": feature_names.tolist(),
        "summary_plot": shap_summary_path,
        "bar_plot": shap_bar_path,
        "importance": shap_importance,
        "top_importance_json": shap_json_path,
        "sample_size": sample_size,
    }



def evaluate_fairness(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_feature: str = SENSITIVE_DEFAULT,
    visuals_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assess fairness metrics across sensitive groups."""

    visuals_dir = visuals_dir or VISUALS_DIR
    reports_dir = reports_dir or REPORTS_DIR

    if sensitive_feature not in X_test.columns:
        raise KeyError(f"Sensitive feature '{sensitive_feature}' not present in features.")

    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    sensitive_series = X_test[sensitive_feature]

    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sensitive_series,
    )

    fairness_table = metric_frame.by_group.round(3)
    overall_metrics = {metric: metric_frame.overall[metric] for metric in fairness_table.columns}
    fairness_table.loc["overall"] = overall_metrics

    # Add ROC AUC per group
    roc_auc_by_group: Dict[Any, Optional[float]] = {}
    for group_value in sorted(sensitive_series.unique()):
        mask = sensitive_series == group_value
        if mask.sum() > 1:
            roc_auc_by_group[group_value] = float(roc_auc_score(y_test[mask], probs[mask]))
        else:  # pragma: no cover - insufficient data for group
            roc_auc_by_group[group_value] = np.nan
    roc_auc_by_group["overall"] = float(roc_auc_score(y_test, probs))
    fairness_table["roc_auc"] = fairness_table.index.map(roc_auc_by_group)
    fairness_table = fairness_table.round(3)

    disparities = pd.DataFrame(
        {
            "difference": metric_frame.difference(method="between_groups"),
            "ratio": metric_frame.ratio(method="between_groups"),
        }
    ).round(3)

    selection_plot_path = visuals_dir / ARTIFACT_FILENAMES["fairness_selection_rate"]
    group_table = fairness_table.drop(index="overall", errors="ignore").reset_index().rename(
        columns={"index": sensitive_feature}
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=group_table,
        x=sensitive_feature,
        y="selection_rate",
        hue=None,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Selection Rate by Sensitive Group")
    ax.set_ylabel("Selection Rate (Positive Predictions)")
    ax.set_xlabel(sensitive_feature.capitalize())
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    fig.savefig(selection_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fairness_csv_path = reports_dir / FAIRNESS_FILENAME
    fairness_table.to_csv(fairness_csv_path)

    return {
        "fairness_table": fairness_table,
        "disparities": disparities,
        "selection_plot": selection_plot_path,
        "fairness_csv": fairness_csv_path,
        "sensitive_feature": sensitive_feature,
    }


# --------------------------------------------------------------------------------------
# Persistence utilities & downstream helpers
# --------------------------------------------------------------------------------------


def serialize_artifacts(
    best_model: Pipeline,
    metrics_df: pd.DataFrame,
    fairness_payload: Dict[str, Any],
    evaluation_payload: Dict[str, Any],
    shap_payload: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    best_model_label: Optional[str] = None,
) -> Dict[str, Path]:
    """Persist trained artefacts for reuse in applications such as Streamlit dashboards."""

    ensure_project_directories()

    model_path = MODELS_DIR / MODEL_FILENAME
    metadata_path = MODELS_DIR / METADATA_FILENAME

    joblib.dump(best_model, model_path)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(model_path.resolve()),
        "best_model": best_model_label or best_model.named_steps["classifier"].__class__.__name__,
        "pipeline_type": best_model.__class__.__name__,
        "original_feature_columns": feature_columns or getattr(best_model, "feature_names_in_", []),
        "transformed_feature_names": evaluation_payload.get("feature_names", []),
        "metrics": metrics_df.to_dict(orient="records"),
        "fairness_overview": fairness_payload["fairness_table"].round(3).to_dict(),
        "fairness_disparities": fairness_payload["disparities"].round(3).to_dict(),
        "artefacts": {
            "confusion_matrix_png": str(evaluation_payload["confusion_matrix_png"].resolve()),
            "roc_curves_png": str(evaluation_payload["roc_curves_png"].resolve()),
            "feature_importance_png": str(evaluation_payload["feature_importance_png"].resolve()),
            "fairness_selection_png": str(fairness_payload["selection_plot"].resolve()),
            "shap_summary_png": str(shap_payload["summary_plot"].resolve()),
            "shap_bar_png": str(shap_payload["bar_plot"].resolve()),
            "metrics_csv": str(evaluation_payload["metrics_csv"].resolve()),
            "fairness_csv": str(fairness_payload["fairness_csv"].resolve()),
            "shap_top_features_json": str(shap_payload["top_importance_json"].resolve()),
        },
        "class_labels": {"0": "No heart disease", "1": "Heart disease present"},
        "sensitive_feature": fairness_payload["sensitive_feature"],
    }

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return {"model_path": model_path, "metadata_path": metadata_path}



def load_serialized_model(model_path: Optional[Path] = None) -> Pipeline:
    """Load the persisted best model."""

    model_path = model_path or (MODELS_DIR / MODEL_FILENAME)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run the training pipeline to generate it."
        )
    return joblib.load(model_path)



def generate_sample_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    n_samples: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a compact dataframe with sample predictions and associated probabilities."""

    if len(X) == 0:
        raise ValueError("Input feature set is empty; cannot generate sample predictions.")

    sample = X.sample(min(n_samples, len(X)), random_state=random_state)
    probabilities = model.predict_proba(sample)[:, 1]
    predictions = model.predict(sample)

    result = sample.copy()
    result["prediction"] = predictions
    result["probability_positive"] = np.round(probabilities, 4)

    return result.reset_index(drop=True)
