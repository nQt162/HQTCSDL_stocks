from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .risk_features import FEATURE_COLUMNS


DEFAULT_RISK_THRESHOLD = 0.6
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_JSON = MODEL_DIR / "output_model5" / "risk_metrics.json"
DEFAULT_MODEL_PATH = MODEL_DIR / "models" / "risk_alert_model.pkl"


def _prepare_training_data(feature_df: pd.DataFrame):
    required_columns = {
        "trading_date",
        "target_date",
        "symbol",
        "risk_drop_label",
        *FEATURE_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        print(f"[train] Missing feature columns, filling nulls: {missing_columns}")
        for column in missing_columns:
            feature_df[column] = pd.NA

    data = feature_df.copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"]).dt.normalize()
    data["target_date"] = pd.to_datetime(data["target_date"]).dt.normalize()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLUMNS + ["risk_drop_label", "target_date"])
    data["risk_drop_label"] = data["risk_drop_label"].astype(int)
    return data.sort_values(["trading_date", "symbol"]).reset_index(drop=True)


def _time_train_test_split(data: pd.DataFrame,train_ratio: float = 0.8,):
    unique_dates = pd.Series(data["trading_date"].unique()).sort_values().reset_index(
        drop=True
    )
    if len(unique_dates) < 10:
        print("[train] Not enough unique trading dates to split train/test by time.")
        return data.iloc[0:0].copy(), data.iloc[0:0].copy(), pd.NaT

    cutoff_index = int(len(unique_dates) * train_ratio)
    cutoff_index = min(max(cutoff_index, 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates.iloc[cutoff_index])

    train_df = data[data["trading_date"] < cutoff_date].copy()
    test_df = data[data["trading_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        print("[train] Train/test split produced an empty dataset.")
        return data.iloc[0:0].copy(), data.iloc[0:0].copy(), pd.NaT

    return train_df, test_df, cutoff_date


def _classification_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float,
):
    y_pred = (y_probability >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_high_risk": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_high_risk": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_high_risk": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_probability))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def _print_metrics(model_name: str, metrics: dict[str, Any]):
    print(f"\n[train] Metrics: {model_name}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_high_risk']:.4f} (HIGH_RISK)")
    print(f"  Recall:    {metrics['recall_high_risk']:.4f} (HIGH_RISK)")
    print(f"  F1-score:  {metrics['f1_high_risk']:.4f} (HIGH_RISK)")
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print(f"  PR-AUC:    {metrics['pr_auc']}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")


def _train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _train_xgboost_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[train] xgboost is not installed. Install it with: pip install xgboost")
        return None

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def train_models(
    feature_df: pd.DataFrame,
    train_ratio: float = 0.8,
    threshold: float = DEFAULT_RISK_THRESHOLD,
):
    """Train baseline and main risk-alert models using a time-based split."""
    data = _prepare_training_data(feature_df)
    if data.empty:
        print("[train] No trainable rows after filtering features and target.")
        return None, pd.DataFrame(), {
            "selected_model": None,
            "threshold": threshold,
            "train_ratio": train_ratio,
            "reason": "no_trainable_rows",
            "models": {},
            "feature_columns": FEATURE_COLUMNS,
        }

    if data["risk_drop_label"].nunique() < 2:
        print("[train] Training data must contain both risk classes 0 and 1.")
        return None, pd.DataFrame(), {
            "selected_model": None,
            "threshold": threshold,
            "train_ratio": train_ratio,
            "reason": "not_enough_classes",
            "models": {},
            "feature_columns": FEATURE_COLUMNS,
        }

    train_df, test_df, cutoff_date = _time_train_test_split(data, train_ratio)
    if train_df.empty or test_df.empty:
        return None, pd.DataFrame(), {
            "selected_model": None,
            "threshold": threshold,
            "train_ratio": train_ratio,
            "reason": "empty_train_test_split",
            "models": {},
            "feature_columns": FEATURE_COLUMNS,
        }

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["risk_drop_label"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["risk_drop_label"]

    print(
        "[train] Time split without shuffle: "
        f"train={len(train_df):,} rows "
        f"({train_df['trading_date'].min().date()} -> {train_df['trading_date'].max().date()}), "
        f"test={len(test_df):,} rows "
        f"({test_df['trading_date'].min().date()} -> {test_df['trading_date'].max().date()}), "
        f"cutoff={cutoff_date.date()}"
    )
    print(
        "[train] Class balance: "
        f"train_positive={int((y_train == 1).sum()):,}, "
        f"train_negative={int((y_train == 0).sum()):,}, "
        f"test_positive={int((y_test == 1).sum()):,}, "
        f"test_negative={int((y_test == 0).sum()):,}"
    )

    model_results: dict[str, dict[str, Any]] = {}

    logistic_model = _train_logistic_regression(x_train, y_train)
    logistic_probability = logistic_model.predict_proba(x_test)[:, 1]
    logistic_metrics = _classification_metrics(y_test, logistic_probability, threshold)
    model_results["logistic_regression_baseline"] = logistic_metrics
    _print_metrics("logistic_regression_baseline", logistic_metrics)

    xgboost_model = _train_xgboost_classifier(x_train, y_train)
    if xgboost_model is not None:
        xgboost_probability = xgboost_model.predict_proba(x_test)[:, 1]
        xgboost_metrics = _classification_metrics(y_test, xgboost_probability, threshold)
        model_results["xgboost_risk_alert"] = xgboost_metrics
        _print_metrics("xgboost_risk_alert", xgboost_metrics)
        selected_model = xgboost_model
        selected_model_name = "xgboost_risk_alert"
        selected_probability = xgboost_probability
    else:
        selected_model = logistic_model
        selected_model_name = "logistic_regression_baseline"
        selected_probability = logistic_probability

    predictions = test_df.copy()
    predictions["prediction_date"] = predictions["trading_date"]
    predictions["model_name"] = selected_model_name
    predictions["risk_probability"] = selected_probability.astype(float)
    predictions["risk_label"] = np.where(
        predictions["risk_probability"] >= threshold, "HIGH_RISK", "LOW_RISK"
    )
    predictions["created_at"] = pd.Timestamp.now().floor("s")

    report = {
        "selected_model": selected_model_name,
        "threshold": threshold,
        "train_ratio": train_ratio,
        "cutoff_date": str(cutoff_date.date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df["trading_date"].min().date()),
        "train_end": str(train_df["trading_date"].max().date()),
        "test_start": str(test_df["trading_date"].min().date()),
        "test_end": str(test_df["trading_date"].max().date()),
        "train_positive": int((y_train == 1).sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int((y_test == 1).sum()),
        "test_negative": int((y_test == 0).sum()),
        "models": model_results,
        "feature_columns": FEATURE_COLUMNS,
    }

    return selected_model, predictions, report


def save_model(
    model,
    metrics: dict[str, Any],
    output_path: Path | str = DEFAULT_MODEL_PATH,
):
    if model is None:
        print("[train] No model to save.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "selected_model": metrics.get("selected_model"),
            "threshold": metrics.get("threshold", DEFAULT_RISK_THRESHOLD),
            "target_type": "risk_drop_label",
            "metrics": metrics,
        },
        output_path,
    )
    print(f"[train] Saved model PKL: {output_path}")
    return output_path


def load_saved_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    saved = joblib.load(model_path)
    return saved["model"], saved.get("features", FEATURE_COLUMNS), saved


def save_metrics_json(
    metrics: dict[str, Any],
    output_path: Path | str = DEFAULT_METRICS_JSON,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] Saved metrics JSON: {output_path}")
    return output_path
