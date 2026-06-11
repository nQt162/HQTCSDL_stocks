# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: train_benchmark_model.py
# Mục đích: Đọc từ stock.stock_prices (ClickHouse),
#           tính features + label, train LightGBM
# ==============================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import clickhouse_connect
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# ==================
# PATHS & CONFIG
# ==================
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
MODEL_DIR      = Path(__file__).resolve().parent
MODEL_OUTPUT   = MODEL_DIR / "output"
MODEL_SAVE_DIR = MODEL_DIR / "models"
MODEL_PATH     = MODEL_SAVE_DIR / "benchmark_outperformance_lgbm.pkl"

load_dotenv(PROJECT_ROOT / ".env")

HORIZON     = 5
TRAIN_RATIO = 0.8

FEATURE_COLUMNS = [
    "encode_sector",
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "ma_5", "ma_20", "ma_50",
    "price_vs_ma20", "ma5_vs_ma20",
    "volatility_5d", "volatility_20d", "volatility_change",
    "rolling_max_20d", "drawdown_20d",
    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",
    "daily_range", "body_ratio", "close_position",
]

# ==================
# 1. KẾT NỐI DB
# ==================
def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT") or "8443"),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE", "default"),
        secure=os.getenv("CLICKHOUSE_SECURE", "true").strip().lower()
        in {"1", "true", "yes"},
    )

# ==================
# 2. ĐỌC DỮ LIỆU
# ==================
def load_stock_prices(client) -> pd.DataFrame:
    print("[model4] Đọc dữ liệu từ stock.stock_prices...")
    df = client.query_df("""
        SELECT
            symbol,
            date as trading_date,
            open, high, low, close, volume
        FROM stock.stock_prices
        ORDER BY symbol, trading_date
    """)
    print(f"[model4] Đọc xong: {len(df):,} dòng, "
          f"{df['symbol'].nunique()} mã cổ phiếu")
    return df


def load_sector(client) -> pd.DataFrame:
    print("[model4] Đọc encode_sector...")
    return client.query_df("""
        SELECT symbol, encode_sector
        FROM stock.symbol_sector_encoding
    """)

# ==================
# 3. TÍNH BENCHMARK + LABEL
# ==================
def calc_benchmark_and_label(df: pd.DataFrame,
                              horizon: int = 5) -> pd.DataFrame:
    print(f"[model4] Tính benchmark return ({horizon} ngày)...")
    df = df.sort_values(["symbol", "trading_date"])
    df["future_return"] = (
        df.groupby("symbol")["close"]
        .pct_change(-horizon) * -1
    )
    benchmark = (
        df.groupby("trading_date")["future_return"]
        .mean()
        .rename("benchmark_return")
        .reset_index()
    )
    df = df.merge(benchmark, on="trading_date", how="left")
    df["label"] = (df["future_return"] > df["benchmark_return"]).astype(int)
    df = df.dropna(subset=["future_return", "benchmark_return"])
    label_counts = df["label"].value_counts()
    print(f"[model4] Label=1 (outperform): {label_counts.get(1, 0):,}")
    print(f"[model4] Label=0 (không outperform): {label_counts.get(0, 0):,}")
    return df

# ==================
# 4. TÍNH FEATURES
# ==================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[model4] Tính features kỹ thuật...")
    g = df.groupby("symbol")

    for window in [1, 3, 5, 10, 20]:
        df[f"return_{window}d"] = g["close"].pct_change(window)

    for window in [5, 20, 50]:
        df[f"ma_{window}"] = g["close"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )

    df["price_vs_ma20"] = (df["close"] / df["ma_20"]) - 1
    df["ma5_vs_ma20"]   = (df["ma_5"]  / df["ma_20"]) - 1

    df["volatility_5d"]  = g["return_1d"].transform(
        lambda s: s.rolling(5,  min_periods=5).std()
    )
    df["volatility_20d"] = g["return_1d"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    df["volatility_change"] = (df["volatility_5d"] / df["volatility_20d"]) - 1

    df["rolling_max_20d"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).max()
    )
    df["drawdown_20d"] = (df["close"] / df["rolling_max_20d"]) - 1

    df["volume_ma_5"]  = g["volume"].transform(
        lambda s: s.rolling(5,  min_periods=5).mean()
    )
    df["volume_ma_20"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    df["volume_ratio_5_20"] = df["volume_ma_5"] / df["volume_ma_20"]
    df["volume_change_1d"]  = g["volume"].pct_change(1)

    hl_range = df["high"] - df["low"]
    df["daily_range"]    = hl_range / df["close"]
    df["body_ratio"]     = (
        (df["close"] - df["open"]).abs() / hl_range
    ).where(hl_range > 0, 0)
    df["close_position"] = (
        (df["close"] - df["low"]) / hl_range
    ).where(hl_range > 0, 0.5)

    df = df.replace([np.inf, -np.inf], np.nan)
    print("[model4] Tính xong features!")
    return df

# ==================
# 5. LOAD FEATURES
# ==================
def load_features() -> pd.DataFrame:
    client      = get_client()
    df          = load_stock_prices(client)
    sector_df   = load_sector(client)
    df          = calc_benchmark_and_label(df, horizon=HORIZON)
    df          = create_features(df)
    df          = df.merge(sector_df, on="symbol", how="left")

    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "trading_date", "symbol"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["trading_date", "symbol"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No trainable rows after processing.")

    print(f"[model4] Trainable rows: {len(df):,}, "
          f"symbols={df['symbol'].nunique():,}, "
          f"{df['trading_date'].min().date()} -> "
          f"{df['trading_date'].max().date()}")
    return df

# ==================
# 6. TRAIN/TEST SPLIT
# ==================
def time_split(df: pd.DataFrame):
    unique_dates = sorted(df["trading_date"].dropna().unique())
    cutoff_idx   = int(len(unique_dates) * TRAIN_RATIO)
    cutoff_idx   = min(max(cutoff_idx, 1), len(unique_dates) - 1)
    cutoff_date  = pd.Timestamp(unique_dates[cutoff_idx])

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df  = df[df["trading_date"] >= cutoff_date].copy()

    print(f"[model4] Train: {len(train_df):,} rows "
          f"({train_df['trading_date'].min().date()} -> "
          f"{train_df['trading_date'].max().date()})")
    print(f"[model4] Test:  {len(test_df):,} rows "
          f"({test_df['trading_date'].min().date()} -> "
          f"{test_df['trading_date'].max().date()})")
    print(f"[model4] Cutoff date: {cutoff_date.date()}")
    return train_df, test_df

# ==================
# 7. TRAIN LIGHTGBM
# ==================
def train_model(train_df: pd.DataFrame):
    print("[model4] Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    print("[model4] Training done.")
    return model

# ==================
# 8. ĐÁNH GIÁ
# ==================
def evaluate_model(model, test_df: pd.DataFrame):
    print("[model4] Evaluating model...")
    y_pred = model.predict(test_df[FEATURE_COLUMNS])
    y_prob = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    y_test = test_df["label"]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob))
                     if y_test.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_ratio": TRAIN_RATIO,
        "test_rows":   int(len(test_df)),
        "feature_columns": FEATURE_COLUMNS,
    }

    print("\n" + "=" * 40)
    print("  MODEL4 EVALUATION")
    print("=" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    print("=" * 40 + "\n")
    return metrics, y_pred, y_prob

# ==================
# 9. LƯU KẾT QUẢ
# ==================
def save_model(model: Any, metrics: dict[str, Any]) -> Path:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "features": FEATURE_COLUMNS,
        "horizon": HORIZON, "target_type": "benchmark_outperformance",
        "train_ratio": TRAIN_RATIO, "metrics": metrics,
    }, MODEL_PATH)
    print(f"[model4] Saved model: {MODEL_PATH}")
    return MODEL_PATH


def save_results(model, metrics, test_df, y_pred, y_prob):
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    (MODEL_OUTPUT / "benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[model4] Saved metrics: {MODEL_OUTPUT / 'benchmark_metrics.json'}")

    predictions = test_df[["symbol", "trading_date", "close", "label"]].copy()
    predictions["predicted_label"]        = y_pred
    predictions["outperform_probability"] = y_prob
    predictions["prediction_correct"]     = (
        predictions["label"] == predictions["predicted_label"]
    )
    predictions.to_csv(MODEL_OUTPUT / "benchmark_predictions.csv", index=False)
    print(f"[model4] Saved predictions: {MODEL_OUTPUT / 'benchmark_predictions.csv'}")

    pd.DataFrame({
        "feature":    FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        MODEL_OUTPUT / "feature_importance.csv", index=False
    )
    print(f"[model4] Saved feature importance: {MODEL_OUTPUT / 'feature_importance.csv'}")

# ==================
# MAIN
# ==================
def main() -> None:
    df = load_features()
    train_df, test_df = time_split(df)
    model = train_model(train_df)
    metrics, y_pred, y_prob = evaluate_model(model, test_df)
    save_model(model, metrics)
    save_results(model, metrics, test_df, y_pred, y_prob)

    import subprocess, sys

    # Bước 1: upload predictions lên ClickHouse
    print("\n[model4] Chạy upload_predictions...")
    subprocess.run(
        [sys.executable, str(MODEL_DIR / "upload_predictions.py")],
        check=True
    )

    # Bước 2: tạo các mart
    print("\n[model4] Chạy create_marts...")
    subprocess.run(
        [sys.executable, str(MODEL_DIR / "create_marts.py")],
        check=True
    )

    print("[model4] Done.")


if __name__ == "__main__":
    main()