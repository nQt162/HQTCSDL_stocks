# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: create_marts.py
# Mục đích: Tạo các bảng mart phục vụ visualization
# ==============================================

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
MODEL_DIR       = Path(__file__).resolve().parent
PREDICTIONS_CSV = MODEL_DIR / "output" / "benchmark_predictions.csv"
METRICS_JSON    = MODEL_DIR / "output" / "benchmark_metrics.json"
DATABASE        = "stock"

load_dotenv(PROJECT_ROOT / ".env")

# ==================
# KẾT NỐI
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
# ĐỌC DỮ LIỆU
# ==================
def load_data(client):
    print("[model4] Đọc predictions từ ClickHouse...")
    df = client.query_df(f"""
        SELECT
            symbol,
            trading_date,
            close,
            label,
            predicted_label,
            outperform_probability,
            prediction_correct
        FROM {DATABASE}.model4_benchmark_predictions
        ORDER BY trading_date DESC, outperform_probability DESC
    """)
    sector_df = client.query_df(f"""
        SELECT symbol, encode_sector
        FROM {DATABASE}.symbol_sector_encoding
    """)
    df = df.merge(sector_df, on="symbol", how="left")
    print(f"[model4] Đọc xong: {len(df):,} dòng")
    return df

# ==================
# MART 1: TOP OUTPERFORMERS
# ==================
def create_mart_top_outperformers(client, df):
    print("[model4] Tạo mart_model4_top_outperformers...")
    latest_date = df["trading_date"].max()
    top_df = (
        df[df["trading_date"] == latest_date]
        .sort_values("outperform_probability", ascending=False)
        .head(20)
        [["symbol", "trading_date", "close",
          "outperform_probability", "predicted_label"]]
        .copy()
    )
    top_df["rank"]       = range(1, len(top_df) + 1)
    top_df["created_at"] = pd.Timestamp.now().floor("s")

    client.command(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.mart_model4_top_outperformers
        (
            rank                    UInt16,
            symbol                  String,
            trading_date            DateTime,
            close                   Float64,
            outperform_probability  Float64,
            predicted_label         Int8,
            created_at              DateTime
        )
        ENGINE = MergeTree
        ORDER BY (trading_date, rank)
    """)
    client.command(f"TRUNCATE TABLE {DATABASE}.mart_model4_top_outperformers")
    client.insert_df(
        f"{DATABASE}.mart_model4_top_outperformers",
        top_df[["rank", "symbol", "trading_date", "close",
                "outperform_probability", "predicted_label", "created_at"]]
    )
    print(f"[model4] Đã upload {len(top_df)} mã top outperform!")

# ==================
# MART 2: DAILY SUMMARY
# ==================
def create_mart_daily_summary(client, df):
    print("[model4] Tạo mart_model4_daily_outperform_summary...")
    summary_df = (
        df.groupby("trading_date")
        .agg(
            total_symbols        = ("symbol", "count"),
            predicted_outperform = ("predicted_label", "sum"),
            avg_probability      = ("outperform_probability", "mean"),
            actual_outperform    = ("label", "sum"),
            correct_predictions  = ("prediction_correct", "sum"),
        )
        .reset_index()
    )
    summary_df["outperform_ratio"] = (
        summary_df["predicted_outperform"] / summary_df["total_symbols"]
    )
    summary_df["accuracy_daily"] = (
        summary_df["correct_predictions"] / summary_df["total_symbols"]
    )
    summary_df["created_at"] = pd.Timestamp.now().floor("s")

    client.command(f"""
        CREATE TABLE IF NOT EXISTS
        {DATABASE}.mart_model4_daily_outperform_summary
        (
            trading_date         DateTime,
            total_symbols        UInt32,
            predicted_outperform UInt32,
            avg_probability      Float64,
            actual_outperform    UInt32,
            correct_predictions  UInt32,
            outperform_ratio     Float64,
            accuracy_daily       Float64,
            created_at           DateTime
        )
        ENGINE = MergeTree
        ORDER BY trading_date
    """)
    client.command(
        f"TRUNCATE TABLE {DATABASE}.mart_model4_daily_outperform_summary"
    )
    client.insert_df(
        f"{DATABASE}.mart_model4_daily_outperform_summary",
        summary_df
    )
    print(f"[model4] Đã upload {len(summary_df)} ngày!")

# ==================
# MART 3: SECTOR OUTPERFORM
# ==================
def create_mart_sector_outperform(client, df):
    print("[model4] Tạo mart_model4_sector_outperform...")
    sector_df = (
        df.dropna(subset=["encode_sector"])
        .groupby("encode_sector")
        .agg(
            total_symbols        = ("symbol", "count"),
            predicted_outperform = ("predicted_label", "sum"),
            actual_outperform    = ("label", "sum"),
            avg_probability      = ("outperform_probability", "mean"),
            correct_predictions  = ("prediction_correct", "sum"),
        )
        .reset_index()
    )
    sector_df["outperform_ratio"] = (
        sector_df["predicted_outperform"] / sector_df["total_symbols"]
    )
    sector_df["accuracy_sector"] = (
        sector_df["correct_predictions"] / sector_df["total_symbols"]
    )
    sector_df["encode_sector"] = sector_df["encode_sector"].astype(int)
    sector_df["created_at"]    = pd.Timestamp.now().floor("s")

    client.command(f"""
        CREATE TABLE IF NOT EXISTS
        {DATABASE}.mart_model4_sector_outperform
        (
            encode_sector        Int32,
            total_symbols        UInt32,
            predicted_outperform UInt32,
            actual_outperform    UInt32,
            avg_probability      Float64,
            correct_predictions  UInt32,
            outperform_ratio     Float64,
            accuracy_sector      Float64,
            created_at           DateTime
        )
        ENGINE = MergeTree
        ORDER BY encode_sector
    """)
    client.command(f"TRUNCATE TABLE {DATABASE}.mart_model4_sector_outperform")
    client.insert_df(f"{DATABASE}.mart_model4_sector_outperform", sector_df)
    print(f"[model4] Đã upload {len(sector_df)} sectors!")

# ==================
# MART 4: METRICS
# ==================
def create_mart_metrics(client):
    print("[model4] Tạo mart_model4_metrics...")
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics_df = pd.DataFrame([{
        "run_at":      pd.Timestamp.now().floor("s"),
        "accuracy":    metrics["accuracy"],
        "precision":   metrics["precision"],
        "recall":      metrics["recall"],
        "f1":          metrics["f1"],
        "roc_auc":     metrics["roc_auc"],
        "test_rows":   metrics["test_rows"],
        "train_ratio": metrics["train_ratio"],
    }])
    client.command(f"DROP TABLE IF EXISTS {DATABASE}.mart_model4_metrics")
    client.command(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.mart_model4_metrics
        (
            run_at      DateTime,
            accuracy    Float64,
            precision   Float64,
            recall      Float64,
            f1          Float64,
            roc_auc     Float64,
            test_rows   UInt32,
            train_ratio Float64
        )
        ENGINE = MergeTree
        ORDER BY run_at
    """)
    # Không TRUNCATE — giữ lại lịch sử mỗi lần chạy!
    client.insert_df(f"{DATABASE}.mart_model4_metrics", metrics_df)
    print(f"[model4] Đã upload metrics lúc {metrics_df['run_at'].iloc[0]}!")

# ==================
# MAIN
# ==================
def main():
    print("[model4] Kết nối ClickHouse...")
    client = get_client()
    print("[model4] Kết nối thành công!")

    df = load_data(client)

    create_mart_top_outperformers(client, df)
    create_mart_daily_summary(client, df)
    create_mart_sector_outperform(client, df)
    create_mart_metrics(client)

    print("\n[model4] TẤT CẢ MART ĐÃ ĐƯỢC TẠO! ✅")
    print(f"""
Các bảng đã upload lên ClickHouse:
  ✅ {DATABASE}.mart_model4_top_outperformers
  ✅ {DATABASE}.mart_model4_daily_outperform_summary
  ✅ {DATABASE}.mart_model4_sector_outperform
  ✅ {DATABASE}.mart_model4_metrics
    """)

if __name__ == "__main__":
    main()