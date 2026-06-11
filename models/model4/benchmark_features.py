# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: benchmark_features.py
# Mục đích: Đọc dữ liệu từ ClickHouse,
#           tính features, gán label outperform,
#           lưu CSV để tham khảo/kiểm tra
# ==============================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

# ==================
# PATHS & CONFIG
# ==================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = Path(__file__).resolve().parent
OUTPUT_PATH  = MODEL_DIR / "output" / "benchmark_features.csv"

load_dotenv(PROJECT_ROOT / ".env")

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
    query = """
        SELECT
            symbol,
            date as trading_date,
            open, high, low, close, volume
        FROM stock.stock_prices
        ORDER BY symbol, date
    """
    df = client.query_df(query)
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
# 3. TÍNH BENCHMARK
# ==================
def calc_benchmark_return(df: pd.DataFrame,
                          horizon: int = 5):
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
    print(f"[model4] Tính xong benchmark: {len(benchmark)} ngày giao dịch")
    return df, benchmark

# ==================
# 4. GÁN LABEL
# ==================
def create_labels(df: pd.DataFrame,
                  benchmark: pd.DataFrame) -> pd.DataFrame:
    print("[model4] Gán nhãn outperform...")
    df = df.merge(benchmark, on="trading_date", how="left")
    df["label"] = (df["future_return"] > df["benchmark_return"]).astype(int)
    df = df.dropna(subset=["future_return", "benchmark_return"])
    label_counts = df["label"].value_counts()
    print(f"[model4] Label=1 (outperform): {label_counts.get(1, 0):,}")
    print(f"[model4] Label=0 (không outperform): {label_counts.get(0, 0):,}")
    return df

# ==================
# 5. TÍNH FEATURES
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
# MAIN
# ==================
if __name__ == "__main__":
    client = get_client()

    df          = load_stock_prices(client)
    sector_df   = load_sector(client)
    df, benchmark = calc_benchmark_return(df, horizon=5)
    df          = create_labels(df, benchmark)
    df          = create_features(df)

    # Merge encode_sector
    df = df.merge(sector_df, on="symbol", how="left")

    # Chỉ giữ dòng đủ features + label
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[model4] Đã lưu {len(df_clean):,} dòng vào {OUTPUT_PATH}")