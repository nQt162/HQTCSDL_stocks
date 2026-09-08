from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent

DEFAULT_FEATURES_ALL_DATABASE = "stock"
DEFAULT_FEATURES_ALL_TABLE = "features_all"
DEFAULT_FEATURE_CSV = MODEL_DIR / "output_model5" / "risk_features.csv"
RISK_DROP_THRESHOLD = -0.05

# These columns are expected to already exist in stock.features_all. Model 5
# only adds the 5-session target and binary risk label on top of them.
TECHNICAL_FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma20",
    "ma5_vs_ma20",
    "volatility_5d",
    "volatility_20d",
    "volatility_change",
    "rolling_max_20d",
    "drawdown_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range",
    "body_ratio",
    "close_position",
]
FEATURE_COLUMNS = ["encode_sector", *TECHNICAL_FEATURE_COLUMNS]

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

FEATURES_ALL_COLUMNS = [
    "trading_date",
    "symbol",
    "encode_sector",
    *PRICE_COLUMNS,
    *TECHNICAL_FEATURE_COLUMNS,
]

FEATURE_TABLE_COLUMNS = [
    "trading_date",
    "symbol",
    "encode_sector",
    *PRICE_COLUMNS,
    *TECHNICAL_FEATURE_COLUMNS,
    "future_return_5d",
    "risk_drop_label",
    "created_at",
]


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _empty_features_all_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURES_ALL_COLUMNS)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def normalize_features_all(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ClickHouse stock.features_all rows to the model 5 schema."""
    if df.empty:
        print("[risk_features] Input features_all dataframe is empty.")
        return _empty_features_all_frame()

    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "trading_date" not in normalized.columns and "date" in normalized.columns:
        normalized = normalized.rename(columns={"date": "trading_date"})
    if "volume" not in normalized.columns and "volumn" in normalized.columns:
        normalized = normalized.rename(columns={"volumn": "volume"})

    required_identity_columns = {"trading_date", "symbol", "close"}
    missing_identity_columns = sorted(
        required_identity_columns - set(normalized.columns)
    )
    if missing_identity_columns:
        print(
            "[risk_features] Missing required features_all columns: "
            f"{missing_identity_columns}"
        )
        return _empty_features_all_frame()

    missing_model_columns = sorted(set(FEATURES_ALL_COLUMNS) - set(normalized.columns))
    if missing_model_columns:
        print(
            "[risk_features] Missing model feature columns, filling nulls: "
            f"{missing_model_columns}"
        )
        for column in missing_model_columns:
            normalized[column] = pd.NA

    normalized["symbol"] = (
        normalized["symbol"].astype(str).str.strip().str.upper()
    )
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"], errors="coerce"
    ).dt.normalize()

    for column in [*PRICE_COLUMNS, *FEATURE_COLUMNS]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["symbol", "trading_date", "close"])
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(
        subset=["symbol", "trading_date"], keep="last"
    )
    normalized = normalized[FEATURES_ALL_COLUMNS]
    normalized = normalized.sort_values(["symbol", "trading_date"]).reset_index(
        drop=True
    )
    return normalized


def load_features_all(
    client: Any,
    database: str = DEFAULT_FEATURES_ALL_DATABASE,
    table: str = DEFAULT_FEATURES_ALL_TABLE,
) -> pd.DataFrame:
    
    query = f"""
        SELECT *
        FROM {quote_identifier(database)}.{quote_identifier(table)}
        ORDER BY symbol, trading_date
    """
    print(f"[risk_features] Loading features from ClickHouse: {database}.{table}")
    normalized = normalize_features_all(client.query_df(query))
    if normalized.empty:
        print("[risk_features] No features_all rows available after normalization.")
        return normalized

    print(
        "[risk_features] Loaded features_all "
        f"{len(normalized):,} rows, "
        f"{normalized['symbol'].nunique():,} symbols, "
        f"{normalized['trading_date'].min().date()} -> "
        f"{normalized['trading_date'].max().date()}"
    )
    return normalized


def create_risk_features(features_all_df: pd.DataFrame) -> pd.DataFrame:
    """Create model 5 labels from features_all.

    The resulting dataframe is the same semantic table as the previous
    risk_features output: common features plus future_return_5d and
    risk_drop_label.
    """
    features = normalize_features_all(features_all_df)
    if features.empty:
        print("[risk_features] No rows available to create model 5 labels.")
        for column in [
            "future_close_5d",
            "target_date",
            "future_return_5d",
            "risk_drop_label",
            "created_at",
        ]:
            if column not in features.columns:
                features[column] = pd.NA
        return features

    group = features.groupby("symbol", group_keys=False)
    features["future_close_5d"] = group["close"].shift(-5)
    features["target_date"] = group["trading_date"].shift(-5)
    features["future_return_5d"] = _safe_divide(
        features["future_close_5d"], features["close"]
    ) - 1
    features["risk_drop_label"] = (
        features["future_return_5d"] <= RISK_DROP_THRESHOLD
    ).astype("Int64")
    features.loc[features["future_return_5d"].isna(), "risk_drop_label"] = pd.NA
    features = features.replace([np.inf, -np.inf], np.nan)
    features["created_at"] = pd.Timestamp.now().floor("s")

    trainable_rows = features.dropna(
        subset=FEATURE_COLUMNS + ["risk_drop_label", "target_date"]
    ).shape[0]
    high_risk_rows = int((features["risk_drop_label"] == 1).sum())
    print(
        "[risk_features] Created model 5 labels from features_all. "
        f"Rows={len(features):,}, "
        f"trainable_rows={trainable_rows:,}, "
        f"HIGH_RISK labels={high_risk_rows:,}"
    )
    return features


def save_risk_features_csv(
    df: pd.DataFrame,
    output_path: Path | str = DEFAULT_FEATURE_CSV,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_columns = sorted(set(FEATURE_TABLE_COLUMNS) - set(df.columns))
    if missing_columns:
        print(f"[risk_features] Missing feature columns, filling nulls: {missing_columns}")
        for column in missing_columns:
            df[column] = pd.NA

    output = df[FEATURE_TABLE_COLUMNS].copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.date
    output.to_csv(output_path, index=False)
    print(f"[risk_features] Saved feature CSV: {output_path} ({len(output):,} rows)")
    return output_path


def create_clickhouse_feature_table(
    client: Any,
    database: str = "stock",
    table: str = "dw_stock_risk_features",
) -> None:
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{table}
        (
            trading_date Date,
            symbol String,
            encode_sector Nullable(Int32),
            open Nullable(Float64),
            high Nullable(Float64),
            low Nullable(Float64),
            close Nullable(Float64),
            volume Nullable(Float64),
            return_1d Nullable(Float64),
            return_3d Nullable(Float64),
            return_5d Nullable(Float64),
            return_10d Nullable(Float64),
            return_20d Nullable(Float64),
            ma_5 Nullable(Float64),
            ma_20 Nullable(Float64),
            ma_50 Nullable(Float64),
            price_vs_ma20 Nullable(Float64),
            ma5_vs_ma20 Nullable(Float64),
            volatility_5d Nullable(Float64),
            volatility_20d Nullable(Float64),
            volatility_change Nullable(Float64),
            rolling_max_20d Nullable(Float64),
            drawdown_20d Nullable(Float64),
            volume_ma_5 Nullable(Float64),
            volume_ma_20 Nullable(Float64),
            volume_ratio_5_20 Nullable(Float64),
            volume_change_1d Nullable(Float64),
            daily_range Nullable(Float64),
            body_ratio Nullable(Float64),
            close_position Nullable(Float64),
            future_return_5d Nullable(Float64),
            risk_drop_label Nullable(UInt8),
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (symbol, trading_date)
        """
    )


def save_risk_features(
    client: Any,
    df: pd.DataFrame,
    database: str = "stock",
    table: str = "dw_stock_risk_features",
) -> None:
    """Future ClickHouse writer. Not used by the current local CSV pipeline."""
    create_clickhouse_feature_table(client, database=database, table=table)
    output = df[FEATURE_TABLE_COLUMNS].copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.date
    output["created_at"] = pd.to_datetime(output["created_at"])
    output = output.where(pd.notna(output), None)
    client.insert_df(table=f"{database}.{table}", df=output)
    print(f"[risk_features] Inserted {len(output):,} rows into {database}.{table}")
