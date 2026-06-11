from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

DEFAULT_DATABASE = os.getenv("UPLOAD_ALL_CLICKHOUSE_DATABASE", "stock")
DEFAULT_CHUNKSIZE = 100_000


@dataclass
class TableSpec:
    table: str
    layer: str
    paths: list[Path] = field(default_factory=list)
    glob_pattern: str | None = None
    order_by: list[str] = field(default_factory=list)
    transform: str | None = None
    model_name: str | None = None
    add_source_file: bool = False
    add_model_metadata: bool = False
    optional: bool = True
    description: str = ""


STRING_HINTS = {
    "symbol",
    "company_name",
    "sector",
    "model_name",
    "model_run_id",
    "feature",
    "source_file",
    "source_system",
    "file_checksum",
    "risk_label",
    "actual_risk_label",
    "predicted_risk_label",
    "signal",
    "predicted_signal",
    "adjusted_signal",
    "actual_signal",
    "real_signal",
    "predict_signal",
    "selected_symbols",
    "row_json",
    "metric_name",
    "metric_group",
    "metric_text",
    "metric_json",
    "insight_type",
    "source_model",
    "severity",
    "title",
    "message",
    "status",
    "error_message",
    "from_table",
    "from_column",
    "to_table",
    "to_column",
    "relationship_type",
    "note",
    "key_type",
    "key_columns",
    "table_name",
    "layer",
    "description",
}

DATE_HINTS = {
    "date",
    "trading_date",
    "prediction_date",
    "target_date",
    "future_trading_date",
    "listed_date",
    "insight_date",
    "validation_start_date",
    "validation_end_date",
    "test_start_date",
    "test_end_date",
    "train_start_date",
    "train_end_date",
}

DATETIME_HINTS = {
    "time",
    "created_at",
    "ingested_at",
    "uploaded_at",
    "started_at",
    "finished_at",
    "last_write_time",
}

BOOL_HINTS = {
    "is_correct",
    "prediction_correct",
    "direction_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload staging, warehouse, mart, and audit CSV outputs to ClickHouse."
    )
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--layers",
        default="",
        help=(
            "Comma-separated layers to upload: staging,warehouse,mart,audit. "
            "Default is all layers."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep uploading other tables if one table fails.",
    )
    return parser.parse_args()


def parse_layers(raw_layers: str) -> set[str]:
    if not raw_layers:
        return set()

    allowed = {"staging", "warehouse", "mart", "audit"}
    layers = {
        layer.strip().lower()
        for layer in str(raw_layers).split(",")
        if layer.strip()
    }
    unknown = layers - allowed
    if unknown:
        raise ValueError(
            "Unknown upload layers: "
            + ", ".join(sorted(unknown))
            + ". Allowed: "
            + ", ".join(sorted(allowed))
        )
    return layers


def get_clickhouse_client():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise ImportError(
            "clickhouse-connect is required. Install it with: pip install clickhouse-connect"
        ) from exc

    host = os.getenv("CLICKHOUSE_HOST", "")
    port = int(os.getenv("CLICKHOUSE_PORT") or "8443")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    database = os.getenv("CLICKHOUSE_DATABASE", "default")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    connect_timeout = int(os.getenv("CLICKHOUSE_CONNECT_TIMEOUT") or "60")
    send_receive_timeout = int(os.getenv("CLICKHOUSE_SEND_RECEIVE_TIMEOUT") or "600")
    query_retries = int(os.getenv("CLICKHOUSE_QUERY_RETRIES") or "3")
    client_retries = int(os.getenv("CLICKHOUSE_CLIENT_RETRIES") or "3")
    retry_sleep_seconds = int(os.getenv("CLICKHOUSE_RETRY_SLEEP_SECONDS") or "10")

    if not host:
        raise ValueError("CLICKHOUSE_HOST is required in .env")
    if not password:
        raise ValueError("CLICKHOUSE_PASSWORD is required in .env")

    last_error = None
    for attempt in range(1, client_retries + 1):
        try:
            client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
                secure=secure,
                connect_timeout=connect_timeout,
                send_receive_timeout=send_receive_timeout,
                query_retries=query_retries,
            )
            print(f"[clickhouse] Connected: {host}:{port} secure={secure}")
            return client
        except Exception as exc:
            last_error = exc
            if attempt >= client_retries:
                break
            print(
                "[clickhouse] Connection attempt "
                f"{attempt}/{client_retries} failed: {exc}. "
                f"Retrying in {retry_sleep_seconds}s..."
            )
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(
        "Cannot connect to ClickHouse after "
        f"{client_retries} attempts. Host={host}, port={port}, "
        f"secure={secure}, connect_timeout={connect_timeout}s"
    ) from last_error


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def full_table_name(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def sanitize_column_name(name: str) -> str:
    value = str(name).strip()
    value = value.replace("%", "pct")
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        value = "column"
    if value[0].isdigit():
        value = f"col_{value}"
    return value


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    seen: dict[str, int] = {}
    columns = []
    for column in result.columns:
        clean = sanitize_column_name(column)
        count = seen.get(clean, 0)
        seen[clean] = count + 1
        if count:
            clean = f"{clean}_{count + 1}"
        columns.append(clean)
    result.columns = columns
    return result


def file_checksum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_paths(spec: TableSpec) -> list[Path]:
    paths = [path for path in spec.paths if path.exists()]
    if spec.glob_pattern:
        paths.extend(sorted(PROJECT_ROOT.glob(spec.glob_pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "volumn": "volume",
    }
    result = result.rename(columns=rename_map)
    return result


def transform_dataframe(
    df: pd.DataFrame,
    spec: TableSpec,
    source_path: Path,
    upload_run_id: str,
    uploaded_at: pd.Timestamp,
) -> pd.DataFrame:
    df = standardize_raw_columns(df)

    if spec.transform == "stock_prices":
        if "time" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"time": "date"})
        required_columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df[required_columns]

    if spec.transform == "features_all":
        if "date" in df.columns and "trading_date" not in df.columns:
            df = df.rename(columns={"date": "trading_date"})
        if "time" in df.columns and "trading_date" not in df.columns:
            df = df.rename(columns={"time": "trading_date"})

    if spec.transform == "symbol_history":
        if "date" in df.columns:
            df = df.rename(columns={"date": "trading_date"})

    if spec.transform == "stock_symbols":
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        mapping_path = PROJECT_ROOT / "data" / "clean" / "symbol_sector_encoding.csv"
        if mapping_path.exists():
            mapping = pd.read_csv(mapping_path)
            mapping.columns = [str(col).strip().lower() for col in mapping.columns]
            if {"symbol", "encode_sector"}.issubset(mapping.columns):
                mapping["symbol"] = mapping["symbol"].astype(str).str.strip().str.upper()
                df = df.merge(mapping[["symbol", "encode_sector"]], on="symbol", how="left")

    if spec.transform == "model1_price_forecast":
        output = pd.DataFrame()
        output["model_run_id"] = upload_run_id
        output["prediction_date"] = df.get("trading_date")
        output["target_date"] = df.get("future_trading_date")
        output["symbol"] = df.get("symbol")
        output["real_close"] = df.get("target_close", df.get("future_close"))
        output["predicted_close"] = df.get(
            "predicted_future_close",
            df.get("predicted_close"),
        )
        output["actual_return"] = df.get("target_return")
        output["predicted_return"] = df.get("predicted_return")
        if "actual_direction" in df.columns and "predicted_direction" in df.columns:
            output["direction_correct"] = df["actual_direction"] == df["predicted_direction"]
        else:
            output["direction_correct"] = pd.NA
        output["model_name"] = "model1"
        output["created_at"] = uploaded_at
        return output

    if spec.transform == "model4_benchmark_outperformance":
        if "model_run_id" not in df.columns:
            df["model_run_id"] = upload_run_id
        if "model_name" not in df.columns:
            df["model_name"] = "model4"
        if "created_at" not in df.columns:
            df["created_at"] = uploaded_at

    if spec.transform == "json_rows":
        rows = []
        for idx, row in df.iterrows():
            rows.append(
                {
                    "source_file": source_path.name,
                    "row_index": int(idx),
                    "row_json": json.dumps(row.where(pd.notna(row), None).to_dict(), ensure_ascii=False),
                    "created_at": uploaded_at,
                }
            )
        return pd.DataFrame(rows)

    if spec.add_model_metadata:
        if "model_run_id" not in df.columns:
            df["model_run_id"] = upload_run_id
        if spec.model_name and "model_name" not in df.columns:
            df["model_name"] = spec.model_name
        if "created_at" not in df.columns:
            df["created_at"] = uploaded_at

    if spec.add_source_file:
        df["source_file"] = source_path.name
        df["file_checksum"] = file_checksum(source_path)
        df["ingested_at"] = uploaded_at
        if "source_system" not in df.columns:
            df["source_system"] = spec.layer

    return df


def read_sample(spec: TableSpec, paths: list[Path], upload_run_id: str, uploaded_at: pd.Timestamp) -> pd.DataFrame:
    samples = []
    for path in paths[:20]:
        try:
            sample = pd.read_csv(path, nrows=2000, encoding="utf-8-sig")
        except UnicodeDecodeError:
            sample = pd.read_csv(path, nrows=2000)
        if sample.empty and spec.transform != "json_rows":
            continue
        sample = transform_dataframe(sample, spec, path, upload_run_id, uploaded_at)
        samples.append(sample)
    if not samples:
        return pd.DataFrame()
    return sanitize_columns(pd.concat(samples, ignore_index=True))


def classify_column(column: str, sample: pd.Series) -> str:
    if column in DATETIME_HINTS or column.endswith("_at"):
        return "datetime"
    if column in DATE_HINTS or column.endswith("_date"):
        return "date"
    if column in BOOL_HINTS or column.startswith("is_") or column.endswith("_correct"):
        return "bool"
    if (
        column.endswith("_count")
        or column.endswith("_rows")
        or column in {"rows", "row_index", "total_symbols", "alert_count"}
    ):
        return "int"
    if column in STRING_HINTS:
        return "string"

    non_null = sample.dropna()
    if non_null.empty:
        return "string"
    numeric = pd.to_numeric(non_null.head(200), errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return "float"
    return "string"


def infer_schema(sample: pd.DataFrame, order_by: list[str]) -> dict[str, str]:
    order_set = set(order_by)
    schema = {}
    for column in sample.columns:
        kind = classify_column(column, sample[column])
        is_key = column in order_set
        if kind == "datetime":
            schema[column] = "DateTime" if is_key else "Nullable(DateTime)"
        elif kind == "date":
            schema[column] = "Date" if is_key else "Nullable(Date)"
        elif kind == "bool":
            schema[column] = "UInt8" if is_key else "Nullable(UInt8)"
        elif kind == "int":
            schema[column] = "Int64" if is_key else "Nullable(Int64)"
        elif kind == "float":
            schema[column] = "Float64" if is_key else "Nullable(Float64)"
        else:
            schema[column] = "String"
    return schema


def apply_schema_overrides(spec: TableSpec, schema: dict[str, str]) -> dict[str, str]:
    if spec.table == "stock_prices":
        return {
            "symbol": "String",
            "date": "DateTime",
            "open": "Float64",
            "high": "Float64",
            "low": "Float64",
            "close": "Float64",
            "volume": "Float64",
        }
    return schema


def create_table(client, database: str, table: str, schema: dict[str, str], order_by: list[str], append: bool):
    full_name = full_table_name(database, table)
    if not append:
        client.command(f"DROP TABLE IF EXISTS {full_name}")

    columns_sql = ",\n            ".join(
        f"{quote_identifier(column)} {col_type}" for column, col_type in schema.items()
    )
    order_sql = ", ".join(quote_identifier(column) for column in order_by)
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {full_name}
        (
            {columns_sql}
        )
        ENGINE = MergeTree
        PRIMARY KEY ({order_sql})
        ORDER BY ({order_sql})
        """
    )
    print(f"[clickhouse] Table ready: {database}.{table}")


def prepare_chunk(df: pd.DataFrame, schema: dict[str, str], order_by: list[str]) -> pd.DataFrame:
    df = sanitize_columns(df)
    for column in schema:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[list(schema.keys())].copy()

    for column, ch_type in schema.items():
        base_type = ch_type.replace("Nullable(", "").rstrip(")")
        if base_type == "String":
            df[column] = df[column].astype("string").fillna("").astype(str)
        elif base_type == "Date":
            parsed = pd.to_datetime(df[column], errors="coerce")
            df[column] = parsed.dt.date
        elif base_type == "DateTime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif base_type in {"UInt8", "Int64"}:
            if df[column].dtype == bool:
                numeric = df[column].astype(int)
            else:
                normalized = df[column].astype("string").str.strip().str.lower()
                normalized = normalized.replace(
                    {"true": "1", "false": "0", "yes": "1", "no": "0", "": pd.NA}
                )
                numeric = pd.to_numeric(normalized, errors="coerce")
            if "Nullable" in ch_type:
                df[column] = numeric.astype("Int64").astype(object).where(numeric.notna(), None)
            else:
                df[column] = numeric.fillna(0).astype(int)
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if order_by:
        df = df.dropna(subset=order_by)
    return df.where(pd.notna(df), None)


def upload_chunks(
    client,
    database: str,
    spec: TableSpec,
    paths: list[Path],
    schema: dict[str, str],
    upload_run_id: str,
    uploaded_at: pd.Timestamp,
    chunksize: int,
) -> int:
    total_rows = 0
    table_name = full_table_name(database, spec.table)
    for path in paths:
        print(f"[upload] {path} -> {database}.{spec.table}")
        try:
            chunks = pd.read_csv(path, chunksize=chunksize, encoding="utf-8-sig")
        except UnicodeDecodeError:
            chunks = pd.read_csv(path, chunksize=chunksize)

        for chunk_index, chunk in enumerate(chunks, 1):
            transformed = transform_dataframe(chunk, spec, path, upload_run_id, uploaded_at)
            prepared = prepare_chunk(transformed, schema, spec.order_by)
            if prepared.empty:
                continue
            client.insert_df(table=table_name, df=prepared)
            total_rows += len(prepared)
            print(
                f"[upload] {spec.table}: chunk {chunk_index} "
                f"rows={len(prepared):,}, total={total_rows:,}"
            )
    return total_rows


def upload_dataframe(
    client,
    database: str,
    table: str,
    df: pd.DataFrame,
    order_by: list[str],
    append: bool,
) -> int:
    if df.empty:
        print(f"[upload] Skip empty dataframe: {database}.{table}")
        return 0
    df = sanitize_columns(df)
    order_by = [sanitize_column_name(column) for column in order_by]
    schema = infer_schema(df, order_by)
    schema = apply_schema_overrides(TableSpec(table=table, layer="dataframe", order_by=order_by), schema)
    create_table(client, database, table, schema, order_by, append=append)
    prepared = prepare_chunk(df, schema, order_by)
    if prepared.empty:
        return 0
    client.insert_df(table=full_table_name(database, table), df=prepared)
    print(f"[upload] Finished {database}.{table}: {len(prepared):,} rows")
    return len(prepared)


def build_json_metrics_table(path: Path, model_name: str, upload_run_id: str, uploaded_at: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            rows.append(
                {
                    "model_run_id": upload_run_id,
                    "model_name": model_name,
                    "metric_name": key,
                    "metric_value": None,
                    "metric_json": json.dumps(value, ensure_ascii=False),
                    "created_at": uploaded_at,
                }
            )
        else:
            rows.append(
                {
                    "model_run_id": upload_run_id,
                    "model_name": model_name,
                    "metric_name": key,
                    "metric_value": value if isinstance(value, (int, float)) else None,
                    "metric_json": None if isinstance(value, (int, float)) else str(value),
                    "created_at": uploaded_at,
                }
            )
    return pd.DataFrame(rows)


def make_table_specs() -> list[TableSpec]:
    return [
        TableSpec(
            table="stg_raw_dirty_prices",
            layer="staging",
            paths=[PROJECT_ROOT / "data" / "dirty" / "Data_500_stocks_dirty.csv"],
            order_by=["symbol", "time"],
            add_source_file=True,
        ),
        TableSpec(
            table="stg_daily_prices",
            layer="staging",
            glob_pattern="data/dirty/stock_*.csv",
            order_by=["target_date", "symbol", "source_file"],
            add_source_file=True,
        ),
        TableSpec(
            table="stg_company_info",
            layer="staging",
            paths=[PROJECT_ROOT / "ingestion" / "company_infor.csv"],
            order_by=["symbol"],
            add_source_file=True,
        ),
        TableSpec(
            table="stg_symbol_history_2026",
            layer="staging",
            glob_pattern="ingestion/data_crawl_2026/*.csv",
            order_by=["symbol", "trading_date", "source_file"],
            transform="symbol_history",
            add_source_file=True,
        ),
        TableSpec(
            table="stock_prices",
            layer="warehouse",
            paths=[PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_clean_ver2.csv"],
            order_by=["symbol", "date"],
            transform="stock_prices",
        ),
        TableSpec(
            table="stock_symbols",
            layer="warehouse",
            paths=[PROJECT_ROOT / "ingestion" / "company_infor.csv"],
            order_by=["symbol"],
            transform="stock_symbols",
        ),
        TableSpec(
            table="sector_label_encoding",
            layer="warehouse",
            paths=[PROJECT_ROOT / "data" / "clean" / "sector_label_encoding.csv"],
            order_by=["encode_sector"],
        ),
        TableSpec(
            table="symbol_sector_encoding",
            layer="warehouse",
            paths=[PROJECT_ROOT / "data" / "clean" / "symbol_sector_encoding.csv"],
            order_by=["symbol"],
        ),
        TableSpec(
            table="features_all",
            layer="warehouse",
            paths=[PROJECT_ROOT / "data" / "clean" / "features_all.csv"],
            order_by=["symbol", "trading_date"],
            transform="features_all",
        ),
        TableSpec(
            table="audit_khaosat_rows",
            layer="audit",
            glob_pattern="data/khaosatdata/*.csv",
            order_by=["source_file", "row_index"],
            transform="json_rows",
        ),
        TableSpec(
            table="audit_clean_log_rows",
            layer="audit",
            glob_pattern="data/clean_log/*.csv",
            order_by=["source_file", "row_index"],
            transform="json_rows",
        ),
        TableSpec(
            table="mart_model1_price_forecast",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model1" / "reports" / "predictions.csv"],
            order_by=["symbol", "prediction_date", "model_run_id"],
            transform="model1_price_forecast",
        ),
        TableSpec(
            table="mart_model1_prediction_accuracy",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model1" / "reports" / "prediction_accuracy.csv"],
            order_by=["symbol", "date"],
            model_name="model1",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model1_feature_importance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model1" / "reports" / "feature_importance.csv"],
            order_by=["model_name", "feature"],
            model_name="model1",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model1_backtest_daily",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model1" / "reports" / "backtest.csv"],
            order_by=["model_name", "trading_date"],
            model_name="model1",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model1_backtest_sweep",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model1" / "reports" / "backtest_sweep.csv"],
            order_by=["model_name", "model_run_id"],
            model_name="model1",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model2_future_return",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model2" / "reports" / "mart_model2_future_return.csv"],
            order_by=["symbol", "trading_date", "model_run_id"],
        ),
        TableSpec(
            table="mart_model2_prediction_error",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model2" / "reports" / "prediction_error.csv"],
            order_by=["symbol", "trading_date"],
            model_name="model2",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model2_feature_importance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model2" / "reports" / "feature_importance.csv"],
            order_by=["model_name", "feature"],
            model_name="model2",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model2_backtest_daily",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model2" / "reports" / "backtest.csv"],
            order_by=["model_name", "trading_date"],
            model_name="model2",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model2_backtest_sweep",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model2" / "reports" / "backtest_sweep.csv"],
            order_by=["model_name", "top_n"],
            model_name="model2",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model3_trading_signals",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "mart_model3_trading_signals.csv"],
            order_by=["symbol", "trading_date", "model_name"],
        ),
        TableSpec(
            table="mart_model3_signal_summary",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "mart_model3_signal_summary.csv"],
            order_by=["model_name", "trading_date"],
        ),
        TableSpec(
            table="mart_model3_backtest_daily",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "mart_model3_backtest_daily.csv"],
            order_by=["model_name", "trading_date"],
        ),
        TableSpec(
            table="mart_model3_metrics",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "mart_model3_metrics.csv"],
            order_by=["model_name", "metric_group", "metric_name"],
        ),
        TableSpec(
            table="mart_model3_daily_insights",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "model3_daily_insights.csv"],
            order_by=["source_model", "insight_date", "metric_name"],
        ),
        TableSpec(
            table="mart_model3_feature_importance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "feature_importance.csv"],
            order_by=["model_name", "feature"],
            model_name="model3",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model3_backtest_sweep",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model3" / "reports" / "backtest_sweep.csv"],
            order_by=["model_name", "model_run_id"],
            model_name="model3",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model4_benchmark_outperformance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model4" / "output" / "benchmark_predictions.csv"],
            order_by=["symbol", "trading_date", "model_run_id"],
            transform="model4_benchmark_outperformance",
        ),
        TableSpec(
            table="mart_model4_feature_importance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model4" / "output" / "feature_importance.csv"],
            order_by=["model_name", "feature"],
            model_name="model4",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model5_risk_features",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "risk_features.csv"],
            order_by=["symbol", "trading_date"],
            model_name="model5",
            add_model_metadata=True,
        ),
        TableSpec(
            table="mart_model5_risk_predictions",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "risk_predictions.csv"],
            order_by=["symbol", "prediction_date", "model_name"],
        ),
        TableSpec(
            table="mart_model5_risk_test_evaluation",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "risk_test_evaluation.csv"],
            order_by=["symbol", "prediction_date", "model_name"],
        ),
        TableSpec(
            table="mart_model5_risk_alerts",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "mart_model5_risk_alerts.csv"],
            order_by=["symbol", "prediction_date", "model_run_id"],
        ),
        TableSpec(
            table="mart_model5_feature_importance",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "feature_importance.csv"],
            order_by=["model_name", "feature"],
        ),
        TableSpec(
            table="mart_model5_backtest_risk_alerts",
            layer="mart",
            paths=[PROJECT_ROOT / "models" / "model5" / "output_model5" / "backtest_risk_alerts.csv"],
            order_by=["model_name", "prediction_date", "model_run_id"],
        ),
    ]


def build_metric_tables(upload_run_id: str, uploaded_at: pd.Timestamp) -> list[tuple[str, pd.DataFrame]]:
    metric_sources = [
        ("mart_model1_metrics", "model1", PROJECT_ROOT / "models" / "model1" / "reports" / "metrics.json"),
        ("mart_model1_backtest_metrics", "model1", PROJECT_ROOT / "models" / "model1" / "reports" / "backtest_metrics.json"),
        ("mart_model2_metrics", "model2", PROJECT_ROOT / "models" / "model2" / "reports" / "metrics.json"),
        ("mart_model2_backtest_metrics", "model2", PROJECT_ROOT / "models" / "model2" / "reports" / "backtest_metrics.json"),
        ("mart_model4_metrics", "model4", PROJECT_ROOT / "models" / "model4" / "output" / "benchmark_metrics.json"),
        ("mart_model5_metrics", "model5", PROJECT_ROOT / "models" / "model5" / "output_model5" / "risk_metrics.json"),
    ]
    tables = []
    for table, model_name, path in metric_sources:
        df = build_json_metrics_table(path, model_name, upload_run_id, uploaded_at)
        if not df.empty:
            tables.append((table, df))
    return tables


def build_relationships(upload_run_id: str, uploaded_at: pd.Timestamp) -> pd.DataFrame:
    rows = [
        ("stg_daily_prices", "symbol", "stock_symbols", "symbol"),
        ("stg_raw_dirty_prices", "symbol", "stock_symbols", "symbol"),
        ("stock_prices", "symbol", "stock_symbols", "symbol"),
        ("features_all", "symbol", "stock_symbols", "symbol"),
        ("features_all", "encode_sector", "sector_label_encoding", "encode_sector"),
        ("symbol_sector_encoding", "symbol", "stock_symbols", "symbol"),
        ("symbol_sector_encoding", "encode_sector", "sector_label_encoding", "encode_sector"),
        ("mart_model1_price_forecast", "symbol", "stock_symbols", "symbol"),
        ("mart_model2_future_return", "symbol", "stock_symbols", "symbol"),
        ("mart_model3_trading_signals", "symbol", "stock_symbols", "symbol"),
        ("mart_model4_benchmark_outperformance", "symbol", "stock_symbols", "symbol"),
        ("mart_model5_risk_alerts", "symbol", "stock_symbols", "symbol"),
    ]
    return pd.DataFrame(
        [
            {
                "upload_run_id": upload_run_id,
                "from_table": src_table,
                "from_column": src_col,
                "to_table": dst_table,
                "to_column": dst_col,
                "relationship_type": "logical_foreign_key",
                "note": "ClickHouse does not enforce foreign keys; this table documents logical relationships.",
                "created_at": uploaded_at,
            }
            for src_table, src_col, dst_table, dst_col in rows
        ]
    )


def build_table_keys(specs: list[TableSpec], upload_run_id: str, uploaded_at: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "upload_run_id": upload_run_id,
                "table_name": spec.table,
                "layer": spec.layer,
                "key_type": "primary_sorting_key",
                "key_columns": ", ".join(spec.order_by),
                "description": spec.description,
                "created_at": uploaded_at,
            }
        )
    return pd.DataFrame(rows)


def run_upload(args: argparse.Namespace) -> int:
    upload_run_id = str(uuid.uuid4())
    uploaded_at = pd.Timestamp(datetime.now().replace(microsecond=0))
    selected_layers = parse_layers(args.layers)
    all_specs = make_table_specs()
    specs = [
        spec for spec in all_specs
        if not selected_layers or spec.layer in selected_layers
    ]
    manifest = []

    print(f"[upload_all] database={args.database}")
    print(
        "[upload_all] layers="
        + (",".join(sorted(selected_layers)) if selected_layers else "all")
    )
    print(f"[upload_all] upload_run_id={upload_run_id}")
    print(f"[upload_all] env={ENV_PATH}")

    if args.dry_run:
        for spec in specs:
            paths = resolve_paths(spec)
            status = "OK" if paths else "MISSING"
            print(f"[dry-run] {status}: {spec.layer}.{spec.table} files={len(paths)}")
            for path in paths[:5]:
                print(f"          - {path}")
            if len(paths) > 5:
                print(f"          ... {len(paths) - 5} more")
        return 0

    client = get_clickhouse_client()
    client.command(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(args.database)}")

    errors = []
    for spec in specs:
        paths = resolve_paths(spec)
        if not paths:
            print(f"[skip] Missing source files for {spec.table}")
            manifest.append(
                {
                    "upload_run_id": upload_run_id,
                    "layer": spec.layer,
                    "table_name": spec.table,
                    "source_file": "",
                    "rows_uploaded": 0,
                    "status": "missing",
                    "error_message": "",
                    "uploaded_at": uploaded_at,
                }
            )
            continue

        try:
            sample = read_sample(spec, paths, upload_run_id, uploaded_at)
            if sample.empty:
                print(f"[skip] Empty source for {spec.table}")
                continue
            order_by = [sanitize_column_name(column) for column in spec.order_by]
            schema = infer_schema(sample, order_by)
            schema = apply_schema_overrides(spec, schema)
            create_table(client, args.database, spec.table, schema, order_by, append=args.append)
            rows_uploaded = upload_chunks(
                client=client,
                database=args.database,
                spec=spec,
                paths=paths,
                schema=schema,
                upload_run_id=upload_run_id,
                uploaded_at=uploaded_at,
                chunksize=args.chunksize,
            )
            manifest.append(
                {
                    "upload_run_id": upload_run_id,
                    "layer": spec.layer,
                    "table_name": spec.table,
                    "source_file": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in paths),
                    "rows_uploaded": rows_uploaded,
                    "status": "success",
                    "error_message": "",
                    "uploaded_at": uploaded_at,
                }
            )
        except Exception as exc:
            message = str(exc)
            print(f"[error] {spec.table}: {message}")
            errors.append((spec.table, message))
            manifest.append(
                {
                    "upload_run_id": upload_run_id,
                    "layer": spec.layer,
                    "table_name": spec.table,
                    "source_file": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in paths),
                    "rows_uploaded": 0,
                    "status": "failed",
                    "error_message": message,
                    "uploaded_at": uploaded_at,
                }
            )
            if not args.continue_on_error:
                break

    upload_mart_metrics = not selected_layers or "mart" in selected_layers
    if upload_mart_metrics and (not errors or args.continue_on_error):
        for table, df in build_metric_tables(upload_run_id, uploaded_at):
            try:
                upload_dataframe(
                    client,
                    args.database,
                    table,
                    df,
                    order_by=["model_name", "metric_name", "model_run_id"],
                    append=args.append,
                )
            except Exception as exc:
                errors.append((table, str(exc)))
                if not args.continue_on_error:
                    break

    upload_audit_tables = not selected_layers or "audit" in selected_layers
    if upload_audit_tables:
        relationship_df = build_relationships(upload_run_id, uploaded_at)
        keys_df = build_table_keys(specs, upload_run_id, uploaded_at)
        manifest_df = pd.DataFrame(manifest)

        for table, df, order_by in [
            ("audit_table_relationships", relationship_df, ["from_table", "from_column", "to_table"]),
            ("audit_table_keys", keys_df, ["table_name", "key_type"]),
            ("audit_upload_manifest", manifest_df, ["upload_run_id", "table_name"]),
        ]:
            try:
                upload_dataframe(client, args.database, table, df, order_by=order_by, append=args.append)
            except Exception as exc:
                errors.append((table, str(exc)))
                if not args.continue_on_error:
                    break

    if errors:
        print("[upload_all] Finished with errors:")
        for table, message in errors:
            print(f"  - {table}: {message}")
        return 1

    print("[upload_all] Done.")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_upload(args))


if __name__ == "__main__":
    main()
