from __future__ import annotations

import json
import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

MODEL1_DIR = PROJECT_ROOT / "models" / "model1"
MODEL2_DIR = PROJECT_ROOT / "models" / "model2"
MODEL3_DIR = PROJECT_ROOT / "models" / "model3"
MODEL4_DIR = PROJECT_ROOT / "models" / "model4"
MODEL5_DIR = PROJECT_ROOT / "models" / "model5"

MODEL1_PATH = MODEL1_DIR / "models" / "price_forecasting_xgb.pkl"
MODEL2_PATH = MODEL2_DIR / "models" / "future_return_lgbm.pkl"
MODEL3_PATH = MODEL3_DIR / "models" / "trading_signal_xgb_classifier.pkl"

MODEL1_REPORT_DIR = MODEL1_DIR / "reports"
MODEL1_PREDICTION_LOG_PATH = MODEL1_REPORT_DIR / "streamlit_predictions.csv"
MODEL1_LATEST_PREDICTION_PATH = MODEL1_REPORT_DIR / "latest_streamlit_prediction.json"

CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "stock")
FEATURES_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
FEATURES_TABLE = os.getenv("CLICKHOUSE_TABLE", "features_all")
PRICE_TABLE = "stock_prices"
SYMBOL_TABLE = "stock_symbols"

MODEL2_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "encode_sector",
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

FEATURE_DEFINITIONS = [
    ("return_1d", "Lợi suất 1 phiên gần nhất."),
    ("return_3d", "Lợi suất 3 phiên gần nhất."),
    ("return_5d", "Lợi suất 5 phiên gần nhất."),
    ("return_10d", "Lợi suất 10 phiên gần nhất."),
    ("return_20d", "Lợi suất 20 phiên gần nhất."),
    ("ma_5, ma_20, ma_50", "Trung bình động giá đóng cửa theo 5/20/50 phiên."),
    ("price_vs_ma20", "Khoảng cách tương đối giữa giá đóng cửa và MA20."),
    ("ma5_vs_ma20", "Chênh lệch tương đối giữa MA5 và MA20."),
    ("volatility_5d", "Độ biến động lợi suất 1 ngày trong cửa sổ 5 phiên."),
    ("volatility_20d", "Độ biến động lợi suất 1 ngày trong cửa sổ 20 phiên."),
    ("volatility_change", "Mức thay đổi biến động ngắn hạn so với dài hạn."),
    ("rolling_max_20d", "Giá đóng cửa cao nhất trong 20 phiên gần nhất."),
    ("drawdown_20d", "Mức sụt giảm từ đỉnh 20 phiên."),
    ("volume_ma_5, volume_ma_20", "Trung bình khối lượng giao dịch 5/20 phiên."),
    ("volume_ratio_5_20", "Tỷ lệ volume MA5 so với volume MA20."),
    ("volume_change_1d", "Tốc độ thay đổi khối lượng so với phiên trước."),
    ("daily_range", "Biên độ trong ngày: high - low so với close."),
    ("body_ratio", "Tỷ lệ thân nến so với biên độ trong ngày."),
    ("close_position", "Vị trí giá đóng cửa trong khoảng low-high."),
    ("encode_sector", "Mã hóa ngành/lĩnh vực của cổ phiếu."),
]

MART_DESCRIPTIONS = {
    "stock.stock_prices": "Dữ liệu giao dịch theo ngày.",
    "stock.features_all": "Dữ liệu OHLCV đã được feature engineering.",
    "stock.stock_symbols": "Danh mục cổ phiếu, tên công ty và ngành.",
    "stock.symbol_sector_encoding": "Bảng mã hóa ngành theo symbol.",
    "stock.mart_future_return_prediction": "Mart dự đoán future return của Model 2.",
    "stock.model4_benchmark_predictions": "Kết quả dự đoán outperform benchmark của Model 4.",
    "stock_mart.mart_future_return_prediction": "Mart dự đoán future return của Model 2.",
    "stock_mart.mart_model1_price_forecast": "Mart dự báo return/giá của Model 1.",
    "stock_mart.mart_model1_top_expected_return": "Top cổ phiếu kỳ vọng tăng theo Model 1.",
    "stock_mart.mart_model1_backtest_daily": "Backtest hằng ngày của Model 1.",
    "stock_mart.mart_model1_backtest_metrics": "Metrics backtest của Model 1.",
    "stock_mart.mart_model1_backtest_sweep": "Kết quả sweep tham số backtest của Model 1.",
    "stock_mart.mart_model1_feature_importance": "Độ quan trọng feature của Model 1.",
    "stock_mart.mart_model1_metrics": "Metrics của Model 1.",
    "stock_mart.mart_model1_prediction_accuracy": "Bảng đánh giá đúng/sai dự đoán của Model 1.",
    "stock_mart.mart_model2_future_return": "Mart dự đoán lợi suất 5 phiên của Model 2.",
    "stock_mart.mart_model2_prediction_error": "Sai số dự đoán future_return_5d của Model 2.",
    "stock_mart.mart_model2_backtest_daily": "Backtest hằng ngày của Model 2.",
    "stock_mart.mart_model2_backtest_metrics": "Metrics backtest của Model 2.",
    "stock_mart.mart_model2_backtest_sweep": "Kết quả sweep tham số backtest của Model 2.",
    "stock_mart.mart_model2_feature_importance": "Độ quan trọng feature của Model 2.",
    "stock_mart.mart_model2_metrics": "Metrics của Model 2.",
    "stock_mart.mart_model3_trading_signals": "Mart tín hiệu BUY/HOLD/SELL của Model 3.",
    "stock_mart.mart_model3_signal_summary": "Tổng hợp số lượng tín hiệu Model 3 theo ngày/ngành.",
    "stock_mart.mart_model3_daily_insights": "Insight hằng ngày sinh từ kết quả Model 3.",
    "stock_mart.mart_model3_backtest_daily": "Backtest hằng ngày của Model 3.",
    "stock_mart.mart_model3_backtest_sweep": "Kết quả sweep tham số backtest của Model 3.",
    "stock_mart.mart_model3_feature_importance": "Độ quan trọng feature của Model 3.",
    "stock_mart.mart_model3_metrics": "Metrics của Model 3.",
    "stock_mart.mart_model4_benchmark_outperformance": "Mart dự đoán outperform benchmark của Model 4.",
    "stock_mart.mart_model4_daily_outperform_summary": "Tổng hợp outperform Model 4 theo ngày.",
    "stock_mart.mart_model4_sector_outperform": "Hiệu suất outperform Model 4 theo ngành.",
    "stock_mart.mart_model4_top_outperformers": "Top cổ phiếu có xác suất outperform cao theo Model 4.",
    "stock_mart.mart_model4_feature_importance": "Độ quan trọng feature của Model 4.",
    "stock_mart.mart_model4_metrics": "Metrics của Model 4.",
    "stock_mart.model4_benchmark_predictions": "Kết quả dự đoán outperform benchmark của Model 4.",
    "stock_mart.mart_model5_risk_alerts": "Mart cảnh báo rủi ro của Model 5.",
    "stock_mart.mart_model5_risk_predictions": "Dự đoán xác suất rủi ro của Model 5.",
    "stock_mart.mart_model5_risk_test_evaluation": "Đánh giá test set của Model 5.",
    "stock_mart.mart_model5_risk_features": "Feature và label rủi ro của Model 5.",
    "stock_mart.mart_model5_backtest_risk_alerts": "Backtest cảnh báo rủi ro của Model 5.",
    "stock_mart.mart_model5_feature_importance": "Độ quan trọng feature của Model 5.",
    "stock_mart.mart_model5_metrics": "Metrics của Model 5.",
    "stock_mart_model5_risk_prediction.risk_features": "Feature và label rủi ro của Model 5.",
    "stock_mart_model5_risk_prediction.risk_predictions": "Dự đoán xác suất rủi ro của Model 5.",
    "stock_mart_model5_risk_prediction.risk_test_evaluation": "Đánh giá test set của Model 5.",
    "stock_mart_model5_risk_prediction.mart_risk_alerts": "Mart cảnh báo rủi ro của Model 5.",
}

IMPORTANT_MARTS = [
    {
        "full_name": "stock_mart.mart_model1_price_forecast",
        "model": "Model 1",
        "mart": "mart_model1_price_forecast",
        "vai trò": "Dự báo giá/return",
        "lý do chọn": "Đầu ra chính của mô hình dự báo giá, dùng để so sánh predicted_close, predicted_return và real_close.",
    },
    {
        "full_name": "stock_mart.mart_model2_future_return",
        "model": "Model 2",
        "mart": "mart_model2_future_return",
        "vai trò": "Dự đoán lợi suất 5 phiên",
        "lý do chọn": "Thể hiện trực tiếp predicted_future_return_5d, actual_future_return_5d và tín hiệu tăng/giảm.",
    },
    {
        "full_name": "stock_mart.mart_model3_trading_signals",
        "model": "Model 3",
        "mart": "mart_model3_trading_signals",
        "vai trò": "BUY/HOLD/SELL",
        "lý do chọn": "Lưu tín hiệu giao dịch theo từng symbol/ngày cùng xác suất SELL, HOLD, BUY.",
    },
    {
        "full_name": "stock_mart.mart_model4_benchmark_outperformance",
        "model": "Model 4",
        "mart": "mart_model4_benchmark_outperformance",
        "vai trò": "Outperform benchmark",
        "lý do chọn": "Cho biết cổ phiếu nào có khả năng vượt benchmark và xác suất outperform.",
    },
    {
        "full_name": "stock_mart.mart_model5_risk_alerts",
        "model": "Model 5",
        "mart": "mart_model5_risk_alerts",
        "vai trò": "Cảnh báo rủi ro",
        "lý do chọn": "Mart quan trọng nhất cho bài toán risk, hiển thị risk_probability, risk_label và các feature cảnh báo.",
    },
]

MART_TABLE_NAME_HINTS = {
    "model4_benchmark_predictions",
}


st.set_page_config(page_title="HQTCSDL Stocks", layout="wide")


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def full_table_name(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def sql_string(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def display_error(message: str, exc: Exception) -> None:
    st.error(message)
    st.caption(str(exc))


@st.cache_resource
def get_clickhouse_client():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện clickhouse-connect. Cài bằng: pip install clickhouse-connect"
        ) from exc

    host = os.getenv("CLICKHOUSE_HOST")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD")
    port = int(os.getenv("CLICKHOUSE_PORT") or "8443")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not host or not username or not password:
        raise RuntimeError(
            "Thiếu cấu hình ClickHouse. Cần CLICKHOUSE_HOST, CLICKHOUSE_USER, "
            "CLICKHOUSE_PASSWORD trong .env hoặc biến môi trường."
        )

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=os.getenv("CLICKHOUSE_DATABASE", "stock"),
        secure=secure,
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    return get_clickhouse_client().query_df(query)


@st.cache_data(ttl=300, show_spinner=False)
def table_exists(database: str, table: str) -> bool:
    result = run_query(
        f"""
        SELECT count() AS cnt
        FROM system.tables
        WHERE database = {sql_string(database)}
          AND name = {sql_string(table)}
        """
    )
    return not result.empty and int(result.iloc[0]["cnt"]) > 0


@st.cache_data(ttl=300, show_spinner=False)
def get_table_columns(database: str, table: str) -> list[str]:
    result = run_query(
        f"""
        SELECT name
        FROM system.columns
        WHERE database = {sql_string(database)}
          AND table = {sql_string(table)}
        ORDER BY position
        """
    )
    if result.empty:
        return []
    return result["name"].astype(str).tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_symbols() -> list[str]:
    query = f"""
        SELECT symbol
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        GROUP BY symbol
        ORDER BY symbol
    """
    result = run_query(query)
    if result.empty or "symbol" not in result.columns:
        return []
    return result["symbol"].astype(str).str.upper().sort_values().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_feature_dates(symbol: str | None = None, limit: int = 2000) -> list[date]:
    where_sql = ""
    if symbol:
        where_sql = f"WHERE upper(trim(symbol)) = upper(trim({sql_string(symbol)}))"

    result = run_query(
        f"""
        SELECT DISTINCT toDate(trading_date) AS trading_date
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        {where_sql}
        ORDER BY trading_date DESC
        LIMIT {int(limit)}
        """
    )
    if result.empty or "trading_date" not in result.columns:
        return []
    return pd.to_datetime(result["trading_date"], errors="coerce").dt.date.dropna().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_feature_symbols() -> list[str]:
    result = run_query(
        f"""
        SELECT symbol
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        GROUP BY symbol
        ORDER BY symbol
        """
    )
    if result.empty or "symbol" not in result.columns:
        return []
    return result["symbol"].astype(str).str.upper().sort_values().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_table_symbols(database: str, table: str, limit: int = 5000) -> list[str]:
    result = run_query(
        f"""
        SELECT toString(symbol) AS symbol
        FROM {full_table_name(database, table)}
        WHERE notEmpty(toString(symbol))
        GROUP BY symbol
        ORDER BY symbol
        LIMIT {int(limit)}
        """
    )
    if result.empty or "symbol" not in result.columns:
        return []
    return result["symbol"].astype(str).str.upper().sort_values().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_table_dates(
    database: str,
    table: str,
    date_col: str,
    symbol: str | None = None,
    limit: int = 3000,
) -> list[date]:
    where_sql = ""
    if symbol and symbol != "Tất cả":
        where_sql = f"WHERE upper(trim(symbol)) = upper(trim({sql_string(symbol)}))"

    result = run_query(
        f"""
        SELECT DISTINCT toDate({quote_identifier(date_col)}) AS date_value
        FROM {full_table_name(database, table)}
        {where_sql}
        ORDER BY date_value DESC
        LIMIT {int(limit)}
        """
    )
    if result.empty or "date_value" not in result.columns:
        return []
    return pd.to_datetime(result["date_value"], errors="coerce").dt.date.dropna().tolist()


def build_mart_where_clause(
    columns: list[str],
    symbol_filter: str | None,
    date_col: str | None,
    date_filter,
) -> str:
    where = []
    if symbol_filter and symbol_filter != "Tất cả" and "symbol" in columns:
        where.append(f"upper(trim(symbol)) = upper(trim({sql_string(symbol_filter)}))")

    if date_col and date_filter:
        if isinstance(date_filter, tuple) and len(date_filter) == 2:
            start_date, end_date = date_filter
            where.append(
                f"toDate({quote_identifier(date_col)}) BETWEEN "
                f"toDate({sql_string(start_date)}) AND toDate({sql_string(end_date)})"
            )
        elif not isinstance(date_filter, tuple):
            where.append(
                f"toDate({quote_identifier(date_col)}) = toDate({sql_string(date_filter)})"
            )

    return "WHERE " + " AND ".join(where) if where else ""


def build_mart_preview_query(
    database: str,
    table: str,
    columns: list[str],
    symbol_filter: str | None,
    date_col: str | None,
    date_filter,
    limit: int,
) -> str:
    where_sql = build_mart_where_clause(columns, symbol_filter, date_col, date_filter)
    order_sql = f"ORDER BY {quote_identifier(date_col)} DESC" if date_col else ""
    return f"""
        SELECT *
        FROM {full_table_name(database, table)}
        {where_sql}
        {order_sql}
        LIMIT {int(limit)}
    """


def get_mart_summary(
    database: str,
    table: str,
    columns: list[str],
    symbol_filter: str | None,
    date_col: str | None,
    date_filter,
) -> pd.DataFrame:
    where_sql = build_mart_where_clause(columns, symbol_filter, date_col, date_filter)
    symbol_expr = "uniqExact(symbol) AS symbol_count" if "symbol" in columns else "0 AS symbol_count"
    if date_col:
        date_expr = (
            f"min(toDate({quote_identifier(date_col)})) AS min_date, "
            f"max(toDate({quote_identifier(date_col)})) AS max_date"
        )
    else:
        date_expr = "NULL AS min_date, NULL AS max_date"

    return run_query(
        f"""
        SELECT
            count() AS total_rows,
            {symbol_expr},
            {date_expr}
        FROM {full_table_name(database, table)}
        {where_sql}
        """
    )


def format_insight_value(column: str, value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    lower_column = column.lower()
    if (
        "probability" in lower_column
        or "return" in lower_column
        or "rate" in lower_column
        or "ratio" in lower_column
        or "accuracy" in lower_column
    ):
        return pct(numeric_value)
    return f"{numeric_value:,.4f}"


def render_mart_insights(
    table_full_name: str,
    mart_df: pd.DataFrame,
    selected_symbol: str | None,
    date_col: str | None,
    summary_df: pd.DataFrame | None = None,
) -> None:
    st.subheader("Insight từ mart đã chọn")
    if mart_df.empty:
        st.info("Không có dữ liệu mart phù hợp với bộ lọc hiện tại.")
        return

    total_rows = len(mart_df)
    symbol_count = mart_df["symbol"].nunique() if "symbol" in mart_df.columns else None
    min_date = None
    max_date = None
    if summary_df is not None and not summary_df.empty:
        row = summary_df.iloc[0]
        total_rows = int(row.get("total_rows", total_rows) or 0)
        symbol_count = row.get("symbol_count", symbol_count)
        min_date = row.get("min_date")
        max_date = row.get("max_date")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Dòng đang phân tích", number(total_rows))
    if "symbol" in mart_df.columns:
        metric_cols[1].metric("Số symbol", number(symbol_count))
    else:
        metric_cols[1].metric("Số symbol", "N/A")

    if max_date is not None and min_date is not None and not pd.isna(max_date) and not pd.isna(min_date):
        metric_cols[2].metric("Ngày mới nhất", pd.to_datetime(max_date).strftime("%Y-%m-%d"))
        metric_cols[3].metric("Ngày cũ nhất", pd.to_datetime(min_date).strftime("%Y-%m-%d"))
    elif date_col and date_col in mart_df.columns:
        parsed_dates = pd.to_datetime(mart_df[date_col], errors="coerce").dropna()
        if not parsed_dates.empty:
            metric_cols[2].metric("Ngày mới nhất", parsed_dates.max().strftime("%Y-%m-%d"))
            metric_cols[3].metric("Ngày cũ nhất", parsed_dates.min().strftime("%Y-%m-%d"))
        else:
            metric_cols[2].metric("Ngày mới nhất", "N/A")
            metric_cols[3].metric("Ngày cũ nhất", "N/A")
    else:
        metric_cols[2].metric("Ngày mới nhất", "N/A")
        metric_cols[3].metric("Ngày cũ nhất", "N/A")

    insights = [
        f"Đang đọc mart `{table_full_name}`"
        + (f" cho symbol `{selected_symbol}`." if selected_symbol and selected_symbol != "Tất cả" else ".")
    ]

    label_columns = [
        column
        for column in [
            "risk_label",
            "predicted_risk_label",
            "actual_risk_label",
            "adjusted_signal",
            "predicted_signal",
            "target_signal",
            "signal",
            "prediction_correct",
            "is_correct",
        ]
        if column in mart_df.columns
    ]
    for column in label_columns[:3]:
        counts = mart_df[column].astype(str).value_counts(dropna=False).head(3)
        summary = ", ".join(f"{idx}: {int(value)}" for idx, value in counts.items())
        insights.append(f"Phân bố `{column}` nổi bật: {summary}.")

    numeric_priority = [
        "risk_probability",
        "outperform_probability",
        "predicted_return",
        "predicted_future_return_5d",
        "actual_future_return_5d",
        "actual_return",
        "target_return",
        "return_5d",
        "signal_confidence",
        "buy_probability",
        "sell_probability",
        "close",
        "volume",
    ]
    numeric_columns = []
    for column in numeric_priority:
        if column in mart_df.columns:
            numeric_columns.append(column)
    for column in mart_df.columns:
        if column in numeric_columns:
            continue
        numeric = pd.to_numeric(mart_df[column], errors="coerce")
        if numeric.notna().sum() > 0:
            numeric_columns.append(column)
        if len(numeric_columns) >= 6:
            break

    numeric_summary_rows = []
    for column in numeric_columns[:6]:
        numeric = pd.to_numeric(mart_df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        numeric_summary_rows.append(
            {
                "metric": column,
                "mean": numeric.mean(),
                "min": numeric.min(),
                "max": numeric.max(),
            }
        )
        insights.append(
            f"`{column}` trung bình {format_insight_value(column, numeric.mean())}, "
            f"cao nhất {format_insight_value(column, numeric.max())}."
        )

    if "risk_probability" in mart_df.columns:
        numeric = pd.to_numeric(mart_df["risk_probability"], errors="coerce")
        if numeric.notna().any():
            top_idx = numeric.idxmax()
            top_row = mart_df.loc[top_idx]
            top_symbol = top_row.get("symbol", selected_symbol or "N/A")
            insights.append(
                f"Rủi ro cao nhất thuộc `{top_symbol}` với "
                f"`risk_probability` = {pct(numeric.loc[top_idx])}."
            )

    if "predicted_return" in mart_df.columns:
        numeric = pd.to_numeric(mart_df["predicted_return"], errors="coerce")
        if numeric.notna().any():
            top_idx = numeric.idxmax()
            top_row = mart_df.loc[top_idx]
            top_symbol = top_row.get("symbol", selected_symbol or "N/A")
            insights.append(
                f"Kỳ vọng return cao nhất thuộc `{top_symbol}`: "
                f"{pct(numeric.loc[top_idx])}."
            )

    for item in insights[:10]:
        st.markdown(f"- {item}")

    if numeric_summary_rows:
        st.dataframe(
            pd.DataFrame(numeric_summary_rows),
            use_container_width=True,
            hide_index=True,
        )


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def pct(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f}"


def add_model_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module_from_path(module_name: str, module_path: Path):
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file module: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không tạo được import spec cho: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def load_model1_artifact():
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL1_PATH}")
    add_model_path(MODEL1_DIR)
    saved = joblib.load(MODEL1_PATH)
    return saved


@st.cache_resource
def load_model2_artifact():
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL2_PATH}")
    return joblib.load(MODEL2_PATH)


@st.cache_resource
def load_model3_artifact():
    if not MODEL3_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL3_PATH}")
    saved = joblib.load(MODEL3_PATH)
    return (
        saved["model"],
        saved["features"],
        saved.get("signal_labels", {0: "SELL", 1: "HOLD", 2: "BUY"}),
    )


def fetch_feature_rows(symbol: str | None = None, trading_date=None, limit: int = 500) -> pd.DataFrame:
    where = []
    if symbol:
        where.append(f"upper(trim(symbol)) = upper(trim({sql_string(symbol)}))")
    if trading_date:
        where.append(f"toDate(trading_date) = toDate({sql_string(trading_date)})")

    where_sql = "WHERE " + " AND ".join(where) if where else ""
    query = f"""
        SELECT *
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        {where_sql}
        ORDER BY trading_date DESC, symbol
        LIMIT {int(limit)}
    """
    result = run_query(query)
    result.columns = [str(col).strip() for col in result.columns]
    if "trading_date" in result.columns:
        result["trading_date"] = pd.to_datetime(result["trading_date"], errors="coerce")
    return result


def add_prediction_signal(predicted_return: float) -> str:
    if predicted_return >= 0.03:
        return "STRONG_BUY"
    if predicted_return >= 0.01:
        return "BUY"
    if predicted_return > -0.01:
        return "HOLD"
    if predicted_return > -0.03:
        return "SELL"
    return "STRONG_SELL"


def apply_confidence_adjusted_signals(
    df: pd.DataFrame,
    min_action_probability: float = 0.60,
    min_action_margin: float = 0.0,
) -> pd.DataFrame:
    result_df = df.copy()
    probabilities = result_df[
        ["sell_probability", "hold_probability", "buy_probability"]
    ].apply(pd.to_numeric, errors="coerce")
    buy_edge = probabilities["buy_probability"] - probabilities["sell_probability"]
    sell_edge = probabilities["sell_probability"] - probabilities["buy_probability"]

    adjusted_signal = np.full(len(result_df), "HOLD", dtype=object)
    adjusted_label = np.full(len(result_df), 1, dtype=int)

    buy_mask = (
        (probabilities["buy_probability"] >= min_action_probability)
        & (probabilities["buy_probability"] >= probabilities["hold_probability"])
        & (buy_edge >= min_action_margin)
    )
    sell_mask = (
        (probabilities["sell_probability"] >= min_action_probability)
        & (probabilities["sell_probability"] >= probabilities["hold_probability"])
        & (sell_edge >= min_action_margin)
    )

    adjusted_signal[buy_mask.to_numpy()] = "BUY"
    adjusted_label[buy_mask.to_numpy()] = 2
    adjusted_signal[sell_mask.to_numpy()] = "SELL"
    adjusted_label[sell_mask.to_numpy()] = 0

    result_df["adjusted_signal_label"] = adjusted_label
    result_df["adjusted_signal"] = adjusted_signal
    result_df["signal_confidence"] = probabilities.max(axis=1)
    result_df["buy_sell_margin"] = buy_edge
    return result_df


def save_model1_prediction(result: dict) -> None:
    MODEL1_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame([result])
    if MODEL1_PREDICTION_LOG_PATH.exists():
        result_df.to_csv(
            MODEL1_PREDICTION_LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        result_df.to_csv(MODEL1_PREDICTION_LOG_PATH, index=False)

    MODEL1_LATEST_PREDICTION_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=4, default=str),
        encoding="utf-8",
    )


def render_header() -> None:
    st.title("HQTCSDL Stocks")
    st.caption(
        "Dashboard dữ liệu chứng khoán, feature engineering, data mart, mô hình dự đoán "
        "và insight từ ClickHouse."
    )


def render_overview_page() -> None:
    st.header("Tổng Quan Dashboard")
    try:
        summary = run_query(
            f"""
            SELECT
                count() AS total_rows,
                countDistinct(symbol) AS total_symbols,
                min(toDate(date)) AS min_date,
                max(toDate(date)) AS max_date
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            """
        )
        sector_summary = run_query(
            f"""
            SELECT countDistinct(sector) AS sector_count
            FROM {full_table_name(CLICKHOUSE_DATABASE, SYMBOL_TABLE)}
            WHERE sector IS NOT NULL AND sector != ''
            """
        )
        feature_rows = run_query(
            f"""
            SELECT count() AS feature_rows
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được dữ liệu tổng quan từ ClickHouse.", exc)
        return

    row = summary.iloc[0] if not summary.empty else {}
    total_symbols = row.get("total_symbols", 0)
    total_rows = row.get("total_rows", 0)
    min_date = row.get("min_date", "N/A")
    max_date = row.get("max_date", "N/A")
    sector_count = (
        int(sector_summary.iloc[0]["sector_count"]) if not sector_summary.empty else 0
    )
    feature_count = int(feature_rows.iloc[0]["feature_rows"]) if not feature_rows.empty else 0

    cols = st.columns(5)
    cols[0].metric("Mã cổ phiếu", number(total_symbols))
    cols[1].metric("Bản ghi giá", number(total_rows))
    cols[2].metric("Bản ghi feature", number(feature_count))
    cols[3].metric("Số ngành", number(sector_count))
    cols[4].metric("Cập nhật mới nhất", str(max_date))

    st.subheader("Tóm tắt hệ thống")
    st.dataframe(
        pd.DataFrame(
            [
                ["Tổng số mã cổ phiếu", f"{number(total_symbols)} mã"],
                ["Khoảng thời gian dữ liệu", f"{min_date} đến {max_date}"],
                ["Tổng số bản ghi", number(total_rows)],
                ["Số ngành/lĩnh vực", number(sector_count)],
                ["Ngày dữ liệu mới nhất", str(max_date)],
            ],
            columns=["Nội dung", "Giá trị"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Thanh khoản và giá trung bình gần đây")
    try:
        market = run_query(
            f"""
            SELECT
                toDate(date) AS trading_date,
                sum(volume) AS total_volume,
                avg(close) AS avg_close
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            GROUP BY trading_date
            ORDER BY trading_date DESC
            LIMIT 180
            """
        ).sort_values("trading_date")
        if not market.empty:
            chart_cols = st.columns(2)
            chart_cols[0].line_chart(market, x="trading_date", y="avg_close")
            chart_cols[1].bar_chart(market, x="trading_date", y="total_volume")
    except Exception as exc:
        st.warning(f"Không vẽ được biểu đồ tổng quan: {exc}")

    st.subheader("Top cổ phiếu theo volume phiên mới nhất")
    try:
        top_volume = run_query(
            f"""
            SELECT
                symbol,
                sum(volume) AS total_volume,
                avg(close) AS avg_close
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            WHERE toDate(date) = (
                SELECT max(toDate(date))
                FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            )
            GROUP BY symbol
            ORDER BY total_volume DESC
            LIMIT 15
            """
        )
        st.bar_chart(top_volume, x="symbol", y="total_volume")
        st.dataframe(top_volume, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Không lấy được top volume: {exc}")


def render_stock_lookup_page() -> None:
    st.header("Tra Cứu Dữ Liệu Cổ Phiếu")
    try:
        symbols = get_symbols()
    except Exception as exc:
        display_error("Không tải được danh sách mã cổ phiếu.", exc)
        return

    if not symbols:
        st.warning("Chưa có danh sách mã cổ phiếu trong ClickHouse.")
        return

    summary = run_query(
        f"""
        SELECT min(toDate(date)) AS min_date, max(toDate(date)) AS max_date
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        """
    )
    min_date = pd.to_datetime(summary.iloc[0]["min_date"]).date()
    max_date = pd.to_datetime(summary.iloc[0]["max_date"]).date()

    col1, col2, col3 = st.columns([1.3, 1.3, 0.8])
    selected_symbols = col1.multiselect(
        "Symbol",
        symbols,
        default=[s for s in ["FPT"] if s in symbols] or symbols[:1],
    )
    selected_range = col2.date_input(
        "Khoảng ngày",
        value=(max(min_date, date(max_date.year - 1, max_date.month, max_date.day)), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    limit = col3.number_input("Số dòng tối đa", min_value=50, max_value=10000, value=1000, step=50)

    if not selected_symbols:
        st.info("Chọn ít nhất một mã cổ phiếu để tra cứu.")
        return

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range

    symbol_sql = ", ".join(sql_string(symbol) for symbol in selected_symbols)
    query = f"""
        SELECT
            toDate(date) AS trading_date,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
        WHERE symbol IN ({symbol_sql})
          AND toDate(date) BETWEEN toDate({sql_string(start_date)})
                              AND toDate({sql_string(end_date)})
        ORDER BY trading_date, symbol
        LIMIT {int(limit)}
    """

    try:
        price_df = run_query(query)
    except Exception as exc:
        display_error("Không truy vấn được dữ liệu cổ phiếu.", exc)
        return

    if price_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        return

    st.dataframe(price_df, use_container_width=True, hide_index=True)
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"])

    st.subheader("Biểu đồ giá đóng cửa")
    close_chart = price_df.pivot_table(
        index="trading_date",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    st.line_chart(close_chart)

    st.subheader("Biểu đồ khối lượng giao dịch")
    volume_chart = price_df.pivot_table(
        index="trading_date",
        columns="symbol",
        values="volume",
        aggfunc="sum",
    )
    st.bar_chart(volume_chart)

    # if len(selected_symbols) == 1:
    #     st.subheader("OHLC preview")
    #     st.dataframe(
    #         price_df[["trading_date", "open", "high", "low", "close", "volume"]]
    #         .tail(30)
    #         .sort_values("trading_date", ascending=False),
    #         use_container_width=True,
    #         hide_index=True,
    #     )


def render_feature_engineering_page() -> None:
    st.header("Feature Engineering")
    st.dataframe(
        pd.DataFrame(FEATURE_DEFINITIONS, columns=["Feature", "Ý nghĩa"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Truy vấn bảng features_all")
    try:
        symbols = get_feature_symbols()
    except Exception as exc:
        display_error("Không tải được danh sách symbol.", exc)
        return

    col1, col2, col3 = st.columns([1, 1.4, 0.7])
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
    )
    date_range = col2.date_input("Khoảng ngày feature", value=())
    limit = col3.number_input("Limit", min_value=50, max_value=5000, value=500, step=50)

    where = [f"symbol = {sql_string(symbol)}"]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        where.append(
            f"toDate(trading_date) BETWEEN toDate({sql_string(date_range[0])}) "
            f"AND toDate({sql_string(date_range[1])})"
        )
    query = f"""
        SELECT *
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        WHERE {" AND ".join(where)}
        ORDER BY trading_date DESC
        LIMIT {int(limit)}
    """

    try:
        feature_df = run_query(query)
    except Exception as exc:
        display_error("Không truy vấn được bảng features_all.", exc)
        return

    if feature_df.empty:
        st.info("Không có feature phù hợp.")
        return

    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    feature_df["trading_date"] = pd.to_datetime(feature_df["trading_date"])
    feature_df = feature_df.sort_values("trading_date")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Close và MA")
        ma_cols = [col for col in ["close", "ma_5", "ma_20", "ma_50"] if col in feature_df]
        st.line_chart(feature_df[["trading_date", *ma_cols]], x="trading_date", y=ma_cols)
    with chart_cols[1]:
        st.subheader("Return và volatility")
        metric_cols = [
            col
            for col in ["return_5d", "return_20d", "volatility_5d", "volatility_20d"]
            if col in feature_df
        ]
        if metric_cols:
            st.line_chart(
                feature_df[["trading_date", *metric_cols]],
                x="trading_date",
                y=metric_cols,
            )


def render_data_mart_page() -> None:
    st.header("Data Mart")
    st.caption(
        "Các bảng mart được lấy trực tiếp từ system.tables của ClickHouse. "
        "Phần đầu trang ưu tiên 5 mart quan trọng nhất, mỗi model một mart chính."
    )

    databases = [
        CLICKHOUSE_DATABASE,
        "stock_mart",
        "stock_mart_model5_risk_prediction",
    ]
    db_sql = ", ".join(sql_string(db) for db in sorted(set(databases)))

    try:
        tables = run_query(
            f"""
            SELECT
                database,
                name AS table_name,
                total_rows,
                total_bytes
            FROM system.tables
            WHERE database IN ({db_sql})
            ORDER BY database, name
            """
        )
    except Exception as exc:
        display_error("Không lấy được danh sách bảng ClickHouse.", exc)
        return

    if tables.empty:
        st.warning("Không tìm thấy bảng trong các database mart/warehouse đã cấu hình.")
        return

    tables["full_name"] = tables["database"].astype(str) + "." + tables["table_name"].astype(str)
    tables["nội dung"] = tables["full_name"].map(MART_DESCRIPTIONS).fillna("Bảng dữ liệu ClickHouse.")
    tables["total_rows"] = tables["total_rows"].fillna(0).astype(int)
    tables["total_bytes"] = pd.to_numeric(tables["total_bytes"], errors="coerce").fillna(0).astype(int)
    tables["is_mart"] = (
        tables["database"].astype(str).str.contains("mart", case=False, na=False)
        | tables["table_name"].astype(str).str.startswith("mart_")
        | tables["table_name"].astype(str).isin(MART_TABLE_NAME_HINTS)
    )
    mart_tables = tables[tables["is_mart"]].copy()
    if mart_tables.empty:
        mart_tables = tables.copy()

    important_rows = []
    important_full_names = []
    for rank, spec in enumerate(IMPORTANT_MARTS, start=1):
        match = mart_tables[mart_tables["full_name"] == spec["full_name"]]
        if match.empty:
            match = mart_tables[mart_tables["table_name"] == spec["mart"]]

        row = {
            "ưu tiên": rank,
            "model": spec["model"],
            "mart": spec["mart"],
            "full_name": spec["full_name"],
            "vai trò": spec["vai trò"],
            "lý do chọn": spec["lý do chọn"],
            "trạng thái": "Chưa thấy trong ClickHouse",
            "total_rows": 0,
        }
        if not match.empty:
            first_match = match.iloc[0]
            row["full_name"] = first_match["full_name"]
            row["trạng thái"] = "Đã có"
            row["total_rows"] = int(first_match["total_rows"])
            important_full_names.append(first_match["full_name"])
        important_rows.append(row)

    important_df = pd.DataFrame(important_rows)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Tổng mart tìm thấy", number(len(mart_tables)))
    metric_cols[1].metric("Mart trọng tâm đã có", f"{len(important_full_names)}/5")
    metric_cols[2].metric("Tổng dòng mart", number(mart_tables["total_rows"].sum()))
    metric_cols[3].metric("Database đang quét", number(len(set(databases))))

    st.subheader("5 data mart trọng tâm")
    st.dataframe(
        important_df[
            ["ưu tiên", "model", "full_name", "vai trò", "lý do chọn", "trạng thái", "total_rows"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    missing_important = important_df[important_df["trạng thái"] != "Đã có"]
    if not missing_important.empty:
        st.info(
            "Một số mart trọng tâm chưa thấy trong ClickHouse hiện tại: "
            + ", ".join(missing_important["mart"].astype(str))
        )

    st.subheader("Toàn bộ data mart trong ClickHouse")
    search = st.text_input("Tìm mart", value="", placeholder="Nhập tên mart, ví dụ: model3, risk, forecast...")
    filtered_marts = mart_tables.copy()
    if search.strip():
        pattern = search.strip().lower()
        filtered_marts = filtered_marts[
            filtered_marts["full_name"].astype(str).str.lower().str.contains(pattern)
            | filtered_marts["nội dung"].astype(str).str.lower().str.contains(pattern)
        ]

    if filtered_marts.empty:
        st.warning("Không có mart phù hợp với từ khóa tìm kiếm.")
        return

    st.dataframe(
        filtered_marts[["full_name", "nội dung", "total_rows", "total_bytes"]],
        use_container_width=True,
        hide_index=True,
    )

    filtered_names = set(filtered_marts["full_name"].tolist())
    important_filtered_names = [name for name in important_full_names if name in filtered_names]
    select_options = important_filtered_names + [
        name for name in filtered_marts["full_name"].tolist() if name not in important_filtered_names
    ]
    selected_table = st.selectbox("Chọn mart để preview", select_options)
    selected_db, selected_name = selected_table.split(".", 1)
    columns = get_table_columns(selected_db, selected_name)

    if not columns:
        st.info("Không đọc được schema bảng đã chọn.")
        return

    st.write("Schema")
    st.dataframe(pd.DataFrame({"column": columns}), use_container_width=True, hide_index=True)

    selected_symbol = None
    date_filter = None
    date_col = first_existing(
        columns,
        [
            "trading_date",
            "prediction_date",
            "target_date",
            "date",
            "run_date",
            "insight_date",
            "created_at",
            "updated_at",
        ],
    )

    filter_cols = st.columns(3)
    if "symbol" in columns:
        try:
            table_symbols = get_table_symbols(selected_db, selected_name)
        except Exception as exc:
            table_symbols = []
            filter_cols[0].caption(f"Không tải được danh sách symbol: {exc}")

        if table_symbols:
            selected_symbol = filter_cols[0].selectbox(
                "Chọn symbol",
                ["Tất cả", *table_symbols],
                index=0,
                key=f"mart_symbol_{selected_db}_{selected_name}",
            )
        else:
            selected_symbol = filter_cols[0].text_input("Lọc symbol", value="")
    if date_col:
        try:
            available_mart_dates = get_table_dates(
                selected_db,
                selected_name,
                date_col,
                selected_symbol,
            )
        except Exception as exc:
            available_mart_dates = []
            filter_cols[1].caption(f"Không tải được danh sách ngày: {exc}")

        if available_mart_dates:
            date_choice = filter_cols[1].selectbox(
                f"Chọn ngày có dữ liệu ({date_col})",
                ["Tất cả", *available_mart_dates],
                index=0,
                format_func=lambda value: value
                if isinstance(value, str)
                else value.strftime("%Y-%m-%d"),
                key=f"mart_date_{selected_db}_{selected_name}_{selected_symbol}",
            )
            date_filter = None if date_choice == "Tất cả" else date_choice
            filter_cols[1].caption(
                "Khoảng ngày mart: "
                f"{available_mart_dates[-1].strftime('%Y-%m-%d')} -> "
                f"{available_mart_dates[0].strftime('%Y-%m-%d')}"
            )
        else:
            filter_cols[1].warning("Mart này chưa có ngày phù hợp với symbol đã chọn.")
    preview_limit = filter_cols[2].number_input(
        "Preview rows",
        min_value=20,
        max_value=1000,
        value=100,
        step=20,
    )

    try:
        preview_query = build_mart_preview_query(
            database=selected_db,
            table=selected_name,
            columns=columns,
            symbol_filter=selected_symbol,
            date_col=date_col,
            date_filter=date_filter,
            limit=int(preview_limit),
        )
        summary = get_mart_summary(
            database=selected_db,
            table=selected_name,
            columns=columns,
            symbol_filter=selected_symbol,
            date_col=date_col,
            date_filter=date_filter,
        )
        preview = run_query(preview_query)
        render_mart_insights(
            table_full_name=selected_table,
            mart_df=preview,
            selected_symbol=selected_symbol,
            date_col=date_col,
            summary_df=summary,
        )
        with st.expander("SQL preview đang chạy"):
            st.code(preview_query, language="sql")
        st.subheader("Bảng mart trong ClickHouse")
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception as exc:
        display_error("Không preview được bảng đã chọn.", exc)


def render_model1_prediction() -> None:
    os.environ["MODEL1_DASHBOARD_EMBEDDED"] = "1"
    add_model_path(MODEL1_DIR)

    try:
        dashboard_model1 = load_module_from_path(
            "embedded_dashboard_model1",
            MODEL1_DIR / "test.py",
        )
    except Exception as exc:
        display_error("Không tải được Streamlit Model 1.", exc)
        st.caption(f"Kiểm tra file: {MODEL1_DIR / 'test.py'}")
        return

    try:
        dashboard_model1.main()
    except Exception as exc:
        display_error("Không render được Streamlit Model 1.", exc)
    return

    st.subheader("Model 1 - Dự đoán return 5 phiên và suy ra giá")
    st.caption("Logic dựa trên models/model1/test.py.")

    try:
        symbols = get_feature_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2 = st.columns(2)
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
        key="model1_symbol",
    )
    try:
        valid_dates = get_feature_dates(symbol)
    except Exception as exc:
        display_error("Không tải được danh sách ngày có feature.", exc)
        return
    if not valid_dates:
        st.warning(f"Không có ngày feature nào cho mã {symbol}.")
        return
    trading_date = col2.selectbox(
        "Ngày giao dịch có dữ liệu",
        valid_dates,
        index=0,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
        key="model1_date",
    )

    if not st.button("Dự đoán Model 1", type="primary"):
        return

    try:
        artifact = load_model1_artifact()
        row_df = fetch_feature_rows(symbol=symbol, trading_date=trading_date, limit=1)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 1.", exc)
        return

    if row_df.empty:
        st.warning("Không tìm thấy feature cho symbol/ngày đã chọn.")
        return

    model = artifact["model"]
    features = artifact["features"]
    horizon = artifact.get("horizon", 5)
    target_type = artifact.get("target_type", "future_return")
    return_calibrator = artifact.get("return_calibrator")

    missing_features = [col for col in features if col not in row_df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    for col in features + ["close"]:
        row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
    if row_df[features + ["close"]].isna().any(axis=None):
        st.error("Dòng dữ liệu được chọn có feature rỗng hoặc không phải số.")
        return

    raw_prediction = np.asarray(model.predict(row_df[features]), dtype=float)
    close = float(row_df["close"].iloc[0])

    if target_type == "future_return":
        if return_calibrator is not None:
            predicted_return = float(return_calibrator.predict(raw_prediction)[0])
        else:
            predicted_return = float(raw_prediction[0])
        predicted_close = close * (1 + predicted_return)
    else:
        predicted_close = float(raw_prediction[0])
        predicted_return = predicted_close / close - 1

    direction = "UP" if predicted_return >= 0 else "DOWN"
    result = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "trading_date": str(trading_date),
        "close": close,
        "predicted_return": predicted_return,
        "predicted_return_pct": predicted_return * 100,
        "predicted_close": predicted_close,
        "direction": direction,
        "horizon": horizon,
        "target_type": target_type,
    }
    save_model1_prediction(result)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Close hiện tại", f"{close:,.2f}")
    metric_cols[1].metric("Return dự đoán", pct(predicted_return))
    metric_cols[2].metric("Close dự đoán", f"{predicted_close:,.2f}")
    metric_cols[3].metric("Xu hướng", direction)

    st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)
    st.write("Feature đưa vào model")
    st.dataframe(row_df[["symbol", "trading_date", *features]], use_container_width=True, hide_index=True)


def render_model2_prediction() -> None:
    st.subheader("Model 2 - Dự đoán future_return_5d")
    st.caption("Logic dựa trên models/model2/streamlit_model2_demo.py, nhưng dùng .env thay vì hardcode credential.")

    try:
        symbols = get_feature_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2 = st.columns(2)
    symbol = col1.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("FPT") if "FPT" in symbols else 0,
        key="model2_symbol",
    )
    try:
        valid_dates = get_feature_dates(symbol)
    except Exception as exc:
        display_error("Không tải được danh sách ngày có feature.", exc)
        return
    if not valid_dates:
        st.warning(f"Không có ngày feature nào cho mã {symbol}.")
        return
    trading_date = col2.selectbox(
        "Ngày giao dịch có dữ liệu",
        valid_dates,
        index=0,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
        key="model2_date",
    )

    if not st.button("Dự đoán Model 2", type="primary"):
        return

    try:
        model = load_model2_artifact()
        row_df = fetch_feature_rows(symbol=symbol, trading_date=trading_date, limit=1)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 2.", exc)
        return

    if row_df.empty:
        st.warning("Không tìm thấy feature cho symbol/ngày đã chọn.")
        return

    missing_features = [col for col in MODEL2_FEATURE_COLUMNS if col not in row_df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    X = row_df[MODEL2_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    prediction = float(model.predict(X)[0])
    signal = add_prediction_signal(prediction)
    close = float(pd.to_numeric(row_df["close"], errors="coerce").iloc[0])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Symbol", symbol)
    metric_cols[1].metric("Close hiện tại", f"{close:,.2f}")
    metric_cols[2].metric("Future return 5D", pct(prediction))
    metric_cols[3].metric("Nhận định", signal)

    result = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trading_date": str(trading_date),
                "close": close,
                "predicted_future_return_5d": prediction,
                "predicted_future_return_pct": prediction * 100,
                "signal": signal,
            }
        ]
    )
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.write("Feature đưa vào model")
    st.dataframe(X, use_container_width=True, hide_index=True)


def render_model3_prediction() -> None:
    st.subheader("Model 3 - Tín hiệu BUY / HOLD / SELL")
    st.caption("Logic dựa trên models/model3/test.py.")

    try:
        symbols = ["Tất cả"] + get_feature_symbols()
    except Exception as exc:
        display_error("Không tải được symbol.", exc)
        return

    col1, col2, col3 = st.columns([1, 1, 1])
    selected_symbol = col1.selectbox("Symbol", symbols, key="model3_symbol")
    symbol_filter = None if selected_symbol == "Tất cả" else selected_symbol
    try:
        valid_dates = get_feature_dates(symbol_filter)
    except Exception as exc:
        display_error("Không tải được danh sách ngày có feature.", exc)
        return
    if not valid_dates:
        st.warning("Không có ngày feature phù hợp với lựa chọn symbol.")
        return
    trading_date = col2.selectbox(
        "Ngày giao dịch có dữ liệu",
        valid_dates,
        index=0,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
        key="model3_date",
    )
    min_prob = col3.slider("Ngưỡng confidence", 0.0, 1.0, 0.60, 0.05)

    if not st.button("Dự đoán Model 3", type="primary"):
        return

    try:
        model, features, signal_labels = load_model3_artifact()
        df = fetch_feature_rows(symbol=symbol_filter, trading_date=trading_date, limit=1000)
    except Exception as exc:
        display_error("Không tải được model hoặc dữ liệu Model 3.", exc)
        return

    if df.empty:
        st.warning("Không tìm thấy dữ liệu feature cho ngày đã chọn.")
        return

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error("Thiếu feature: " + ", ".join(missing_features))
        return

    df = df.dropna(subset=features).copy()
    if df.empty:
        st.error("Sau khi loại dòng thiếu feature, không còn dữ liệu hợp lệ.")
        return

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[features]
    pred_label = model.predict(X).astype(int)
    pred_proba = model.predict_proba(X)

    result = df.copy()
    result["predicted_signal_label"] = pred_label
    result["predicted_signal"] = result["predicted_signal_label"].map(signal_labels)
    for label_id, label_name in signal_labels.items():
        result[f"{label_name.lower()}_probability"] = pred_proba[:, label_id]
    result["predicted_signal_score"] = result["buy_probability"] - result["sell_probability"]
    result = apply_confidence_adjusted_signals(
        result,
        min_action_probability=min_prob,
        min_action_margin=0.0,
    )

    display_cols = [
        "symbol",
        "trading_date",
        "close",
        "predicted_signal",
        "adjusted_signal",
        "buy_probability",
        "hold_probability",
        "sell_probability",
        "signal_confidence",
        "buy_sell_margin",
    ]
    available_cols = [col for col in display_cols if col in result.columns]

    counts = result["adjusted_signal"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Số mã hợp lệ", number(len(result)))
    metric_cols[1].metric("BUY", number(counts.get("BUY", 0)))
    metric_cols[2].metric("HOLD", number(counts.get("HOLD", 0)))
    metric_cols[3].metric("SELL", number(counts.get("SELL", 0)))

    st.dataframe(
        result[available_cols].sort_values("buy_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render_model4_prediction() -> None:
    os.environ["MODEL4_DASHBOARD_EMBEDDED"] = "1"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        dashboard_model4 = load_module_from_path(
            "embedded_dashboard_model4",
            MODEL4_DIR / "dashboard_model4.py",
        )
    except Exception as exc:
        display_error("Không tải được dashboard Model 4.", exc)
        st.caption(f"Kiểm tra file: {MODEL4_DIR / 'dashboard_model4.py'}")
        return

    try:
        dashboard_model4.main()
    except Exception as exc:
        display_error("Không render được dashboard Model 4.", exc)


def render_model5_prediction() -> None:
    os.environ["MODEL5_DASHBOARD_EMBEDDED"] = "1"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        dashboard_model5 = load_module_from_path(
            "embedded_dashboard_model5",
            MODEL5_DIR / "dashboard_model5.py",
        )
    except Exception as exc:
        display_error("Không tải được dashboard Model 5.", exc)
        st.caption(f"Kiểm tra file: {MODEL5_DIR / 'dashboard_model5.py'}")
        return

    try:
        dashboard_model5.render_dashboard(embed=True)
    except Exception as exc:
        display_error("Không render được dashboard Model 5.", exc)


def render_prediction_models_page() -> None:
    st.header("Mô Hình Dự Đoán")
    model_page = st.radio(
        "Chọn mô hình",
        [
            "Model 1 - Future close",
            "Model 2 - Dự đoán lợi suất 5 ngày",
            "Model 3 - BUY/HOLD/SELL",
            "Model 4 - Outperform benchmark",
            "Model 5 - Risk alert",
        ],
        horizontal=True,
    )

    if model_page.startswith("Model 1"):
        render_model1_prediction()
    elif model_page.startswith("Model 2"):
        render_model2_prediction()
    elif model_page.startswith("Model 3"):
        render_model3_prediction()
    elif model_page.startswith("Model 4"):
        render_model4_prediction()
    else:
        render_model5_prediction()


def render_insight_page() -> None:
    st.header("Insight")

    try:
        available_dates = get_feature_dates(limit=3000)
    except Exception as exc:
        display_error("Không lấy được danh sách ngày feature.", exc)
        return

    if not available_dates:
        st.warning("Chưa có ngày feature nào để sinh insight.")
        return

    selected_date = st.selectbox(
        "Ngày insight",
        available_dates,
        index=0,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
    )
    selected_filter = (
        "toDate(f.trading_date) = "
        f"toDate({sql_string(selected_date)})"
    )

    latest_available_date = available_dates[0]
    st.caption(
        f"Đang sinh insight cho ngày {selected_date}. "
        f"Ngày feature mới nhất hiện có: {latest_available_date}."
    )

    try:
        top_gain = run_query(
            f"""
            SELECT symbol, close, return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {selected_filter} AND return_5d IS NOT NULL
            ORDER BY return_5d DESC
            LIMIT 10
            """
        )
        top_loss = run_query(
            f"""
            SELECT symbol, close, return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {selected_filter} AND return_5d IS NOT NULL
            ORDER BY return_5d ASC
            LIMIT 10
            """
        )
        top_volume = run_query(
            f"""
            SELECT symbol, close, volume
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {selected_filter}
            ORDER BY volume DESC
            LIMIT 10
            """
        )
        sector_perf = run_query(
            f"""
            SELECT
                coalesce(s.sector, 'UNKNOWN') AS sector,
                avg(f.return_5d) AS avg_return_5d,
                count() AS stock_count
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            LEFT JOIN {full_table_name(CLICKHOUSE_DATABASE, SYMBOL_TABLE)} AS s
                ON f.symbol = s.symbol
            WHERE {selected_filter}
              AND f.return_5d IS NOT NULL
            GROUP BY sector
            ORDER BY avg_return_5d DESC
            LIMIT 15
            """
        )
        volatile = run_query(
            f"""
            SELECT symbol, close, volatility_20d, abs(return_5d) AS abs_return_5d
            FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)} AS f
            WHERE {selected_filter}
            ORDER BY volatility_20d DESC, abs_return_5d DESC
            LIMIT 10
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được insight từ ClickHouse.", exc)
        return

    insight_rows = [
        ["Top cổ phiếu tăng mạnh nhất", ", ".join(top_gain["symbol"].astype(str).head(5))],
        ["Top cổ phiếu giảm mạnh nhất", ", ".join(top_loss["symbol"].astype(str).head(5))],
        ["Ngành có hiệu suất tốt nhất", ", ".join(sector_perf["sector"].astype(str).head(3))],
        ["Cổ phiếu biến động mạnh nhất", ", ".join(volatile["symbol"].astype(str).head(5))],
        ["Cổ phiếu thanh khoản cao nhất", ", ".join(top_volume["symbol"].astype(str).head(5))],
    ]
    st.dataframe(pd.DataFrame(insight_rows, columns=["Insight", "Kết quả"]), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top tăng 5 phiên")
        st.bar_chart(top_gain, x="symbol", y="return_5d")
        st.dataframe(top_gain, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Top giảm 5 phiên")
        st.bar_chart(top_loss, x="symbol", y="return_5d")
        st.dataframe(top_loss, use_container_width=True, hide_index=True)

    st.subheader("Hiệu suất theo ngành")
    st.bar_chart(sector_perf, x="sector", y="avg_return_5d")
    st.dataframe(sector_perf, use_container_width=True, hide_index=True)

    st.subheader("Thanh khoản cao nhất")
    st.bar_chart(top_volume, x="symbol", y="volume")


def render_data_quality_page() -> None:
    st.header("Chất Lượng Dữ Liệu")

    try:
        quality = run_query(
            f"""
            SELECT
                count() AS total_rows,
                countDistinct(symbol) AS total_symbols,
                min(toDate(date)) AS min_date,
                max(toDate(date)) AS max_date,
                countIf(open <= 0 OR high <= 0 OR low <= 0 OR close <= 0) AS non_positive_prices,
                countIf(volume < 0) AS negative_volume,
                countIf(
                    high < low
                    OR high < open
                    OR high < close
                    OR low > open
                    OR low > close
                ) AS invalid_ohlc,
                count() - uniqExact(tuple(symbol, date)) AS duplicate_symbol_date
            FROM {full_table_name(CLICKHOUSE_DATABASE, PRICE_TABLE)}
            """
        )
    except Exception as exc:
        display_error("Không truy vấn được quality check từ ClickHouse.", exc)
        return

    row = quality.iloc[0] if not quality.empty else {}
    cols = st.columns(4)
    cols[0].metric("Tổng dòng", number(row.get("total_rows")))
    cols[1].metric("Số mã", number(row.get("total_symbols")))
    cols[2].metric("Ngày đầu", str(row.get("min_date")))
    cols[3].metric("Ngày cuối", str(row.get("max_date")))

    cols = st.columns(4)
    cols[0].metric("Giá <= 0", number(row.get("non_positive_prices")))
    cols[1].metric("Volume âm", number(row.get("negative_volume")))
    cols[2].metric("OHLC lỗi", number(row.get("invalid_ohlc")))
    cols[3].metric("Trùng symbol-date", number(row.get("duplicate_symbol_date")))

    st.dataframe(quality, use_container_width=True, hide_index=True)

    clean_summary = PROJECT_ROOT / "data" / "clean_log" / "clean_summary.txt"
    survey_summary = PROJECT_ROOT / "data" / "khaosatdata" / "khaosat_summary.txt"

    st.subheader("Log làm sạch dữ liệu")
    if clean_summary.exists():
        st.text(clean_summary.read_text(encoding="utf-8"))
    else:
        st.info("Chưa tìm thấy data/clean_log/clean_summary.txt")

    st.subheader("Log khảo sát dữ liệu dirty")
    if survey_summary.exists():
        st.text(survey_summary.read_text(encoding="utf-8"))
    else:
        st.info("Chưa tìm thấy data/khaosatdata/khaosat_summary.txt")


def main() -> None:
    render_header()

    page = st.sidebar.radio(
        "Chức năng",
        [
            "1. Tổng quan Dashboard",
            "2. Tra cứu dữ liệu cổ phiếu",
            "3. Feature Engineering",
            "4. Data Mart",
            "5. Mô hình dự đoán",
            "6. Insight",
            "7. Chất lượng dữ liệu",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"ClickHouse database: {CLICKHOUSE_DATABASE}")
    st.sidebar.caption(f"Feature table: {FEATURES_DATABASE}.{FEATURES_TABLE}")

    if page.startswith("1."):
        render_overview_page()
    elif page.startswith("2."):
        render_stock_lookup_page()
    elif page.startswith("3."):
        render_feature_engineering_page()
    elif page.startswith("4."):
        render_data_mart_page()
    elif page.startswith("5."):
        render_prediction_models_page()
    elif page.startswith("6."):
        render_insight_page()
    else:
        render_data_quality_page()


if __name__ == "__main__":
    main()
