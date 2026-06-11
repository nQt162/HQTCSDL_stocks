from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


MODEL5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL5_DIR.parents[1]
MODEL_PATH = MODEL5_DIR / "models" / "risk_alert_model.pkl"
OUTPUT_DIR = MODEL5_DIR / "output_model5"
METRICS_PATH = OUTPUT_DIR / "risk_metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"

for _env_candidate in [
    MODEL5_DIR / ".env",
    PROJECT_ROOT / ".env",
    MODEL5_DIR.parent / ".env",
    Path(".") / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break
else:
    load_dotenv()

FEATURES_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
FEATURES_TABLE = os.getenv("CLICKHOUSE_TABLE", "features_all")
DEFAULT_THRESHOLD = 0.6

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

FEATURE_DESCRIPTIONS = {
    "encode_sector": "Ma hoa nganh",
    "return_1d": "Loi suat 1 ngay",
    "return_3d": "Loi suat 3 ngay",
    "return_5d": "Loi suat 5 ngay gan nhat",
    "return_10d": "Loi suat 10 ngay",
    "return_20d": "Loi suat 20 ngay",
    "ma_5": "Trung binh dong 5 ngay",
    "ma_20": "Trung binh dong 20 ngay",
    "ma_50": "Trung binh dong 50 ngay",
    "price_vs_ma20": "Gia so voi MA20",
    "ma5_vs_ma20": "MA5 so voi MA20",
    "volatility_5d": "Bien dong 5 ngay",
    "volatility_20d": "Bien dong 20 ngay",
    "volatility_change": "Thay doi bien dong ngan/dai han",
    "rolling_max_20d": "Dinh 20 ngay",
    "drawdown_20d": "Muc sut giam so voi dinh 20 ngay",
    "volume_ma_5": "Khoi luong trung binh 5 ngay",
    "volume_ma_20": "Khoi luong trung binh 20 ngay",
    "volume_ratio_5_20": "Ty le volume MA5/MA20",
    "volume_change_1d": "Thay doi volume 1 ngay",
    "daily_range": "Bien do trong ngay",
    "body_ratio": "Ty le than nen",
    "close_position": "Vi tri gia dong cua trong nen",
}

TABLE_CANDIDATES = {
    "alerts": [
        ("stock_mart", "mart_model5_risk_alerts"),
        ("stock_mart_model5_risk_prediction", "mart_risk_alerts"),
        ("stock_mart_model5_risk_prediction", "dashboard_risk_alerts"),
    ],
    "predictions": [
        ("stock_mart", "mart_model5_risk_predictions"),
        ("stock_mart_model5_risk_prediction", "risk_predictions"),
    ],
    "evaluation": [
        ("stock_mart", "mart_model5_risk_test_evaluation"),
        ("stock_mart_model5_risk_prediction", "risk_test_evaluation"),
    ],
    "features": [
        ("stock_mart", "mart_model5_risk_features"),
        ("stock_mart_model5_risk_prediction", "risk_features"),
    ],
    "metrics": [
        ("stock_mart", "mart_model5_metrics"),
    ],
    "feature_importance": [
        ("stock_mart", "mart_model5_feature_importance"),
    ],
    "backtest": [
        ("stock_mart", "mart_model5_backtest_risk_alerts"),
    ],
}


if os.getenv("MODEL5_DASHBOARD_EMBEDDED") != "1":
    st.set_page_config(
        page_title="Module 5 - Risk Alert",
        page_icon="!",
        layout="wide",
    )


st.markdown(
    """
<style>
.model5-kpi {
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #dc2626;
    border-radius: 8px;
    padding: 16px 18px;
    min-height: 116px;
}
.model5-kpi-label {
    color: #4b5563;
    font-size: 13px;
    margin-bottom: 6px;
}
.model5-kpi-value {
    color: #111827;
    font-size: 26px;
    font-weight: 700;
    margin: 0;
}
.model5-kpi-sub {
    color: #6b7280;
    font-size: 12px;
    margin-top: 6px;
}
.model5-section {
    font-size: 20px;
    font-weight: 700;
    margin: 20px 0 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def full_table_name(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def sql_str(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return str(value)


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def display_error(message: str, exc: Exception) -> None:
    st.error(message)
    st.caption(str(exc))


@st.cache_resource
def get_clickhouse_client():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise RuntimeError("Thieu clickhouse-connect. Cai bang: pip install clickhouse-connect") from exc

    host = os.getenv("CLICKHOUSE_HOST")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD")
    port = int(os.getenv("CLICKHOUSE_PORT") or "8443")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not host or not password:
        raise RuntimeError("Thieu CLICKHOUSE_HOST hoac CLICKHOUSE_PASSWORD trong .env")

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
        WHERE database = {sql_str(database)}
          AND name = {sql_str(table)}
        """
    )
    return not result.empty and int(result.iloc[0]["cnt"]) > 0


@st.cache_data(ttl=300, show_spinner=False)
def get_table_columns(database: str, table: str) -> list[str]:
    result = run_query(
        f"""
        SELECT name
        FROM system.columns
        WHERE database = {sql_str(database)}
          AND table = {sql_str(table)}
        ORDER BY position
        """
    )
    if result.empty or "name" not in result.columns:
        return []
    return result["name"].astype(str).tolist()


def resolve_table(kind: str) -> tuple[str, str] | None:
    for database, table in TABLE_CANDIDATES.get(kind, []):
        try:
            if table_exists(database, table):
                return database, table
        except Exception:
            continue
    return None


def resolved_name(kind: str) -> str | None:
    resolved = resolve_table(kind)
    if not resolved:
        return None
    return full_table_name(*resolved)


@st.cache_resource
def load_model5():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    saved = joblib.load(MODEL_PATH)
    return saved


@st.cache_data(ttl=300, show_spinner=False)
def load_alert_summary() -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    return run_query(
        f"""
        SELECT
            count() AS total_rows,
            uniqExact(symbol) AS symbol_count,
            min(toDate(prediction_date)) AS min_date,
            max(toDate(prediction_date)) AS max_date,
            countIf(risk_label = 'HIGH_RISK') AS high_risk_rows,
            avg(risk_probability) AS avg_risk_probability,
            max(risk_probability) AS max_risk_probability
        FROM {table}
        """
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_alerts(limit: int = 100) -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    df = run_query(
        f"""
        SELECT *
        FROM {table}
        WHERE toDate(prediction_date) = (
            SELECT max(toDate(prediction_date))
            FROM {table}
        )
        ORDER BY risk_probability DESC, symbol
        LIMIT {int(limit)}
        """
    )
    return normalize_dates(df, ["prediction_date", "target_date"])


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_alert_summary() -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    df = run_query(
        f"""
        SELECT
            toDate(prediction_date) AS prediction_date,
            count() AS total_symbols,
            countIf(risk_label = 'HIGH_RISK') AS high_risk_count,
            avg(risk_probability) AS avg_risk_probability,
            max(risk_probability) AS max_risk_probability
        FROM {table}
        GROUP BY prediction_date
        ORDER BY prediction_date ASC
        """
    )
    df = normalize_dates(df, ["prediction_date"])
    if not df.empty:
        df["high_risk_ratio"] = (
            pd.to_numeric(df["high_risk_count"], errors="coerce")
            / pd.to_numeric(df["total_symbols"], errors="coerce").replace(0, np.nan)
        )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_evaluation_summary() -> pd.DataFrame:
    table = resolved_name("evaluation")
    if not table:
        return pd.DataFrame()
    return run_query(
        f"""
        SELECT
            count() AS total_rows,
            avg(toFloat64(prediction_correct)) AS accuracy,
            countIf(actual_risk_label = 'HIGH_RISK') AS actual_high_risk,
            countIf(predicted_risk_label = 'HIGH_RISK') AS predicted_high_risk,
            avg(risk_probability) AS avg_risk_probability
        FROM {table}
        """
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_evaluation_daily() -> pd.DataFrame:
    table = resolved_name("evaluation")
    if not table:
        return pd.DataFrame()
    df = run_query(
        f"""
        SELECT
            toDate(prediction_date) AS prediction_date,
            count() AS total_rows,
            avg(toFloat64(prediction_correct)) AS accuracy,
            countIf(actual_risk_label = 'HIGH_RISK') AS actual_high_risk,
            countIf(predicted_risk_label = 'HIGH_RISK') AS predicted_high_risk,
            avg(risk_probability) AS avg_risk_probability
        FROM {table}
        GROUP BY prediction_date
        ORDER BY prediction_date ASC
        """
    )
    return normalize_dates(df, ["prediction_date"])


@st.cache_data(ttl=300, show_spinner=False)
def load_sector_summary() -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    symbol_table = full_table_name("stock", "stock_symbols")
    df = run_query(
        f"""
        SELECT
            coalesce(s.sector, 'UNKNOWN') AS sector,
            count() AS total_rows,
            countIf(a.risk_label = 'HIGH_RISK') AS high_risk_count,
            avg(a.risk_probability) AS avg_risk_probability,
            max(a.risk_probability) AS max_risk_probability
        FROM {table} AS a
        LEFT JOIN {symbol_table} AS s
            ON upper(trim(a.symbol)) = upper(trim(s.symbol))
        GROUP BY sector
        ORDER BY high_risk_count DESC, avg_risk_probability DESC
        LIMIT 30
        """
    )
    if not df.empty:
        df["high_risk_ratio"] = (
            pd.to_numeric(df["high_risk_count"], errors="coerce")
            / pd.to_numeric(df["total_rows"], errors="coerce").replace(0, np.nan)
        )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_symbol_summary(limit: int = 100) -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    df = run_query(
        f"""
        SELECT
            symbol,
            count() AS total_rows,
            countIf(risk_label = 'HIGH_RISK') AS high_risk_count,
            avg(risk_probability) AS avg_risk_probability,
            max(risk_probability) AS max_risk_probability
        FROM {table}
        GROUP BY symbol
        ORDER BY high_risk_count DESC, max_risk_probability DESC
        LIMIT {int(limit)}
        """
    )
    if not df.empty:
        df["high_risk_ratio"] = (
            pd.to_numeric(df["high_risk_count"], errors="coerce")
            / pd.to_numeric(df["total_rows"], errors="coerce").replace(0, np.nan)
        )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_symbols_for_demo() -> list[str]:
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
    return result["symbol"].astype(str).str.upper().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_dates_for_symbol(symbol: str) -> list:
    result = run_query(
        f"""
        SELECT DISTINCT toDate(trading_date) AS d
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        WHERE upper(trim(symbol)) = upper(trim({sql_str(symbol)}))
        ORDER BY d DESC
        LIMIT 500
        """
    )
    if result.empty or "d" not in result.columns:
        return []
    return pd.to_datetime(result["d"], errors="coerce").dt.date.dropna().tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_features_for_demo(symbol: str, trading_date) -> pd.DataFrame:
    df = run_query(
        f"""
        SELECT *
        FROM {full_table_name(FEATURES_DATABASE, FEATURES_TABLE)}
        WHERE upper(trim(symbol)) = upper(trim({sql_str(symbol)}))
          AND toDate(trading_date) = toDate({sql_str(trading_date)})
        LIMIT 1
        """
    )
    return normalize_dates(df, ["trading_date"])


@st.cache_data(ttl=300, show_spinner=False)
def get_existing_alert_for_symbol_date(symbol: str, trading_date) -> pd.DataFrame:
    table = resolved_name("alerts")
    if not table:
        return pd.DataFrame()
    df = run_query(
        f"""
        SELECT *
        FROM {table}
        WHERE upper(trim(symbol)) = upper(trim({sql_str(symbol)}))
          AND toDate(prediction_date) = toDate({sql_str(trading_date)})
        LIMIT 1
        """
    )
    return normalize_dates(df, ["prediction_date", "target_date"])


def normalize_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def load_metrics_json() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_feature_importance_frame() -> pd.DataFrame:
    resolved = resolve_table("feature_importance")
    if resolved:
        try:
            df = run_query(
                f"""
                SELECT feature, importance
                FROM {full_table_name(*resolved)}
                ORDER BY importance DESC
                """
            )
            if not df.empty:
                return df
        except Exception:
            pass

    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if {"feature", "importance"}.issubset(df.columns):
                return df[["feature", "importance"]]
        except Exception:
            pass

    try:
        saved = load_model5()
        model = saved.get("model", saved)
        features = saved.get("features", FEATURE_COLUMNS)
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
    except Exception:
        pass

    return pd.DataFrame(columns=["feature", "importance"])


def kpi_card(label: str, value: str, sub: str = "", color: str = "#dc2626") -> None:
    st.markdown(
        f"""
        <div class="model5-kpi" style="border-left-color:{color}">
            <div class="model5-kpi-label">{label}</div>
            <p class="model5-kpi-value">{value}</p>
            <div class="model5-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab_overview() -> None:
    st.markdown('<p class="model5-section">Tong quan canh bao rui ro</p>', unsafe_allow_html=True)

    try:
        summary = load_alert_summary()
        latest = load_latest_alerts(limit=100)
        eval_summary = load_evaluation_summary()
    except Exception as exc:
        display_error("Khong tai duoc mart risk alert tu ClickHouse.", exc)
        return

    if summary.empty:
        st.warning("Chua tim thay bang mart risk alert cua Model 5 tren ClickHouse.")
        st.caption("Dashboard se hien thi day du sau khi chay model5 pipeline va upload output len ClickHouse.")
        return

    row = summary.iloc[0]
    total_rows = int(row.get("total_rows", 0) or 0)
    high_risk_rows = int(row.get("high_risk_rows", 0) or 0)
    high_risk_ratio = high_risk_rows / total_rows if total_rows else 0
    latest_date = row.get("max_date", "N/A")
    avg_risk_probability = row.get("avg_risk_probability")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Tong dong mart", number(total_rows), "Tat ca prediction rows", "#991b1b")
    with c2:
        kpi_card("So ma co phieu", number(row.get("symbol_count")), "Distinct symbol", "#be123c")
    with c3:
        kpi_card("HIGH_RISK", number(high_risk_rows), f"Ty le {pct(high_risk_ratio)}", "#dc2626")
    with c4:
        kpi_card("Risk probability TB", pct(avg_risk_probability), "Trung binh toan mart", "#ea580c")
    with c5:
        kpi_card("Ngay moi nhat", str(latest_date), f"Bat dau: {row.get('min_date', 'N/A')}", "#7c2d12")

    if not eval_summary.empty:
        eval_row = eval_summary.iloc[0]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Accuracy test", pct(eval_row.get("accuracy")))
        e2.metric("Actual HIGH_RISK", number(eval_row.get("actual_high_risk")))
        e3.metric("Predicted HIGH_RISK", number(eval_row.get("predicted_high_risk")))
        e4.metric("Risk probability TB test", pct(eval_row.get("avg_risk_probability")))

    st.markdown('<p class="model5-section">Top canh bao rui ro ngay moi nhat</p>', unsafe_allow_html=True)
    if latest.empty:
        st.info("Chua co du lieu ngay moi nhat.")
        return

    show_cols = [
        col
        for col in [
            "symbol",
            "prediction_date",
            "target_date",
            "company_name",
            "close",
            "return_5d",
            "drawdown_20d",
            "volatility_5d",
            "risk_probability",
            "risk_label",
            "model_name",
        ]
        if col in latest.columns
    ]
    st.dataframe(latest[show_cols], use_container_width=True, hide_index=True, height=420)

    chart_df = latest.copy()
    chart_df["risk_probability"] = pd.to_numeric(chart_df["risk_probability"], errors="coerce")
    st.bar_chart(chart_df.head(25), x="symbol", y="risk_probability", height=300)


def render_tab_demo() -> None:
    st.markdown('<p class="model5-section">Demo du doan rui ro theo ma/ngay</p>', unsafe_allow_html=True)
    st.caption("Model lay 23 features tu stock.features_all va tra ve xac suat HIGH_RISK.")

    try:
        symbols = get_symbols_for_demo()
    except Exception as exc:
        display_error("Khong tai duoc danh sach symbol tu features_all.", exc)
        return

    if not symbols:
        st.warning("Chua co du lieu features_all de demo.")
        return

    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col1:
        symbol = st.selectbox(
            "Ma co phieu",
            symbols,
            index=symbols.index("FPT") if "FPT" in symbols else 0,
        )
    with col2:
        dates = get_dates_for_symbol(symbol)
        if not dates:
            st.warning(f"Khong co ngay feature cho {symbol}.")
            return
        selected_date = st.selectbox("Ngay feature", dates, index=0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pred = st.button("Du doan risk", type="primary", use_container_width=True)

    if not run_pred:
        st.info("Chon ma, ngay feature roi bam du doan de xem xac suat rui ro.")
        return

    try:
        feature_df = get_features_for_demo(symbol, str(selected_date))
        if feature_df.empty:
            st.error(f"Khong tim thay feature cho {symbol} ngay {selected_date}.")
            return

        saved = load_model5()
        model = saved.get("model", saved)
        features = saved.get("features", FEATURE_COLUMNS)
        threshold = float(saved.get("threshold", DEFAULT_THRESHOLD))
    except Exception as exc:
        display_error("Khong tai duoc feature hoac model5 pkl.", exc)
        return

    missing_features = [feature for feature in features if feature not in feature_df.columns]
    if missing_features:
        st.error("Thieu feature: " + ", ".join(missing_features))
        return

    X = feature_df[features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if X.isna().any(axis=None):
        st.warning("Mot so feature bi null, dashboard tam thoi fill 0 de demo inference.")
        X = X.fillna(0)

    try:
        probability = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        display_error("Model khong tra ve predict_proba duoc.", exc)
        return

    risk_label = "HIGH_RISK" if probability >= threshold else "LOW_RISK"
    margin = probability - threshold

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", symbol)
    c2.metric("Ngay prediction", str(selected_date))
    c3.metric("Risk probability", pct(probability))
    c4.metric("Nhan risk", risk_label, delta=f"{margin * 100:.2f}% so voi nguong")

    if risk_label == "HIGH_RISK":
        st.error("Model dang canh bao HIGH_RISK: xac suat giam manh trong 5 phien vuot nguong.")
    else:
        st.success("Model danh gia LOW_RISK theo nguong hien tai.")

    existing = get_existing_alert_for_symbol_date(symbol, str(selected_date))
    if not existing.empty:
        st.markdown("**Ban ghi mart da co tren ClickHouse:**")
        keep_cols = [
            col
            for col in [
                "symbol",
                "prediction_date",
                "target_date",
                "close",
                "risk_probability",
                "risk_label",
                "model_name",
            ]
            if col in existing.columns
        ]
        st.dataframe(existing[keep_cols], use_container_width=True, hide_index=True)

    with st.expander("Xem 23 features dua vao Model 5"):
        feature_rows = []
        for feature in features:
            value = float(X[feature].iloc[0]) if feature in X.columns else np.nan
            feature_rows.append(
                {
                    "feature": feature,
                    "mo_ta": FEATURE_DESCRIPTIONS.get(feature, ""),
                    "gia_tri": value,
                }
            )
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)


def render_tab_daily() -> None:
    st.markdown('<p class="model5-section">Phan tich rui ro theo ngay</p>', unsafe_allow_html=True)

    try:
        daily_alert = load_daily_alert_summary()
        daily_eval = load_evaluation_daily()
    except Exception as exc:
        display_error("Khong tai duoc daily summary.", exc)
        return

    if daily_alert.empty:
        st.info("Chua co daily summary tu mart risk alert.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("So ngay", number(len(daily_alert)))
    c2.metric("Tong HIGH_RISK", number(daily_alert["high_risk_count"].sum()))
    c3.metric("HIGH_RISK/ngay TB", number(daily_alert["high_risk_count"].mean()))
    c4.metric("Risk prob TB", pct(daily_alert["avg_risk_probability"].mean()))

    chart_df = daily_alert.copy()
    chart_df = chart_df.sort_values("prediction_date")
    st.subheader("So luong HIGH_RISK theo ngay")
    st.line_chart(chart_df, x="prediction_date", y="high_risk_count", height=280)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ty le HIGH_RISK")
        st.area_chart(chart_df, x="prediction_date", y="high_risk_ratio", height=260)
    with col2:
        st.subheader("Risk probability trung binh")
        st.line_chart(chart_df, x="prediction_date", y="avg_risk_probability", height=260)

    if not daily_eval.empty:
        st.subheader("Accuracy test theo ngay")
        st.line_chart(daily_eval, x="prediction_date", y="accuracy", height=260)

    with st.expander("Bang daily summary"):
        st.dataframe(daily_alert, use_container_width=True, hide_index=True)


def render_tab_sector_symbol() -> None:
    st.markdown('<p class="model5-section">Rui ro theo nganh va theo ma</p>', unsafe_allow_html=True)

    try:
        sector_df = load_sector_summary()
        symbol_df = load_symbol_summary(limit=100)
    except Exception as exc:
        display_error("Khong tai duoc tong hop nganh/symbol.", exc)
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nganh co nhieu HIGH_RISK")
        if sector_df.empty:
            st.info("Chua co sector summary.")
        else:
            st.bar_chart(sector_df.head(20), x="sector", y="high_risk_count", height=320)
            st.dataframe(sector_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Ma co rui ro lap lai cao")
        if symbol_df.empty:
            st.info("Chua co symbol summary.")
        else:
            st.bar_chart(symbol_df.head(25), x="symbol", y="max_risk_probability", height=320)
            st.dataframe(symbol_df, use_container_width=True, hide_index=True)


def render_tab_model() -> None:
    st.markdown('<p class="model5-section">Chi tiet mo hinh</p>', unsafe_allow_html=True)

    metrics = load_metrics_json()
    selected_model = metrics.get("selected_model")
    model_metrics = metrics.get("models", {}).get(selected_model, {}) if selected_model else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected model", selected_model or "N/A")
    c2.metric("Threshold", pct(metrics.get("threshold", DEFAULT_THRESHOLD)))
    c3.metric("Train rows", number(metrics.get("train_rows")))
    c4.metric("Test rows", number(metrics.get("test_rows")))
    c5.metric("Cutoff date", str(metrics.get("cutoff_date", "N/A")))

    if model_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", pct(model_metrics.get("accuracy")))
        m2.metric("Precision HIGH_RISK", pct(model_metrics.get("precision_high_risk")))
        m3.metric("Recall HIGH_RISK", pct(model_metrics.get("recall_high_risk")))
        m4.metric("F1 HIGH_RISK", pct(model_metrics.get("f1_high_risk")))
        m5.metric("ROC-AUC", pct(model_metrics.get("roc_auc")))

    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        st.subheader("Feature importance")
        fi_df = load_feature_importance_frame()
        if fi_df.empty:
            st.info("Chua co feature_importance.csv hoac bang mart feature importance.")
        else:
            fi_df["importance"] = pd.to_numeric(fi_df["importance"], errors="coerce")
            chart = fi_df.sort_values("importance", ascending=True).tail(30)
            st.bar_chart(chart, x="feature", y="importance", height=520)
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Confusion matrix")
        matrix = model_metrics.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            cm_df = pd.DataFrame(
                matrix,
                index=["Actual LOW_RISK", "Actual HIGH_RISK"],
                columns=["Pred LOW_RISK", "Pred HIGH_RISK"],
            )
            st.dataframe(cm_df, use_container_width=True)
        else:
            st.info("Chua co confusion_matrix trong risk_metrics.json.")

        st.subheader("Thong tin label")
        st.markdown(
            """
            - Bai toan: Binary Classification / Risk Alert.
            - Nhan `HIGH_RISK`: future_return_5d <= -5%.
            - Nguong canh bao mac dinh: risk_probability >= 60%.
            - Chia train/test theo thoi gian, khong shuffle de tranh data leakage.
            """
        )

    with st.expander("Danh sach features Model 5"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"feature": feature, "mo_ta": FEATURE_DESCRIPTIONS.get(feature, "")}
                    for feature in FEATURE_COLUMNS
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_dashboard(embed: bool = False) -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#7f1d1d,#dc2626);
                    padding:26px 30px;border-radius:8px;margin-bottom:22px">
            <h1 style="color:white;margin:0;font-size:28px">
                Module 5 - Risk Alert Model
            </h1>
            <p style="color:rgba(255,255,255,0.88);margin:8px 0 0 0;font-size:15px">
                Binary Classification · Canh bao co phieu co nguy co giam manh trong 5 phien toi
                · XGBoost/Logistic baseline · 23 features tu stock.features_all
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_target = st.expander("Thong tin Module 5", expanded=False) if embed else st.sidebar
    info_target.markdown("### Module 5")
    info_target.markdown(
        """
        **Bai toan:** Risk Alert / Binary Classification  
        **Nhan:** HIGH_RISK neu future_return_5d <= -5%  
        **Nguong:** risk_probability >= 60%  
        **Nguon du lieu:** ClickHouse `features_all` va mart model5  
        **Output:** risk_probability, risk_label
        """
    )
    info_target.markdown("**Cach doc ket qua:**")
    info_target.info("HIGH_RISK nghia la model canh bao xac suat giam manh vuot nguong.")
    info_target.warning("LOW_RISK khong co nghia la chac chan an toan, chi la xac suat rui ro thap hon nguong.")
    info_target.caption(f"Feature table: {FEATURES_DATABASE}.{FEATURES_TABLE}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Tong quan",
            "Demo du doan",
            "Theo ngay",
            "Theo nganh/ma",
            "Model details",
        ]
    )

    with tab1:
        render_tab_overview()
    with tab2:
        render_tab_demo()
    with tab3:
        render_tab_daily()
    with tab4:
        render_tab_sector_symbol()
    with tab5:
        render_tab_model()


def main() -> None:
    render_dashboard(embed=False)


if __name__ == "__main__":
    main()
