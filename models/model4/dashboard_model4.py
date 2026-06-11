# =============================================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: dashboard_model4.py
# Chạy: streamlit run dashboard_model4.py
# =============================================================

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ==================== PATHS & CONFIG ====================
MODEL4_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL4_DIR.parents[1]   # models/model4 -> models -> project root
MODEL_PATH   = MODEL4_DIR / "models" / "benchmark_outperformance_lgbm.pkl"

# Tìm .env từ nhiều vị trí
for _env_candidate in [
    MODEL4_DIR / ".env",
    PROJECT_ROOT / ".env",
    MODEL4_DIR.parent / ".env",
    Path(".") / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break
else:
    load_dotenv()  # fallback: tìm tự động

DATABASE = "stock"

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

FEATURE_DESCRIPTIONS = {
    "encode_sector":       "Mã hóa ngành",
    "return_1d":           "Lợi suất 1 ngày",
    "return_3d":           "Lợi suất 3 ngày",
    "return_5d":           "Lợi suất 5 ngày",
    "return_10d":          "Lợi suất 10 ngày",
    "return_20d":          "Lợi suất 20 ngày",
    "ma_5":                "MA 5 ngày",
    "ma_20":               "MA 20 ngày",
    "ma_50":               "MA 50 ngày",
    "price_vs_ma20":       "Giá so với MA20",
    "ma5_vs_ma20":         "MA5 so với MA20",
    "volatility_5d":       "Biến động 5 ngày",
    "volatility_20d":      "Biến động 20 ngày",
    "volatility_change":   "Thay đổi biến động",
    "rolling_max_20d":     "Đỉnh 20 ngày",
    "drawdown_20d":        "Drawdown 20 ngày",
    "volume_ma_5":         "Trung bình KL 5 ngày",
    "volume_ma_20":        "Trung bình KL 20 ngày",
    "volume_ratio_5_20":   "Tỷ lệ KL 5/20",
    "volume_change_1d":    "Thay đổi KL 1 ngày",
    "daily_range":         "Biên độ trong ngày",
    "body_ratio":          "Tỷ lệ thân nến",
    "close_position":      "Vị trí giá đóng cửa",
}

# ==================== PAGE CONFIG ====================
if os.getenv("MODEL4_DASHBOARD_EMBEDDED") != "1":
    st.set_page_config(
        page_title="Module 4 — Benchmark Outperformance",
        page_icon="📈",
        layout="wide",
    )

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
body, .stMarkdown, .stDataFrame, p, div, span, td, th {
    font-size: 15px !important;
}
h1, h2, h3 { font-size: 20px !important; }
.stMetric label { font-size: 14px !important; }
.stMetric [data-testid="stMetricValue"] { font-size: 28px !important; }
.stSelectbox label, .stDateInput label { font-size: 15px !important; }
.stButton button { font-size: 16px !important; padding: 10px 20px !important; }
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 5px solid #2a9d8f;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 8px;
}
.kpi-card .kpi-label {
    font-size: 13px;
    color: #5f6c7b;
    margin: 0 0 6px 0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-card .kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #1d3557;
    margin: 0;
}
.kpi-card .kpi-sub {
    font-size: 12px;
    color: #5f6c7b;
    margin: 4px 0 0 0;
}
.op-badge-yes {
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 18px;
}
.op-badge-no {
    background: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 18px;
}
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: #1d3557;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8eef2;
}
</style>
""", unsafe_allow_html=True)

# ==================== HELPERS ====================
def pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "N/A"

def number(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"

def display_error(message: str, exc: Exception) -> None:
    st.error(f"⚠️ {message}")
    st.caption(str(exc))

def sql_str(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"

# ==================== CLICKHOUSE ====================
@st.cache_resource
def get_clickhouse_client():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise RuntimeError("Thiếu clickhouse-connect. Cài: pip install clickhouse-connect") from exc

    host     = os.getenv("CLICKHOUSE_HOST")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD")
    port     = int(os.getenv("CLICKHOUSE_PORT") or "8443")
    secure   = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {"1","true","yes"}

    if not host or not password:
        raise RuntimeError("Thiếu CLICKHOUSE_HOST hoặc CLICKHOUSE_PASSWORD trong .env")

    return clickhouse_connect.get_client(
        host=host, port=port, username=username, password=password,
        database=os.getenv("CLICKHOUSE_DATABASE", "default"), secure=secure,
    )

@st.cache_data(ttl=300, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    return get_clickhouse_client().query_df(query)

# ==================== DATA LOADERS ====================
@st.cache_data(ttl=300, show_spinner=False)
def load_top_outperformers() -> pd.DataFrame:
    return run_query(f"""
        SELECT rank, symbol, trading_date, close,
               outperform_probability, predicted_label
        FROM {DATABASE}.mart_model4_top_outperformers
        ORDER BY rank ASC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_daily_summary() -> pd.DataFrame:
    df = run_query(f"""
        SELECT trading_date, total_symbols,
               predicted_outperform, actual_outperform,
               avg_probability, accuracy_daily, outperform_ratio
        FROM {DATABASE}.mart_model4_daily_outperform_summary
        ORDER BY trading_date ASC
    """)
    if not df.empty and "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_sector_outperform() -> pd.DataFrame:
    return run_query(f"""
        SELECT encode_sector, total_symbols,
               predicted_outperform, actual_outperform,
               avg_probability, accuracy_sector, outperform_ratio
        FROM {DATABASE}.mart_model4_sector_outperform
        ORDER BY accuracy_sector DESC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_metrics_history() -> pd.DataFrame:
    df = run_query(f"""
        SELECT run_at, accuracy, precision, recall, f1, roc_auc,
               test_rows, train_ratio
        FROM {DATABASE}.mart_model4_metrics
        ORDER BY run_at DESC
    """)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return run_query(f"""
        SELECT feature, importance
        FROM {DATABASE}.mart_model4_feature_importance
        ORDER BY importance DESC
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_predictions_sample(limit: int = 5000) -> pd.DataFrame:
    df = run_query(f"""
        SELECT symbol, trading_date, label, predicted_label,
               outperform_probability, prediction_correct
        FROM {DATABASE}.mart_model4_benchmark_outperformance
        ORDER BY trading_date DESC, outperform_probability DESC
        LIMIT {limit}
    """)
    if not df.empty and "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def get_symbols_for_demo() -> list[str]:
    result = run_query(f"""
        SELECT DISTINCT symbol
        FROM {DATABASE}.mart_model4_benchmark_outperformance
        ORDER BY symbol
        LIMIT 1000
    """)
    if result.empty or "symbol" not in result.columns:
        return []
    return result["symbol"].astype(str).tolist()

@st.cache_data(ttl=300, show_spinner=False)
def get_dates_for_symbol(symbol: str) -> list:
    result = run_query(f"""
        SELECT DISTINCT toDate(trading_date) AS d
        FROM {DATABASE}.mart_model4_benchmark_outperformance
        WHERE symbol = {sql_str(symbol)}
        ORDER BY d DESC
        LIMIT 100
    """)
    if result.empty:
        return []
    return pd.to_datetime(result["d"]).dt.date.tolist()

@st.cache_data(ttl=300, show_spinner=False)
def get_features_for_demo(symbol: str, trading_date) -> pd.DataFrame:
    return run_query(f"""
        SELECT *
        FROM {DATABASE}.features_all
        WHERE upper(trim(symbol)) = upper(trim({sql_str(symbol)}))
          AND toDate(trading_date) = toDate({sql_str(trading_date)})
        LIMIT 1
    """)

# ==================== MODEL ====================
@st.cache_resource
def load_model4():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    saved = joblib.load(MODEL_PATH)
    return saved

# ==================== KPI CARD ====================
def kpi_card(label, value, sub="", color="#2a9d8f"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        <p class="kpi-sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 1: TỔNG QUAN ====================
def render_tab_overview():
    st.markdown('<p class="section-header">📊 Chỉ số hiệu suất mô hình</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("🎯 Accuracy",  "57.28%", "Tỷ lệ đoán đúng tổng thể", "#2a9d8f")
    with c2: kpi_card("🔍 Precision", "56.55%", "Khi hô OP, đúng 56.55%",   "#1d6fa4")
    with c3: kpi_card("📡 Recall",    "61.20%", "Bắt được 61.2% OP thật",   "#6a4c93")
    with c4: kpi_card("⚖️ F1-Score",  "58.78%", "Cân bằng Precision/Recall", "#e76f51")
    with c5: kpi_card("📈 ROC-AUC",   "60.85%", "Tốt hơn đoán mò 10.85%",  "#2d6a4f")

    st.markdown('<p class="section-header">🏆 Top 20 Cổ phiếu Outperform ngày mới nhất</p>', unsafe_allow_html=True)

    try:
        top_df = load_top_outperformers()
    except Exception as exc:
        display_error("Không tải được top outperformers.", exc)
        return

    if top_df.empty:
        st.info("Chưa có dữ liệu top outperformers.")
        return

    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        chart_df = top_df[["symbol","outperform_probability"]].copy()
        chart_df["outperform_probability"] = pd.to_numeric(
            chart_df["outperform_probability"], errors="coerce")
        chart_df = chart_df.sort_values("outperform_probability", ascending=True)
        chart_df.columns = ["Mã CP", "Xác suất OP"]
        st.bar_chart(chart_df.set_index("Mã CP"), color="#2a9d8f", height=450)

    with col_table:
        display = top_df.copy()
        display["outperform_probability"] = display["outperform_probability"].apply(
            lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "N/A")
        display["close"] = display["close"].apply(
            lambda x: f"{float(x):,.0f}" if pd.notna(x) else "N/A")
        display["Nhận định"] = display["predicted_label"].apply(
            lambda x: "✅ OP" if int(x) == 1 else "❌ Không OP")
        display = display.rename(columns={
            "rank": "#", "symbol": "Mã", "close": "Giá",
            "outperform_probability": "Xác suất OP",
        })
        cols_show = ["#", "Mã", "Giá", "Xác suất OP", "Nhận định"]
        st.dataframe(display[cols_show], use_container_width=True, hide_index=True, height=450)

    st.markdown('<p class="section-header">📋 Tóm tắt từ predictions gần nhất</p>', unsafe_allow_html=True)
    try:
        pred_df = load_predictions_sample(limit=2000)
        if not pred_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            total = len(pred_df)
            correct = int(pred_df["prediction_correct"].sum())
            op_pred = int((pred_df["predicted_label"]==1).sum())
            avg_prob = float(pred_df["outperform_probability"].mean())

            c1.metric("📦 Tổng dự đoán (mẫu)", number(total))
            c2.metric("✅ Đoán đúng", f"{correct/total*100:.1f}%")
            c3.metric("🔥 Số lần hô OP", number(op_pred))
            c4.metric("📊 Xác suất TB", pct(avg_prob))
    except Exception as exc:
        st.caption(f"Không tải được thống kê predictions: {exc}")

# ==================== TAB 2: DEMO DỰ ĐOÁN ====================
def render_tab_demo():
    st.markdown('<p class="section-header">🎯 Demo dự đoán thời gian thực</p>', unsafe_allow_html=True)
    st.caption("Chọn mã cổ phiếu và ngày → model tự lấy features từ ClickHouse → dự đoán xác suất Outperform")

    try:
        symbols = get_symbols_for_demo()
    except Exception as exc:
        display_error("Không tải được danh sách mã.", exc)
        return

    if not symbols:
        st.warning("Chưa có dữ liệu trong mart.")
        return

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        symbol = st.selectbox("🔎 Mã cổ phiếu",
            symbols, index=symbols.index("FPT") if "FPT" in symbols else 0)
    with col2:
        try:
            dates = get_dates_for_symbol(symbol)
        except Exception:
            dates = []
        if dates:
            selected_date = st.selectbox("📅 Ngày giao dịch", dates,
                format_func=lambda d: str(d))
        else:
            st.warning("Không có ngày cho mã này.")
            return
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pred = st.button("🚀 Dự đoán", type="primary", use_container_width=True)

    if not run_pred:
        st.info("👆 Chọn mã và ngày rồi bấm **Dự đoán** để xem kết quả.")
        return

    with st.spinner("Đang lấy features và chạy model..."):
        try:
            feature_df = get_features_for_demo(symbol, str(selected_date))
        except Exception as exc:
            display_error("Không lấy được features từ ClickHouse.", exc)
            return

        if feature_df.empty:
            st.error(f"Không tìm thấy features cho {symbol} ngày {selected_date}.")
            return

        try:
            saved   = load_model4()
            model   = saved["model"]
            features = saved.get("features", FEATURE_COLUMNS)
        except Exception as exc:
            display_error("Không tải được model pkl.", exc)
            return

        missing = [f for f in features if f not in feature_df.columns]
        if missing:
            st.error(f"Thiếu features: {', '.join(missing)}")
            return

        X = feature_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        prob = float(model.predict_proba(X)[0][1])
        pred_label = int(model.predict(X)[0])
        is_op = pred_label == 1

    # ── Kết quả chính ──
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1])

    with res_col1:
        close_val = feature_df["close"].values[0] if "close" in feature_df.columns else None
        st.metric("📌 Mã cổ phiếu", symbol)
        st.metric("📅 Ngày", str(selected_date))
        if close_val is not None:
            st.metric("💰 Giá đóng cửa", f"{float(close_val):,.0f} đ")

    with res_col2:
        badge_class = "op-badge-yes" if is_op else "op-badge-no"
        badge_text  = "✅ OUTPERFORM" if is_op else "❌ KHÔNG OUTPERFORM"
        st.markdown(f"""
        <div style="text-align:center; padding:20px">
            <p style="font-size:14px;color:#5f6c7b;margin-bottom:8px">Nhận định</p>
            <span class="{badge_class}">{badge_text}</span>
            <p style="font-size:13px;color:#5f6c7b;margin-top:12px">
                Xác suất Outperform trong 5 ngày tới
            </p>
            <p style="font-size:48px;font-weight:700;color:{'#15803d' if is_op else '#b91c1c'};margin:0">
                {prob*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob)

    with res_col3:
        if is_op:
            st.success("🔥 Model tin tưởng cổ phiếu này sẽ vượt benchmark thị trường trong 5 ngày tới.")
        else:
            st.warning("⚠️ Model dự đoán cổ phiếu này sẽ không vượt benchmark thị trường.")

        threshold = 0.5
        margin = abs(prob - threshold)
        confidence = "Cao" if margin > 0.15 else "Trung bình" if margin > 0.07 else "Thấp"
        color_conf = "🟢" if confidence == "Cao" else "🟡" if confidence == "Trung bình" else "🔴"
        st.metric(f"{color_conf} Độ tin cậy", confidence)
        st.metric("📏 Khoảng cách ngưỡng", f"{margin*100:.1f}%")

    # ── Feature values ──
    with st.expander("🔬 Xem chi tiết 23 features đưa vào model"):
        feat_display = []
        for feat in features:
            val = float(X[feat].values[0]) if feat in X.columns else None
            feat_display.append({
                "Feature": feat,
                "Mô tả": FEATURE_DESCRIPTIONS.get(feat, ""),
                "Giá trị": f"{val:.6f}" if val is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(feat_display), use_container_width=True, hide_index=True)

    # ── So sánh với thực tế ──
    try:
        actual_df = run_query(f"""
            SELECT label, predicted_label, outperform_probability, prediction_correct
            FROM {DATABASE}.mart_model4_benchmark_outperformance
            WHERE upper(trim(symbol)) = upper(trim({sql_str(symbol)}))
              AND toDate(trading_date) = toDate({sql_str(str(selected_date))})
            LIMIT 1
        """)
        if not actual_df.empty:
            st.markdown("---")
            st.markdown("**📋 So sánh với kết quả thực tế trong tập test:**")
            row = actual_df.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Label thực tế", "✅ OP" if int(row["label"])==1 else "❌ Không OP")
            c2.metric("Model dự đoán", "✅ OP" if int(row["predicted_label"])==1 else "❌ Không OP")
            c3.metric("Xác suất (mart)", pct(float(row["outperform_probability"])))
            c4.metric("Kết quả", "✅ Đúng" if int(row["prediction_correct"])==1 else "❌ Sai")
    except Exception:
        pass

# ==================== TAB 3: THEO NGÀY ====================
def render_tab_daily():
    st.markdown('<p class="section-header">📅 Phân tích hiệu suất theo ngày</p>', unsafe_allow_html=True)

    try:
        daily_df = load_daily_summary()
    except Exception as exc:
        display_error("Không tải được daily summary.", exc)
        return

    if daily_df.empty:
        st.info("Chưa có dữ liệu daily summary.")
        return

    # ── KPIs tổng ──
    c1, c2, c3, c4 = st.columns(4)
    avg_acc    = daily_df["accuracy_daily"].mean()
    avg_op_r   = daily_df["outperform_ratio"].mean()
    total_days = len(daily_df)
    best_day   = daily_df.loc[daily_df["accuracy_daily"].idxmax(), "trading_date"]

    c1.metric("📆 Số ngày trong test", number(total_days))
    c2.metric("🎯 Accuracy trung bình", pct(avg_acc))
    c3.metric("📊 Tỷ lệ OP dự đoán TB", pct(avg_op_r))
    c4.metric("🏅 Ngày chính xác nhất", str(best_day)[:10])

    # ── Line chart accuracy ──
    st.markdown('<p class="section-header">📈 Accuracy theo từng ngày giao dịch</p>', unsafe_allow_html=True)
    chart_acc = daily_df[["trading_date","accuracy_daily"]].copy()
    chart_acc = chart_acc.set_index("trading_date")
    chart_acc.columns = ["Accuracy"]
    st.line_chart(chart_acc, color="#2a9d8f", height=300)

    # ── Predicted vs Actual ──
    st.markdown('<p class="section-header">📊 Số mã dự đoán OP vs. thực tế OP theo ngày</p>', unsafe_allow_html=True)

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        date_from = st.date_input("Từ ngày", value=daily_df["trading_date"].min().date(),
            min_value=daily_df["trading_date"].min().date(),
            max_value=daily_df["trading_date"].max().date(), key="daily_from")
    with col_date2:
        date_to = st.date_input("Đến ngày", value=daily_df["trading_date"].max().date(),
            min_value=daily_df["trading_date"].min().date(),
            max_value=daily_df["trading_date"].max().date(), key="daily_to")

    mask = (
        (daily_df["trading_date"].dt.date >= date_from) &
        (daily_df["trading_date"].dt.date <= date_to)
    )
    filtered = daily_df[mask].copy()

    if not filtered.empty:
        chart_vs = filtered[["trading_date","predicted_outperform","actual_outperform"]].copy()
        chart_vs = chart_vs.set_index("trading_date")
        chart_vs.columns = ["Dự đoán OP", "Thực tế OP"]
        st.line_chart(chart_vs, height=280)

    # ── Avg probability theo ngày ──
    st.markdown('<p class="section-header">📉 Xác suất trung bình dự đoán theo ngày</p>', unsafe_allow_html=True)
    chart_prob = daily_df[["trading_date","avg_probability"]].copy().set_index("trading_date")
    chart_prob.columns = ["Xác suất TB"]
    st.area_chart(chart_prob, color="#6a4c93", height=220)

    # ── Bảng chi tiết ──
    with st.expander("📋 Xem bảng dữ liệu đầy đủ (daily summary)"):
        show_df = daily_df.copy()
        for col in ["accuracy_daily","avg_probability","outperform_ratio"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].apply(
                    lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "N/A")
        show_df["trading_date"] = show_df["trading_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

# ==================== TAB 4: THEO NGÀNH ====================
def render_tab_sector():
    st.markdown('<p class="section-header">🏭 Phân tích hiệu suất theo ngành</p>', unsafe_allow_html=True)

    try:
        sector_df = load_sector_outperform()
    except Exception as exc:
        display_error("Không tải được sector data.", exc)
        return

    if sector_df.empty:
        st.info("Chưa có dữ liệu sector.")
        return

    sector_df["encode_sector"] = sector_df["encode_sector"].astype(str)
    sector_df["Ngành"] = "Ngành " + sector_df["encode_sector"]

    # ── KPIs ──
    best_sector = sector_df.loc[sector_df["accuracy_sector"].idxmax()]
    worst_sector = sector_df.loc[sector_df["accuracy_sector"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏭 Số ngành", number(len(sector_df)))
    c2.metric("🎯 Accuracy TB", pct(sector_df["accuracy_sector"].mean()))
    c3.metric("🥇 Ngành tốt nhất",
              f"Ngành {best_sector['encode_sector']} ({pct(best_sector['accuracy_sector'])})")
    c4.metric("⚠️ Ngành kém nhất",
              f"Ngành {worst_sector['encode_sector']} ({pct(worst_sector['accuracy_sector'])})")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**🎯 Accuracy theo ngành**")
        chart_acc = sector_df[["Ngành","accuracy_sector"]].copy().set_index("Ngành")
        chart_acc.columns = ["Accuracy"]
        st.bar_chart(chart_acc, color="#2a9d8f", height=350)

    with col_right:
        st.markdown("**📊 Tỷ lệ dự đoán Outperform theo ngành**")
        chart_ratio = sector_df[["Ngành","outperform_ratio"]].copy().set_index("Ngành")
        chart_ratio.columns = ["Tỷ lệ OP"]
        st.bar_chart(chart_ratio, color="#e76f51", height=350)

    # ── So sánh predicted vs actual ──
    st.markdown('<p class="section-header">📊 Dự đoán OP vs. Thực tế OP theo ngành</p>', unsafe_allow_html=True)
    cmp_df = sector_df[["Ngành","predicted_outperform","actual_outperform"]].copy().set_index("Ngành")
    cmp_df.columns = ["Dự đoán OP", "Thực tế OP"]
    st.bar_chart(cmp_df, height=300)

    # ── Bảng chi tiết ──
    st.markdown('<p class="section-header">📋 Bảng chi tiết theo ngành</p>', unsafe_allow_html=True)
    detail = sector_df.copy()
    for col in ["accuracy_sector","avg_probability","outperform_ratio"]:
        if col in detail.columns:
            detail[col] = detail[col].apply(
                lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "N/A")
    detail = detail.rename(columns={
        "encode_sector":       "Mã ngành",
        "total_symbols":       "Tổng mã",
        "predicted_outperform":"Dự đoán OP",
        "actual_outperform":   "Thực tế OP",
        "avg_probability":     "Xác suất TB",
        "accuracy_sector":     "Accuracy",
        "outperform_ratio":    "Tỷ lệ OP",
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)

# ==================== TAB 5: MODEL DETAILS ====================
def render_tab_model():
    st.markdown('<p class="section-header">🔬 Chi tiết mô hình</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("**📊 Feature Importance (Top 23)**")
        try:
            fi_df = load_feature_importance()
            if not fi_df.empty:
                fi_df["importance"] = pd.to_numeric(fi_df["importance"], errors="coerce")
                fi_df = fi_df.sort_values("importance", ascending=True)
                chart_fi = fi_df.set_index("feature")[["importance"]]
                chart_fi.columns = ["Importance"]
                st.bar_chart(chart_fi, color="#1d6fa4", height=520)
            else:
                st.info("Chưa có dữ liệu feature importance.")
        except Exception as exc:
            display_error("Không tải được feature importance.", exc)

    with col_right:
        st.markdown("**📋 Thông tin model**")
        info_data = {
            "Thuật toán":   "LightGBM Classifier",
            "n_estimators": "300",
            "max_depth":    "4",
            "learning_rate":"0.05",
            "subsample":    "0.9",
            "colsample":    "0.9",
            "random_state": "42",
            "Horizon":      "5 ngày giao dịch",
            "Features":     "23 (22 kỹ thuật + ngành)",
            "Train size":   "529,958 dòng",
            "Test size":    "163,132 dòng",
            "Cutoff date":  "2024-04-17",
        }
        info_df = pd.DataFrame(list(info_data.items()), columns=["Tham số", "Giá trị"])
        st.dataframe(info_df, use_container_width=True, hide_index=True, height=430)

    # ── Metrics history ──
    st.markdown('<p class="section-header">📈 Lịch sử chạy pipeline (mart_model4_metrics)</p>', unsafe_allow_html=True)
    try:
        metrics_df = load_metrics_history()
        if not metrics_df.empty:
            display_metrics = metrics_df.copy()
            for col in ["accuracy","precision","recall","f1","roc_auc","train_ratio"]:
                if col in display_metrics.columns:
                    display_metrics[col] = display_metrics[col].apply(
                        lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "N/A")
            display_metrics["test_rows"] = display_metrics["test_rows"].apply(number)
            display_metrics = display_metrics.rename(columns={
                "run_at":     "Lần chạy",
                "accuracy":   "Accuracy",
                "precision":  "Precision",
                "recall":     "Recall",
                "f1":         "F1",
                "roc_auc":    "ROC-AUC",
                "test_rows":  "Test rows",
                "train_ratio":"Train ratio",
            })
            st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có lịch sử metrics.")
    except Exception as exc:
        display_error("Không tải được metrics history.", exc)

    # ── Confusion Matrix ──
    st.markdown('<p class="section-header">🔲 Confusion Matrix (tập test 163,132 dòng)</p>', unsafe_allow_html=True)

    cm_data = {
        "": ["Thực tế: Không OP (0)", "Thực tế: OP (1)"],
        "Đoán: Không OP (0)": ["✅ TN = 43,748", "❌ FN = 31,505"],
        "Đoán: OP (1)":       ["❌ FP = 38,185", "✅ TP = 49,694"],
    }
    cm_df = pd.DataFrame(cm_data).set_index("")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Ô xanh (TN, TP) = đoán đúng | Ô đỏ (FP, FN) = đoán sai")
    with c2:
        cm_metrics = [
            ["Accuracy",  "(43748+49694)/163132", "57.28%"],
            ["Precision", "49694/(49694+38185)",   "56.55%"],
            ["Recall",    "49694/(49694+31505)",    "61.20%"],
            ["F1",        "2×P×R/(P+R)",            "58.78%"],
            ["FPR",       "38185/(38185+43748)",     "46.60%"],
            ["TPR",       "= Recall",                "61.20%"],
        ]
        cm_meta_df = pd.DataFrame(cm_metrics, columns=["Metric","Công thức","Giá trị"])
        st.dataframe(cm_meta_df, use_container_width=True, hide_index=True)

    # ── Feature descriptions ──
    with st.expander("📚 Mô tả 23 features"):
        feat_desc_df = pd.DataFrame([
            {"Feature": k, "Mô tả": v} for k, v in FEATURE_DESCRIPTIONS.items()
        ])
        st.dataframe(feat_desc_df, use_container_width=True, hide_index=True)

# ==================== MAIN ====================
def main():
    # Header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1d3557,#2a9d8f);
                padding:28px 32px;border-radius:12px;margin-bottom:24px">
        <h1 style="color:white;margin:0;font-size:28px">
            📈 Module 4 — Benchmark Outperformance Model
        </h1>
        <p style="color:rgba(255,255,255,0.85);margin:8px 0 0 0;font-size:15px">
            LightGBM Classifier · Dự đoán cổ phiếu vượt trội thị trường trong 5 ngày tới
            · Train: 2015-12 → 2024-04 · Test: 2024-04 → 2026-06
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar info is useful when this file runs alone, but should not
    # overwrite the main application sidebar when embedded in main_web.py.
    if os.getenv("MODEL4_DASHBOARD_EMBEDDED") != "1":
        st.sidebar.markdown("### ℹ️ Module 4")
        st.sidebar.markdown("""
        **Bài toán:** Binary Classification  
        **Thuật toán:** LightGBM  
        **Horizon:** 5 ngày giao dịch  
        **Features:** 23  
        """)
        st.sidebar.divider()
        st.sidebar.markdown("**🗺️ Hướng dẫn sử dụng:**")
        st.sidebar.markdown("""
| Tab | Nội dung |
|-----|----------|
| 📊 Tổng quan | KPI + Top 20 cổ phiếu |
| 🎯 Demo | Dự đoán 1 mã cụ thể |
| 📅 Theo ngày | Trend 524 ngày test |
| 🏭 Theo ngành | So sánh 19 ngành |
| 🔬 Model | Feature importance |
""")
        st.sidebar.divider()
        st.sidebar.markdown("**💡 Cách đọc kết quả:**")
        st.sidebar.info("✅ Xác suất > 50% → Model dự đoán cổ phiếu sẽ **vượt benchmark** thị trường trong 5 ngày tới")
        st.sidebar.warning("⚠️ Xác suất < 50% → Model dự đoán cổ phiếu **không vượt** được thị trường")
        st.sidebar.divider()
        st.sidebar.caption("🏦 Dữ liệu: ClickHouse Cloud (AWS Singapore)")
        st.sidebar.caption("📅 Test period: 2024-04-17 → 2026-06-01")
        st.sidebar.caption("🤖 Model: LightGBM · 300 cây · 23 features")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng quan",
        "🎯 Demo dự đoán",
        "📅 Theo ngày",
        "🏭 Theo ngành",
        "🔬 Model Details",
    ])

    with tab1: render_tab_overview()
    with tab2: render_tab_demo()
    with tab3: render_tab_daily()
    with tab4: render_tab_sector()
    with tab5: render_tab_model()

if __name__ == "__main__":
    main()
