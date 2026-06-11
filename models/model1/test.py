from pathlib import Path
import os

import pandas as pd

from src.config import HORIZON, MART_MODEL1_PRICE_FORECAST_PATH


REQUIRED_COLUMNS = [
    "prediction_date",
    "target_date",
    "symbol",
    "real_close",
    "predicted_close",
]

DISPLAY_COLUMNS = [
    "symbol",
    "prediction_date",
    "target_date",
    "real_close",
    "predicted_close",
    "price_error",
    "error_pct",
]


def prepare_forecast_data(forecasts_df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in forecasts_df.columns]
    if missing_columns:
        raise ValueError("Missing model1 forecast columns: " + ", ".join(missing_columns))

    df = forecasts_df.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")

    numeric_columns = ["real_close", "predicted_close"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["prediction_date", "symbol", "predicted_close"])
    return df


def load_forecasts(path=MART_MODEL1_PRICE_FORECAST_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model1 forecast mart not found: {path}. Run python main.py first."
        )

    return prepare_forecast_data(pd.read_csv(path))


def available_prediction_dates(forecasts_df):
    dates = forecasts_df["prediction_date"].dropna().dt.date.unique().tolist()
    return sorted(dates)


def filter_forecasts_by_date(forecasts_df, selected_date):
    selected_timestamp = pd.Timestamp(selected_date).normalize()
    prediction_dates = forecasts_df["prediction_date"].dt.normalize()
    day_df = forecasts_df[prediction_dates == selected_timestamp].copy()
    return day_df.sort_values("symbol").reset_index(drop=True)


def build_daily_summary(day_df):
    if day_df.empty:
        return {
            "num_symbols": 0,
            "target_date": None,
            "avg_predicted_close": None,
            "median_predicted_close": None,
        }

    target_dates = day_df["target_date"].dropna().dt.date.unique().tolist()
    target_date = target_dates[0].isoformat() if len(target_dates) == 1 else "Multiple"
    return {
        "num_symbols": int(len(day_df)),
        "target_date": target_date,
        "avg_predicted_close": float(day_df["predicted_close"].mean()),
        "median_predicted_close": float(day_df["predicted_close"].median()),
    }


def build_display_table(day_df):
    display_df = day_df.copy()
    display_df["price_error"] = display_df["predicted_close"] - display_df["real_close"]
    display_df["error_pct"] = (
        display_df["price_error"].abs() / display_df["real_close"].abs() * 100
    )
    return display_df[DISPLAY_COLUMNS]


def format_number(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.2f}"


def format_percent_value(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def main():
    import streamlit as st

    if os.getenv("MODEL1_DASHBOARD_EMBEDDED") != "1":
        st.set_page_config(page_title="Model1 Future Close", layout="wide")
    st.title("Model1 Future Close")

    @st.cache_data(show_spinner=False)
    def cached_load_forecasts(path):
        return load_forecasts(Path(path))

    try:
        forecasts_df = cached_load_forecasts(str(MART_MODEL1_PRICE_FORECAST_PATH))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    prediction_dates = available_prediction_dates(forecasts_df)
    if not prediction_dates:
        st.warning("No model1 forecast dates found.")
        st.stop()

    selected_date = st.date_input(
        "Prediction date",
        value=prediction_dates[-1],
        min_value=prediction_dates[0],
        max_value=prediction_dates[-1],
    )

    day_df = filter_forecasts_by_date(forecasts_df, selected_date)
    if day_df.empty:
        st.warning("No model1 result for this date.")
        st.stop()

    summary = build_daily_summary(day_df)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Symbols", f"{summary['num_symbols']:,}")
    metric_cols[1].metric("Target date", summary["target_date"] or "-")
    metric_cols[2].metric(
        "Avg predicted close",
        format_number(summary["avg_predicted_close"]),
    )
    metric_cols[3].metric(
        "Median predicted close",
        format_number(summary["median_predicted_close"]),
    )

    st.caption(f"Model1 predicts future close after {HORIZON} trading sessions.")

    display_df = build_display_table(day_df)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Predicted close by symbol")
    chart_df = display_df.set_index("symbol")[["predicted_close"]].head(30)
    st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
