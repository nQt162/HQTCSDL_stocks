from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL_NAME = "model3"
SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]


def _created_at():
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _latest_date(df):
    if df.empty or "trading_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["trading_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def load_sector_mapping(path):
    if not path:
        return None

    mapping_path = Path(path)
    if not mapping_path.exists():
        return None

    mapping_df = pd.read_csv(mapping_path)
    required_columns = {"encode_sector", "sector"}
    if not required_columns.issubset(mapping_df.columns):
        return None

    mapping_df = mapping_df[["encode_sector", "sector"]].drop_duplicates()
    mapping_df["encode_sector"] = pd.to_numeric(
        mapping_df["encode_sector"], errors="coerce"
    )
    return mapping_df.dropna(subset=["encode_sector"])


def add_sector_name(df, sector_mapping_df=None):
    result_df = df.copy()
    if "sector" in result_df.columns:
        return result_df

    if "encode_sector" not in result_df.columns:
        return result_df

    if sector_mapping_df is None or sector_mapping_df.empty:
        result_df["sector"] = result_df["encode_sector"].astype(str)
        return result_df

    result_df["encode_sector"] = pd.to_numeric(
        result_df["encode_sector"], errors="coerce"
    )
    return result_df.merge(
        sector_mapping_df,
        on="encode_sector",
        how="left",
    )


def build_trading_signals_mart(predictions_df, sector_mapping_df=None):
    required_columns = [
        "trading_date",
        "symbol",
        "predicted_signal",
        "adjusted_signal",
        "sell_probability",
        "hold_probability",
        "buy_probability",
    ]
    missing_columns = [
        col for col in required_columns if col not in predictions_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing model3 trading signal mart columns: "
            + ", ".join(missing_columns)
        )

    enriched_df = add_sector_name(predictions_df, sector_mapping_df)

    optional_columns = [
        "encode_sector",
        "sector",
        "close",
        "volume",
        "target_signal",
        "target_return",
        "predicted_signal_score",
        "signal_confidence",
        "buy_sell_margin",
    ]
    columns = required_columns + [
        col for col in optional_columns if col in enriched_df.columns
    ]
    mart_df = enriched_df[columns].copy()
    mart_df["trading_date"] = pd.to_datetime(
        mart_df["trading_date"], errors="coerce"
    ).dt.date
    mart_df.insert(0, "model_name", MODEL_NAME)
    mart_df["created_at"] = _created_at()
    return mart_df


def build_signal_summary_mart(predictions_df, sector_mapping_df=None):
    required_columns = ["trading_date", "adjusted_signal"]
    missing_columns = [
        col for col in required_columns if col not in predictions_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing model3 signal summary columns: " + ", ".join(missing_columns)
        )

    df = add_sector_name(predictions_df, sector_mapping_df)
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date

    group_columns = ["trading_date"]
    if "encode_sector" in df.columns:
        group_columns.append("encode_sector")
    if "sector" in df.columns:
        group_columns.append("sector")

    summary = (
        df.groupby(group_columns + ["adjusted_signal"], dropna=False)
        .size()
        .unstack("adjusted_signal", fill_value=0)
        .reset_index()
    )
    for signal in SIGNAL_NAMES:
        if signal not in summary.columns:
            summary[signal] = 0

    summary = summary.rename(
        columns={
            "SELL": "sell_count",
            "HOLD": "hold_count",
            "BUY": "buy_count",
        }
    )
    summary["total_symbols"] = (
        summary["sell_count"] + summary["hold_count"] + summary["buy_count"]
    )
    summary.insert(0, "model_name", MODEL_NAME)
    summary["created_at"] = _created_at()
    return summary


def build_backtest_daily_mart(backtest_df):
    mart_df = backtest_df.copy()
    if "trading_date" in mart_df.columns:
        mart_df["trading_date"] = pd.to_datetime(
            mart_df["trading_date"], errors="coerce"
        ).dt.date
    mart_df.insert(0, "model_name", MODEL_NAME)
    mart_df["created_at"] = _created_at()
    return mart_df


def build_metrics_mart(metrics, backtest_metrics=None):
    rows = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (list, dict)):
            continue
        rows.append(
            {
                "model_name": MODEL_NAME,
                "metric_group": "classification",
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
        )

    for metric_name, metric_value in (backtest_metrics or {}).items():
        if isinstance(metric_value, (list, dict)):
            continue
        rows.append(
            {
                "model_name": MODEL_NAME,
                "metric_group": "backtest",
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    metrics_df["created_at"] = _created_at()
    return metrics_df


def build_daily_insights(predictions_df, metrics=None, backtest_metrics=None):
    metrics = metrics or {}
    backtest_metrics = backtest_metrics or {}
    latest_date = _latest_date(predictions_df)
    if latest_date is None:
        return pd.DataFrame(
            columns=[
                "insight_date",
                "insight_type",
                "source_model",
                "symbol",
                "sector",
                "severity",
                "metric_name",
                "metric_value",
                "title",
                "message",
                "created_at",
            ]
        )

    latest_df = predictions_df[
        pd.to_datetime(predictions_df["trading_date"], errors="coerce") == latest_date
    ].copy()
    created_at = _created_at()
    rows = []
    counts = latest_df["adjusted_signal"].value_counts()
    buy_count = int(counts.get("BUY", 0))
    sell_count = int(counts.get("SELL", 0))
    hold_count = int(counts.get("HOLD", 0))

    rows.append(
        {
            "insight_date": latest_date.date(),
            "insight_type": "model3",
            "source_model": MODEL_NAME,
            "symbol": None,
            "sector": None,
            "severity": "success" if buy_count else "info",
            "metric_name": "adjusted_buy_count",
            "metric_value": buy_count,
            "title": "Model3 BUY signal count",
            "message": (
                f"Model3 has {buy_count} strong BUY signals, {sell_count} strong "
                f"SELL signals, and {hold_count} HOLD signals on {latest_date.date()}."
            ),
            "created_at": created_at,
        }
    )

    if "buy_probability" in latest_df.columns and not latest_df.empty:
        top_buy = latest_df.sort_values("buy_probability", ascending=False).iloc[0]
        rows.append(
            {
                "insight_date": latest_date.date(),
                "insight_type": "model3",
                "source_model": MODEL_NAME,
                "symbol": top_buy.get("symbol"),
                "sector": top_buy.get("sector", top_buy.get("encode_sector")),
                "severity": "success",
                "metric_name": "buy_probability",
                "metric_value": top_buy.get("buy_probability"),
                "title": "Top BUY probability",
                "message": (
                    f"{top_buy.get('symbol')} has the highest BUY probability "
                    f"for Model3 on {latest_date.date()}."
                ),
                "created_at": created_at,
            }
        )

    if "sell_probability" in latest_df.columns and not latest_df.empty:
        top_sell = latest_df.sort_values("sell_probability", ascending=False).iloc[0]
        rows.append(
            {
                "insight_date": latest_date.date(),
                "insight_type": "model3",
                "source_model": MODEL_NAME,
                "symbol": top_sell.get("symbol"),
                "sector": top_sell.get("sector", top_sell.get("encode_sector")),
                "severity": "warning",
                "metric_name": "sell_probability",
                "metric_value": top_sell.get("sell_probability"),
                "title": "Top SELL risk",
                "message": (
                    f"{top_sell.get('symbol')} has the highest SELL probability "
                    f"for Model3 on {latest_date.date()}."
                ),
                "created_at": created_at,
            }
        )

    if "Accuracy" in metrics:
        rows.append(
            {
                "insight_date": latest_date.date(),
                "insight_type": "model3",
                "source_model": MODEL_NAME,
                "symbol": None,
                "sector": None,
                "severity": "info",
                "metric_name": "accuracy",
                "metric_value": metrics.get("Accuracy"),
                "title": "Model3 classification accuracy",
                "message": (
                    "Model3 test accuracy is "
                    f"{float(metrics.get('Accuracy')):.2f}%."
                ),
                "created_at": created_at,
            }
        )

    if "Cumulative_Return_Net" in backtest_metrics:
        rows.append(
            {
                "insight_date": latest_date.date(),
                "insight_type": "model3",
                "source_model": MODEL_NAME,
                "symbol": None,
                "sector": None,
                "severity": "success",
                "metric_name": "cumulative_return_net",
                "metric_value": backtest_metrics.get("Cumulative_Return_Net"),
                "title": "Model3 backtest net return",
                "message": (
                    "Model3 top BUY backtest net cumulative return is "
                    f"{float(backtest_metrics.get('Cumulative_Return_Net')):.2%}."
                ),
                "created_at": created_at,
            }
        )

    return pd.DataFrame(rows)


def save_model3_marts(
    predictions_df,
    backtest_df,
    metrics,
    backtest_metrics,
    trading_signals_path,
    signal_summary_path,
    backtest_daily_path,
    metrics_path,
    daily_insights_path,
    sector_mapping_path=None,
):
    sector_mapping_df = load_sector_mapping(sector_mapping_path)
    enriched_predictions_df = add_sector_name(predictions_df, sector_mapping_df)

    build_trading_signals_mart(
        enriched_predictions_df, sector_mapping_df=sector_mapping_df
    ).to_csv(
        trading_signals_path, index=False
    )
    build_signal_summary_mart(
        enriched_predictions_df, sector_mapping_df=sector_mapping_df
    ).to_csv(signal_summary_path, index=False)
    build_backtest_daily_mart(backtest_df).to_csv(backtest_daily_path, index=False)
    build_metrics_mart(metrics, backtest_metrics).to_csv(metrics_path, index=False)
    build_daily_insights(enriched_predictions_df, metrics, backtest_metrics).to_csv(
        daily_insights_path, index=False
    )
