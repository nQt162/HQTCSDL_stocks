from pathlib import Path
from uuid import uuid4

import pandas as pd


MODEL_NAME = "model1"


def _created_at_value(created_at=None):
    if created_at is None:
        return pd.Timestamp.utcnow().tz_localize(None).floor("s")
    return pd.Timestamp(created_at)


def _model_run_id_value(model_run_id=None):
    return str(model_run_id or uuid4())


def build_price_forecast_mart(predictions_df, model_run_id=None, created_at=None):
    required_columns = [
        "trading_date",
        "future_trading_date",
        "symbol",
        "target_close",
        "predicted_close",
        "target_return",
        "predicted_return",
        "actual_direction",
        "predicted_direction",
    ]
    missing_columns = [col for col in required_columns if col not in predictions_df.columns]
    if missing_columns:
        raise ValueError(
            "Missing model1 prediction columns: " + ", ".join(missing_columns)
        )

    mart_df = pd.DataFrame(
        {
            "model_run_id": _model_run_id_value(model_run_id),
            "prediction_date": pd.to_datetime(predictions_df["trading_date"]),
            "target_date": pd.to_datetime(predictions_df["future_trading_date"]),
            "symbol": predictions_df["symbol"],
            "real_close": predictions_df["target_close"],
            "predicted_close": predictions_df["predicted_close"],
            "actual_return": predictions_df["target_return"],
            "predicted_return": predictions_df["predicted_return"],
            "direction_correct": (
                predictions_df["actual_direction"] == predictions_df["predicted_direction"]
            ).astype("Int64"),
            "model_name": MODEL_NAME,
            "created_at": _created_at_value(created_at),
        }
    )

    return mart_df[
        [
            "model_run_id",
            "prediction_date",
            "target_date",
            "symbol",
            "real_close",
            "predicted_close",
            "actual_return",
            "predicted_return",
            "direction_correct",
            "model_name",
            "created_at",
        ]
    ]


def build_top_expected_return_mart(price_forecast_df, top_n=10):
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    top_df = (
        price_forecast_df.sort_values(
            ["prediction_date", "predicted_return"],
            ascending=[True, False],
        )
        .groupby("prediction_date", group_keys=False)
        .head(top_n)
        .copy()
    )
    top_df.insert(
        2,
        "rank",
        top_df.groupby("prediction_date")["predicted_return"].rank(
            method="first", ascending=False
        ).astype(int),
    )
    return top_df.sort_values(["prediction_date", "rank"]).reset_index(drop=True)


def build_backtest_daily_mart(backtest_df, model_run_id=None, created_at=None):
    if "trading_date" not in backtest_df.columns:
        raise ValueError("Missing model1 backtest column: trading_date")

    mart_df = backtest_df.copy()
    mart_df.insert(0, "model_run_id", _model_run_id_value(model_run_id))
    mart_df["trading_date"] = pd.to_datetime(mart_df["trading_date"])
    mart_df["model_name"] = MODEL_NAME
    mart_df["created_at"] = _created_at_value(created_at)
    return mart_df


def build_metrics_mart(metrics, backtest_metrics, model_run_id=None, created_at=None):
    rows = []
    for group_name, metric_dict in [
        ("test", metrics),
        ("backtest", backtest_metrics),
    ]:
        for metric_name, metric_value in metric_dict.items():
            rows.append(
                {
                    "model_run_id": _model_run_id_value(model_run_id),
                    "model_name": MODEL_NAME,
                    "metric_group": group_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "created_at": _created_at_value(created_at),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "model_run_id",
            "model_name",
            "metric_group",
            "metric_name",
            "metric_value",
            "created_at",
        ],
    )


def build_daily_insights(price_forecast_df, backtest_metrics, model_run_id=None, created_at=None, top_n=10):
    created_at_value = _created_at_value(created_at)
    model_run_id_value = _model_run_id_value(model_run_id)
    if price_forecast_df.empty:
        return _empty_daily_insights()

    latest_prediction_date = price_forecast_df["prediction_date"].max()
    latest_top_df = (
        price_forecast_df[price_forecast_df["prediction_date"] == latest_prediction_date]
        .sort_values("predicted_return", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    rows = []
    for rank, row in enumerate(latest_top_df.itertuples(index=False), start=1):
        predicted_return_pct = float(row.predicted_return) * 100
        severity = "success" if row.predicted_return >= 0.02 else "info"
        rows.append(
            {
                "model_run_id": model_run_id_value,
                "insight_date": latest_prediction_date,
                "insight_type": "model1",
                "source_model": MODEL_NAME,
                "symbol": row.symbol,
                "sector": pd.NA,
                "severity": severity,
                "metric_name": "expected_return_5d",
                "metric_value": row.predicted_return,
                "title": f"Model1 top expected return #{rank}: {row.symbol}",
                "message": (
                    f"{row.symbol} ranks #{rank} by model1 predicted 5-session "
                    f"return at {predicted_return_pct:.2f}%."
                ),
                "created_at": created_at_value,
            }
        )

    cumulative_return = backtest_metrics.get("Cumulative_Return")
    benchmark_return = backtest_metrics.get("Benchmark_Cumulative_Return")
    if cumulative_return is not None and benchmark_return is not None:
        outperformance = cumulative_return - benchmark_return
        outperformance_pct = outperformance * 100
        verb = "outperformed" if outperformance >= 0 else "underperformed"
        rows.append(
            {
                "model_run_id": model_run_id_value,
                "insight_date": latest_prediction_date,
                "insight_type": "model1",
                "source_model": MODEL_NAME,
                "symbol": pd.NA,
                "sector": pd.NA,
                "severity": "success" if outperformance >= 0 else "warning",
                "metric_name": "model1_backtest_outperformance",
                "metric_value": outperformance,
                "title": "Model1 backtest vs benchmark",
                "message": (
                    f"Model1 {verb} benchmark by {abs(outperformance_pct):.2f} "
                    "percentage points in cumulative return."
                ),
                "created_at": created_at_value,
            }
        )

    return pd.DataFrame(rows, columns=_daily_insight_columns())


def build_model1_marts(
    predictions_df,
    backtest_df,
    metrics,
    backtest_metrics,
    model_run_id=None,
    created_at=None,
    top_n=10,
):
    model_run_id_value = _model_run_id_value(model_run_id)
    created_at_value = _created_at_value(created_at)

    price_forecast_df = build_price_forecast_mart(
        predictions_df,
        model_run_id=model_run_id_value,
        created_at=created_at_value,
    )
    top_expected_return_df = build_top_expected_return_mart(
        price_forecast_df,
        top_n=top_n,
    )
    backtest_daily_df = build_backtest_daily_mart(
        backtest_df,
        model_run_id=model_run_id_value,
        created_at=created_at_value,
    )
    metrics_df = build_metrics_mart(
        metrics,
        backtest_metrics,
        model_run_id=model_run_id_value,
        created_at=created_at_value,
    )
    daily_insights_df = build_daily_insights(
        price_forecast_df,
        backtest_metrics,
        model_run_id=model_run_id_value,
        created_at=created_at_value,
        top_n=top_n,
    )

    return {
        "price_forecast": price_forecast_df,
        "top_expected_return": top_expected_return_df,
        "backtest_daily": backtest_daily_df,
        "metrics": metrics_df,
        "daily_insights": daily_insights_df,
    }


def save_model1_marts(marts, output_paths):
    for mart_name, output_path in output_paths.items():
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        marts[mart_name].to_csv(path, index=False)


def _daily_insight_columns():
    return [
        "model_run_id",
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


def _empty_daily_insights():
    return pd.DataFrame(columns=_daily_insight_columns())