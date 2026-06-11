import unittest

import pandas as pd

from src.marts import build_model1_marts


class Model1MartsTests(unittest.TestCase):
    def test_build_model1_marts_returns_dashboard_ready_tables(self):
        predictions_df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "future_trading_date": ["2024-01-08", "2024-01-08", "2024-01-09"],
                "symbol": ["AAA", "BBB", "CCC"],
                "close": [100.0, 50.0, 20.0],
                "target_close": [110.0, 45.0, 22.0],
                "target_return": [0.10, -0.10, 0.10],
                "predicted_close": [112.0, 55.0, 21.0],
                "predicted_return": [0.12, 0.10, 0.05],
                "actual_direction": [1, 0, 1],
                "predicted_direction": [1, 1, 1],
            }
        )
        backtest_df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01"],
                "daily_return": [0.03],
                "benchmark_return": [0.01],
                "cumulative_return": [0.08],
                "benchmark_cumulative_return": [0.02],
            }
        )

        marts = build_model1_marts(
            predictions_df=predictions_df,
            backtest_df=backtest_df,
            metrics={"MAE": 1.2},
            backtest_metrics={"Cumulative_Return": 0.08, "Benchmark_Cumulative_Return": 0.02},
            model_run_id="run-1",
            created_at=pd.Timestamp("2024-02-01 10:00:00"),
            top_n=1,
        )

        price_forecast = marts["price_forecast"]
        self.assertEqual(
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
            ],
            price_forecast.columns.tolist(),
        )
        self.assertEqual("run-1", price_forecast.loc[0, "model_run_id"])
        self.assertEqual(pd.Timestamp("2024-01-01"), price_forecast.loc[0, "prediction_date"])
        self.assertEqual(pd.Timestamp("2024-01-08"), price_forecast.loc[0, "target_date"])
        self.assertAlmostEqual(110.0, price_forecast.loc[0, "real_close"])
        self.assertEqual([1, 0, 1], price_forecast["direction_correct"].tolist())

        top_expected_return = marts["top_expected_return"]
        self.assertEqual(["AAA", "CCC"], top_expected_return["symbol"].tolist())
        self.assertEqual([1, 1], top_expected_return["rank"].tolist())

        metrics_mart = marts["metrics"]
        metric_keys = set(zip(metrics_mart["metric_group"], metrics_mart["metric_name"]))
        self.assertIn(("test", "MAE"), metric_keys)
        self.assertIn(("backtest", "Cumulative_Return"), metric_keys)

        insights = marts["daily_insights"]
        self.assertEqual(
            [
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
            ],
            insights.columns.tolist(),
        )
        self.assertIn("CCC", insights["symbol"].dropna().tolist())
        self.assertIn("model1_backtest_outperformance", insights["metric_name"].tolist())


if __name__ == "__main__":
    unittest.main()
