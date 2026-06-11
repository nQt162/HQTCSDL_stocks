import importlib
import unittest

import pandas as pd


model1_app = importlib.import_module("test")


class Model1StreamlitAppTests(unittest.TestCase):
    def test_filter_by_prediction_date_returns_future_close_results_and_summary(self):
        forecasts_df = pd.DataFrame(
            {
                "model_run_id": ["run-1", "run-1", "run-1"],
                "prediction_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "target_date": ["2024-01-08", "2024-01-08", "2024-01-09"],
                "symbol": ["BBB", "AAA", "CCC"],
                "real_close": [55.0, 110.0, 21.0],
                "predicted_close": [58.0, 112.0, 22.0],
                "created_at": ["2024-02-01 10:00:00"] * 3,
            }
        )

        prepared_df = model1_app.prepare_forecast_data(forecasts_df)
        filtered_df = model1_app.filter_forecasts_by_date(
            prepared_df,
            pd.Timestamp("2024-01-01").date(),
        )
        summary = model1_app.build_daily_summary(filtered_df)
        display_df = model1_app.build_display_table(filtered_df)

        self.assertEqual(["AAA", "BBB"], filtered_df["symbol"].tolist())
        self.assertEqual(2, summary["num_symbols"])
        self.assertEqual("2024-01-08", summary["target_date"])
        self.assertAlmostEqual(85.0, summary["avg_predicted_close"])
        self.assertAlmostEqual(85.0, summary["median_predicted_close"])
        self.assertEqual(
            [
                "symbol",
                "prediction_date",
                "target_date",
                "real_close",
                "predicted_close",
                "price_error",
                "error_pct",
            ],
            display_df.columns.tolist(),
        )
        self.assertAlmostEqual(2.0, display_df.loc[0, "price_error"])

    def test_available_prediction_dates_are_sorted_unique_dates(self):
        forecasts_df = pd.DataFrame(
            {
                "prediction_date": ["2024-01-02", "2024-01-01", "2024-01-02"],
                "target_date": ["2024-01-09", "2024-01-08", "2024-01-09"],
                "symbol": ["CCC", "AAA", "BBB"],
                "real_close": [21.0, 110.0, 55.0],
                "predicted_close": [22.0, 112.0, 58.0],
            }
        )

        prepared_df = model1_app.prepare_forecast_data(forecasts_df)

        self.assertEqual(
            [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
            model1_app.available_prediction_dates(prepared_df),
        )


if __name__ == "__main__":
    unittest.main()