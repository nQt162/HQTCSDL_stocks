import importlib.util
import unittest

import pandas as pd

from src.mart import (
    build_daily_insights,
    build_metrics_mart,
    build_signal_summary_mart,
    build_trading_signals_mart,
)


class MartTests(unittest.TestCase):
    def _predictions(self):
        return pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "symbol": ["AAA", "BBB", "CCC"],
                "encode_sector": [1, 2, 1],
                "predicted_signal": ["BUY", "SELL", "BUY"],
                "adjusted_signal": ["BUY", "SELL", "HOLD"],
                "sell_probability": [0.05, 0.80, 0.20],
                "hold_probability": [0.10, 0.10, 0.45],
                "buy_probability": [0.85, 0.10, 0.35],
                "target_signal": ["BUY", "SELL", "HOLD"],
                "target_return": [0.04, -0.05, 0.01],
            }
        )

    def _sector_mapping(self):
        return pd.DataFrame(
            {
                "encode_sector": [1, 2],
                "sector": ["Banks", "Real Estate"],
            }
        )

    def test_build_trading_signals_mart_keeps_required_model3_columns(self):
        mart_df = build_trading_signals_mart(
            self._predictions(), sector_mapping_df=self._sector_mapping()
        )

        self.assertEqual("model3", mart_df.loc[0, "model_name"])
        self.assertIn("buy_probability", mart_df.columns)
        self.assertIn("sector", mart_df.columns)
        self.assertEqual("Banks", mart_df.loc[0, "sector"])
        self.assertIn("created_at", mart_df.columns)
        self.assertEqual(pd.Timestamp("2024-01-01").date(), mart_df.loc[0, "trading_date"])

    def test_build_signal_summary_mart_counts_adjusted_signals_by_day_and_sector(self):
        summary_df = build_signal_summary_mart(
            self._predictions(), sector_mapping_df=self._sector_mapping()
        )

        row = summary_df[
            (summary_df["trading_date"] == pd.Timestamp("2024-01-01").date())
            & (summary_df["sector"] == "Banks")
        ].iloc[0]
        self.assertEqual(1, row["encode_sector"])
        self.assertEqual(1, row["buy_count"])
        self.assertEqual(0, row["sell_count"])
        self.assertEqual(1, row["total_symbols"])

    def test_build_metrics_mart_flattens_classification_and_backtest_metrics(self):
        metrics_df = build_metrics_mart(
            {"Accuracy": 56.3, "Confusion_Matrix": [[1, 0], [0, 1]]},
            {"Cumulative_Return_Net": 2.2},
        )

        self.assertEqual(
            ["Accuracy", "Cumulative_Return_Net"],
            metrics_df["metric_name"].tolist(),
        )
        self.assertEqual(["classification", "backtest"], metrics_df["metric_group"].tolist())

    def test_build_daily_insights_creates_latest_signal_and_metric_messages(self):
        insights_df = build_daily_insights(
            self._predictions(),
            metrics={"Accuracy": 56.3},
            backtest_metrics={"Cumulative_Return_Net": 2.2},
        )

        self.assertEqual(pd.Timestamp("2024-01-02").date(), insights_df.loc[0, "insight_date"])
        self.assertIn("adjusted_buy_count", insights_df["metric_name"].tolist())
        self.assertIn("accuracy", insights_df["metric_name"].tolist())
        self.assertIn("cumulative_return_net", insights_df["metric_name"].tolist())

    def test_walk_forward_script_imports_without_missing_config_names(self):
        spec = importlib.util.spec_from_file_location("model3_walk_forward_script", "walk_forward.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
