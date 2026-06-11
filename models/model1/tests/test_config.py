import unittest

import pandas as pd

from src.config import DATA_PATH, FEATURES


class ConfigFeatureTests(unittest.TestCase):
    def test_configured_features_exist_in_raw_dataset(self):
        if not DATA_PATH.exists():
            self.skipTest(f"Dataset is not available: {DATA_PATH}")

        columns = set(pd.read_csv(DATA_PATH, nrows=0).columns.str.strip())

        missing_features = [feature for feature in FEATURES if feature not in columns]

        self.assertEqual([], missing_features)

    def test_backtest_config_is_available(self):
        from src import config

        expected_report_dir = config.PROJECT_ROOT / "models" / "model1" / "reports"

        self.assertEqual(5, config.HORIZON)
        self.assertEqual(expected_report_dir / "backtest.csv", config.BACKTEST_PATH)
        self.assertEqual(
            expected_report_dir / "backtest_metrics.json",
            config.BACKTEST_METRICS_PATH,
        )
        self.assertEqual(
            expected_report_dir / "backtest_sweep.csv",
            config.BACKTEST_SWEEP_PATH,
        )
        self.assertGreater(config.BACKTEST_TOP_K, 0)
        self.assertGreater(config.BACKTEST_MIN_VOLUME, 0)
        self.assertGreater(config.BACKTEST_MIN_CLOSE, 0)
        self.assertGreaterEqual(config.BACKTEST_MIN_PREDICTED_RETURN, 0)
        self.assertGreaterEqual(config.TRANSACTION_COST_RATE, 0)
        self.assertGreaterEqual(config.SLIPPAGE_RATE, 0)
        self.assertIn(config.BACKTEST_TOP_K, config.BACKTEST_TOP_K_VALUES)
        self.assertIn(config.BACKTEST_MIN_VOLUME, config.BACKTEST_MIN_VOLUME_VALUES)
        self.assertIn(config.BACKTEST_MIN_CLOSE, config.BACKTEST_MIN_CLOSE_VALUES)
        self.assertIn(
            config.BACKTEST_MIN_PREDICTED_RETURN,
            config.BACKTEST_MIN_PREDICTED_RETURN_VALUES,
        )

    def test_walk_forward_config_is_available(self):
        from src import config

        expected_report_dir = config.PROJECT_ROOT / "models" / "model1" / "reports"

        self.assertEqual(
            expected_report_dir / "walk_forward_predictions.csv",
            config.WALK_FORWARD_PREDICTION_PATH,
        )
        self.assertEqual(
            expected_report_dir / "walk_forward_fold_metrics.csv",
            config.WALK_FORWARD_FOLD_METRICS_PATH,
        )
        self.assertEqual(
            expected_report_dir / "walk_forward_backtest.csv",
            config.WALK_FORWARD_BACKTEST_PATH,
        )
        self.assertEqual(
            expected_report_dir / "walk_forward_backtest_metrics.json",
            config.WALK_FORWARD_BACKTEST_METRICS_PATH,
        )
        self.assertGreater(config.WALK_FORWARD_INITIAL_TRAIN_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_VALIDATION_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_TEST_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_STEP_RATIO, 0)



    def test_model1_mart_config_is_available(self):
        from src import config

        expected_report_dir = config.PROJECT_ROOT / "models" / "model1" / "reports"

        self.assertEqual(
            expected_report_dir / "mart_model1_price_forecast.csv",
            config.MART_MODEL1_PRICE_FORECAST_PATH,
        )
        self.assertEqual(
            expected_report_dir / "mart_model1_top_expected_return.csv",
            config.MART_MODEL1_TOP_EXPECTED_RETURN_PATH,
        )
        self.assertEqual(
            expected_report_dir / "mart_model1_backtest_daily.csv",
            config.MART_MODEL1_BACKTEST_DAILY_PATH,
        )
        self.assertEqual(
            expected_report_dir / "mart_model1_metrics.csv",
            config.MART_MODEL1_METRICS_PATH,
        )
        self.assertEqual(
            expected_report_dir / "model1_daily_insights.csv",
            config.MODEL1_DAILY_INSIGHTS_PATH,
        )

if __name__ == "__main__":
    unittest.main()
