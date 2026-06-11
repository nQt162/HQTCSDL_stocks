import unittest

import numpy as np
import pandas as pd

from src.evaluate import build_prediction_accuracy_table, evaluate_model


class DummyModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions


class EvaluateModelTests(unittest.TestCase):
    def test_evaluate_model_reports_close_price_and_derived_return_metrics(self):
        test_df = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "future_close": [110.0, 90.0],
                "target_close": [110.0, 90.0],
                "target_return": [0.10, -0.10],
            }
        )
        X_test = pd.DataFrame({"feature": [1.0, 2.0]})
        model = DummyModel([108.0, 95.0])

        metrics, result_df = evaluate_model(model=model, X_test=X_test, test_df=test_df)

        price_errors = np.array([-2.0, 5.0])
        return_errors = np.array([-0.02, 0.05])
        self.assertAlmostEqual(metrics["MAE"], np.abs(price_errors).mean())
        self.assertAlmostEqual(metrics["Return_MAE"], np.abs(return_errors).mean())
        self.assertAlmostEqual(
            metrics["Return_RMSE"], np.sqrt(np.mean(return_errors**2))
        )
        self.assertAlmostEqual(metrics["Baseline_Return_MAE"], 0.10)
        self.assertAlmostEqual(metrics["Baseline_Return_RMSE"], 0.10)
        self.assertIn("predicted_close", result_df.columns)
        self.assertAlmostEqual(108.0, result_df.loc[0, "predicted_close"])
        self.assertAlmostEqual(95.0, result_df.loc[1, "predicted_close"])
        self.assertIn("predicted_return", result_df.columns)
        self.assertAlmostEqual(0.08, result_df.loc[0, "predicted_return"])
        self.assertAlmostEqual(-0.05, result_df.loc[1, "predicted_return"])

    def test_build_prediction_accuracy_table_reports_close_error_percent(self):
        result_df = pd.DataFrame(
            {
                "future_trading_date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
                "symbol": ["AAA", "BBB"],
                "target_close": [100.0, 80.0],
                "predicted_close": [95.0, 88.0],
            }
        )

        accuracy_df = build_prediction_accuracy_table(result_df)

        self.assertEqual(
            ["date", "symbol", "real_close", "predict_close", "accuracy_pct", "error_pct"],
            accuracy_df.columns.tolist(),
        )
        self.assertEqual(pd.Timestamp("2024-01-10"), accuracy_df.loc[0, "date"])
        self.assertEqual("AAA", accuracy_df.loc[0, "symbol"])
        self.assertAlmostEqual(100.0, accuracy_df.loc[0, "real_close"])
        self.assertAlmostEqual(95.0, accuracy_df.loc[0, "predict_close"])
        self.assertAlmostEqual(95.0, accuracy_df.loc[0, "accuracy_pct"])
        self.assertAlmostEqual(5.0, accuracy_df.loc[0, "error_pct"])
        self.assertAlmostEqual(90.0, accuracy_df.loc[1, "accuracy_pct"])
        self.assertAlmostEqual(10.0, accuracy_df.loc[1, "error_pct"])


if __name__ == "__main__":
    unittest.main()
