import unittest

import numpy as np
import pandas as pd

from src.walk_forward import create_walk_forward_folds, run_walk_forward_backtest


class ConstantCloseModel:
    def __init__(self, predicted_close):
        self.predicted_close = predicted_close
        self.best_iteration = 3

    def predict(self, X):
        return np.full(len(X), self.predicted_close)


class WalkForwardTests(unittest.TestCase):
    def test_create_walk_forward_folds_expands_train_window_chronologically(self):
        df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "symbol": ["AAA"] * 10,
                "close": range(10),
            }
        )

        folds = create_walk_forward_folds(
            df=df,
            initial_train_ratio=0.4,
            validation_ratio=0.2,
            test_ratio=0.2,
            step_ratio=0.2,
        )

        self.assertEqual(2, len(folds))
        self.assertEqual(4, len(folds[0]["train_df"]))
        self.assertEqual(2, len(folds[0]["validation_df"]))
        self.assertEqual(2, len(folds[0]["test_df"]))
        self.assertEqual(6, len(folds[1]["train_df"]))
        self.assertLess(
            folds[0]["train_df"]["trading_date"].max(),
            folds[0]["validation_df"]["trading_date"].min(),
        )
        self.assertLess(
            folds[0]["validation_df"]["trading_date"].max(),
            folds[0]["test_df"]["trading_date"].min(),
        )
        self.assertLess(
            folds[0]["test_df"]["trading_date"].max(),
            folds[1]["test_df"]["trading_date"].min(),
        )

    def test_run_walk_forward_backtest_combines_predictions_metrics_and_backtest(self):
        df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "symbol": ["AAA"] * 10,
                "feature": np.arange(10),
                "close": [100.0] * 10,
                "future_close": [101.0] * 10,
                "target_close": [101.0] * 10,
                "target_return": [0.01] * 10,
            }
        )
        train_calls = []

        def train_model_fn(
            X_train,
            y_train,
            params,
            X_val,
            y_val,
            early_stopping_rounds,
            verbose,
        ):
            train_calls.append((X_train, y_train, X_val, y_val))
            return ConstantCloseModel(101.0)

        predictions_df, fold_metrics_df, backtest_df, backtest_metrics = (
            run_walk_forward_backtest(
                df=df,
                features=["feature"],
                params={"n_estimators": 10},
                initial_train_ratio=0.4,
                validation_ratio=0.2,
                test_ratio=0.2,
                step_ratio=0.2,
                early_stopping_rounds=5,
                backtest_kwargs={"top_k": 1},
                train_model_fn=train_model_fn,
            )
        )

        self.assertEqual(2, len(train_calls))
        self.assertTrue((train_calls[0][1] == 101.0).all())
        self.assertEqual(4, len(predictions_df))
        self.assertEqual([1, 2], fold_metrics_df["fold_id"].tolist())
        self.assertIn("MAE", fold_metrics_df.columns)
        self.assertIn("predicted_close", predictions_df.columns)
        self.assertEqual(4, len(backtest_df))
        self.assertEqual(2, backtest_metrics["Walk_Forward_Folds"])
        self.assertEqual(4, backtest_metrics["Backtest_Days"])


if __name__ == "__main__":
    unittest.main()
