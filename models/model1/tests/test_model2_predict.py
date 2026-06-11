import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODEL2_DIR = Path(__file__).resolve().parents[1] / "model2"


def clear_model2_modules():
    for module_name in [
        "config",
        "feature_engineering",
        "load_data",
        "predict",
        "model2_predict_under_test",
    ]:
        sys.modules.pop(module_name, None)


def load_model2_predict_module(fake_clickhouse=True):
    if str(MODEL2_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL2_DIR))

    clear_model2_modules()

    if fake_clickhouse:
        sys.modules.setdefault(
            "clickhouse_connect",
            types.SimpleNamespace(get_client=lambda **kwargs: None),
        )

    spec = importlib.util.spec_from_file_location(
        "model2_predict_under_test",
        MODEL2_DIR / "predict.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLoader:
    def load_data(self):
        return pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "symbol": ["AAA", "AAA", "AAA"],
                "close": [10.0, 11.0, 12.0],
                "feature": [1.0, 2.0, 3.0],
            }
        )


class FakeEngineer:
    def run(self, df):
        return df


class FakeModel:
    def __init__(self):
        self.seen_X = None

    def predict(self, X):
        self.seen_X = X.copy()
        return np.array([0.25])


class Model2PredictTests(unittest.TestCase):
    def test_predict_module_imports_without_clickhouse_connect_installed(self):
        with patch.dict(sys.modules, {"clickhouse_connect": None}):
            predict_module = load_model2_predict_module(fake_clickhouse=False)

        self.assertTrue(hasattr(predict_module, "StockPredictor"))

    def test_predict_latest_by_symbol_uses_latest_feature_row_without_future_target(self):
        predict_module = load_model2_predict_module()
        model = FakeModel()

        predictor = predict_module.StockPredictor(
            model=model,
            loader=FakeLoader(),
            engineer=FakeEngineer(),
        )

        with patch.object(predict_module, "FEATURE_COLUMNS", ["feature"]):
            result = predictor.predict_latest_by_symbol("AAA")

        self.assertEqual("AAA", result["symbol"])
        self.assertEqual(pd.Timestamp("2024-01-03"), result["latest_date"])
        self.assertAlmostEqual(12.0, result["latest_close"])
        self.assertAlmostEqual(0.25, result["predicted_return_5d"])
        self.assertAlmostEqual(15.0, result["predicted_close_5d"])
        self.assertEqual([3.0], model.seen_X["feature"].tolist())


if __name__ == "__main__":
    unittest.main()
