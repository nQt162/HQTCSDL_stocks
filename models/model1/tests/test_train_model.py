import unittest
from unittest.mock import patch

import pandas as pd

from src.train_model import train_xgboost_model


class TrainModelTests(unittest.TestCase):
    def test_train_xgboost_model_uses_validation_set_for_early_stopping(self):
        X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        y_train = pd.Series([101.0, 102.0, 103.0])
        X_val = pd.DataFrame({"feature": [4.0, 5.0]})
        y_val = pd.Series([104.0, 105.0])

        with patch("src.train_model.XGBRegressor") as regressor_cls:
            model = regressor_cls.return_value

            result = train_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                params={"n_estimators": 100},
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=10,
                verbose=False,
            )

        self.assertIs(result, model)
        self.assertEqual(10, regressor_cls.call_args.kwargs["early_stopping_rounds"])
        fit_kwargs = model.fit.call_args.kwargs
        self.assertIs(X_val, fit_kwargs["eval_set"][0][0])
        self.assertIs(y_val, fit_kwargs["eval_set"][0][1])
        self.assertFalse(fit_kwargs["verbose"])


if __name__ == "__main__":
    unittest.main()
