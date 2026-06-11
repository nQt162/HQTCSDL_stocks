import joblib
import pandas as pd
import numpy as np


def load_saved_model(model_path):
    saved = joblib.load(model_path)

    model = saved["model"]
    features = saved["features"]

    return model, features


def predict_latest_price(df, model, features):
    df = df.copy()

    df = df.replace(["NULL", "null", "None", ""], np.nan)

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values(["symbol", "trading_date"])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_df = df.groupby("symbol").tail(1).copy()

    latest_df = latest_df.dropna(subset=features)

    X_latest = latest_df[features]

    latest_df["predicted_close"] = model.predict(X_latest)
    latest_df["predicted_future_close"] = latest_df["predicted_close"]
    latest_df["predicted_return"] = latest_df["predicted_close"] / latest_df["close"] - 1

    latest_df["predicted_future_close_from_signal_close"] = latest_df[
        "predicted_close"
    ]

    return latest_df
