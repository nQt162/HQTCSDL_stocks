import joblib
from xgboost import XGBRegressor


def train_xgboost_model(
    X_train,
    y_train,
    params,
    X_val=None,
    y_val=None,
    early_stopping_rounds=None,
    verbose=False,
):
    model_params = params.copy()
    fit_kwargs = {"verbose": verbose}

    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided together")

        fit_kwargs["eval_set"] = [(X_val, y_val)]

        if early_stopping_rounds is not None:
            model_params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def save_model(model, features, horizon, model_path):
    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon": horizon,
            "target_type": "future_close_price"
        },
        model_path
    )
