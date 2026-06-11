import os
import json

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    PREDICTION_PATH,
    PREDICTION_ACCURACY_PATH,
    FEATURE_IMPORTANCE_PATH,
    BACKTEST_PATH,
    BACKTEST_METRICS_PATH,
    BACKTEST_SWEEP_PATH,
    MART_MODEL1_BACKTEST_DAILY_PATH,
    MART_MODEL1_METRICS_PATH,
    MART_MODEL1_PRICE_FORECAST_PATH,
    MART_MODEL1_TOP_EXPECTED_RETURN_PATH,
    MODEL1_DAILY_INSIGHTS_PATH,
    HORIZON,
    FEATURES,
    XGB_PARAMS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    EARLY_STOPPING_ROUNDS,
    BACKTEST_TOP_K,
    BACKTEST_MIN_VOLUME,
    BACKTEST_MIN_CLOSE,
    BACKTEST_MIN_PREDICTED_RETURN,
    BACKTEST_TOP_K_VALUES,
    BACKTEST_MIN_VOLUME_VALUES,
    BACKTEST_MIN_CLOSE_VALUES,
    BACKTEST_MIN_PREDICTED_RETURN_VALUES,
    TRANSACTION_COST_RATE,
    SLIPPAGE_RATE,
    MAX_ABS_TARGET_RETURN
)

from src.data_loader import load_data
from src.marts import build_model1_marts, save_model1_marts
from src.preprocessing import preprocess_data, split_train_validation_test_by_time
from src.train_model import train_xgboost_model, save_model
from src.evaluate import (
    build_prediction_accuracy_table,
    evaluate_model,
    save_metrics,
    save_feature_importance,
)
from src.backtest import compute_top_k_backtest, run_backtest_sweep, save_backtest_metrics


def create_folders():
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    os.makedirs(METRICS_PATH.parent, exist_ok=True)


def main():
    create_folders()

    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    df, final_features = preprocess_data(
        df=df,
        features=FEATURES,
        horizon=HORIZON,
        max_abs_target_return=MAX_ABS_TARGET_RETURN
    )

    print("Splitting train/validation/test...")
    train_df, validation_df, test_df, validation_start_date, test_start_date = (
        split_train_validation_test_by_time(
            df,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO
        )
    )

    X_train = train_df[final_features]
    y_train = train_df["target_close"]

    X_val = validation_df[final_features]
    y_val = validation_df["target_close"]

    X_test = test_df[final_features]

    print("Training XGBoost Regressor...")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        params=XGB_PARAMS,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    print("Evaluating model...")
    metrics, result_df = evaluate_model(
        model=model,
        X_test=X_test,
        test_df=test_df
    )

    print("Running top-k backtest...")
    backtest_df, backtest_metrics = compute_top_k_backtest(
        result_df=result_df,
        top_k=BACKTEST_TOP_K,
        min_volume=BACKTEST_MIN_VOLUME,
        min_close=BACKTEST_MIN_CLOSE,
        min_predicted_return=BACKTEST_MIN_PREDICTED_RETURN,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        slippage_rate=SLIPPAGE_RATE
    )

    print("Running backtest parameter sweep...")
    backtest_sweep_df = run_backtest_sweep(
        result_df=result_df,
        top_k_values=BACKTEST_TOP_K_VALUES,
        min_volume_values=BACKTEST_MIN_VOLUME_VALUES,
        min_close_values=BACKTEST_MIN_CLOSE_VALUES,
        min_predicted_return_values=BACKTEST_MIN_PREDICTED_RETURN_VALUES,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        slippage_rate=SLIPPAGE_RATE
    )

    print("Saving model and reports...")
    save_model(
        model=model,
        features=final_features,
        horizon=HORIZON,
        model_path=MODEL_PATH
    )

    save_metrics(metrics, METRICS_PATH)

    result_df.to_csv(PREDICTION_PATH, index=False)
    prediction_accuracy_df = build_prediction_accuracy_table(result_df)
    prediction_accuracy_df.to_csv(PREDICTION_ACCURACY_PATH, index=False)
    backtest_df.to_csv(BACKTEST_PATH, index=False)
    backtest_sweep_df.to_csv(BACKTEST_SWEEP_PATH, index=False)
    save_backtest_metrics(backtest_metrics, BACKTEST_METRICS_PATH)

    save_feature_importance(
        model=model,
        features=final_features,
        path=FEATURE_IMPORTANCE_PATH
    )

    model1_marts = build_model1_marts(
        predictions_df=result_df,
        backtest_df=backtest_df,
        metrics=metrics,
        backtest_metrics=backtest_metrics,
    )
    save_model1_marts(
        model1_marts,
        {
            "price_forecast": MART_MODEL1_PRICE_FORECAST_PATH,
            "top_expected_return": MART_MODEL1_TOP_EXPECTED_RETURN_PATH,
            "backtest_daily": MART_MODEL1_BACKTEST_DAILY_PATH,
            "metrics": MART_MODEL1_METRICS_PATH,
            "daily_insights": MODEL1_DAILY_INSIGHTS_PATH,
        },
    )
    print("Done.")
    print("Validation start date:", validation_start_date)
    print("Test start date:", test_start_date)
    print("Features used:", final_features)
    print(json.dumps(metrics, indent=4))
    print("Backtest metrics:")
    print(json.dumps(backtest_metrics, indent=4))


if __name__ == "__main__":
    main()
