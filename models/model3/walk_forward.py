import json
import os

from src.backtest import save_backtest_metrics
from src.config import (
    BACKTEST_MIN_CLOSE,
    BACKTEST_MIN_BUY_SELL_MARGIN,
    BACKTEST_MIN_BUY_PROBABILITY,
    BACKTEST_MIN_VOLUME,
    BACKTEST_TOP_K,
    EARLY_STOPPING_ROUNDS,
    FEATURES,
    HORIZON,
    BUY_RETURN_THRESHOLD,
    MAX_ABS_TARGET_RETURN,
    SELL_RETURN_THRESHOLD,
    SLIPPAGE_RATE,
    TRANSACTION_COST_RATE,
    WALK_FORWARD_BACKTEST_METRICS_PATH,
    WALK_FORWARD_BACKTEST_PATH,
    WALK_FORWARD_FOLD_METRICS_PATH,
    WALK_FORWARD_INITIAL_TRAIN_RATIO,
    WALK_FORWARD_MAX_FOLDS,
    WALK_FORWARD_PREDICTION_PATH,
    WALK_FORWARD_STEP_RATIO,
    WALK_FORWARD_TEST_RATIO,
    WALK_FORWARD_VALIDATION_RATIO,
    XGB_PARAMS,
)
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.walk_forward import run_walk_forward_backtest


def create_folders():
    os.makedirs("reports", exist_ok=True)


def main():
    create_folders()

    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    df, final_features = preprocess_data(
        df=df,
        features=FEATURES,
        horizon=HORIZON,
        max_abs_target_return=MAX_ABS_TARGET_RETURN,
        sell_return_threshold=SELL_RETURN_THRESHOLD,
        buy_return_threshold=BUY_RETURN_THRESHOLD,
    )

    print("Running walk-forward backtest...")
    predictions_df, fold_metrics_df, backtest_df, backtest_metrics = (
        run_walk_forward_backtest(
            df=df,
            features=final_features,
            params=XGB_PARAMS,
            initial_train_ratio=WALK_FORWARD_INITIAL_TRAIN_RATIO,
            validation_ratio=WALK_FORWARD_VALIDATION_RATIO,
            test_ratio=WALK_FORWARD_TEST_RATIO,
            step_ratio=WALK_FORWARD_STEP_RATIO,
            max_folds=WALK_FORWARD_MAX_FOLDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            backtest_kwargs={
                "top_k": BACKTEST_TOP_K,
                "min_volume": BACKTEST_MIN_VOLUME,
                "min_close": BACKTEST_MIN_CLOSE,
                "min_buy_probability": BACKTEST_MIN_BUY_PROBABILITY,
                "min_buy_sell_margin": BACKTEST_MIN_BUY_SELL_MARGIN,
                "transaction_cost_rate": TRANSACTION_COST_RATE,
                "slippage_rate": SLIPPAGE_RATE,
            },
        )
    )

    print("Saving walk-forward reports...")
    predictions_df.to_csv(WALK_FORWARD_PREDICTION_PATH, index=False)
    fold_metrics_df.to_csv(WALK_FORWARD_FOLD_METRICS_PATH, index=False)
    backtest_df.to_csv(WALK_FORWARD_BACKTEST_PATH, index=False)
    save_backtest_metrics(backtest_metrics, WALK_FORWARD_BACKTEST_METRICS_PATH)

    print("Done.")
    print(fold_metrics_df[["fold_id", "test_start_date", "test_end_date"]])
    print(json.dumps(backtest_metrics, indent=4))


if __name__ == "__main__":
    main()
