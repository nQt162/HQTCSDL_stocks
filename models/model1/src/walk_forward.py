import pandas as pd

from src.backtest import compute_top_k_backtest
from src.evaluate import evaluate_model
from src.train_model import train_xgboost_model


def _ratio_to_size(date_count, ratio, name):
    if ratio <= 0:
        raise ValueError(f"{name} must be positive")

    size = int(date_count * ratio)
    if size <= 0:
        raise ValueError(f"{name} creates an empty date window")

    return size


def create_walk_forward_folds(
    df,
    initial_train_ratio,
    validation_ratio,
    test_ratio,
    step_ratio,
    max_folds=None,
):
    dates = pd.Series(pd.to_datetime(df["trading_date"].unique())).sort_values()
    dates = dates.reset_index(drop=True)
    date_count = len(dates)

    train_size = _ratio_to_size(date_count, initial_train_ratio, "initial_train_ratio")
    validation_size = _ratio_to_size(date_count, validation_ratio, "validation_ratio")
    test_size = _ratio_to_size(date_count, test_ratio, "test_ratio")
    step_size = _ratio_to_size(date_count, step_ratio, "step_ratio")

    folds = []
    train_end_index = train_size

    while True:
        validation_start_index = train_end_index
        validation_end_index = validation_start_index + validation_size
        test_start_index = validation_end_index
        test_end_index = test_start_index + test_size

        if test_end_index > date_count:
            break

        train_dates = set(dates.iloc[:train_end_index])
        validation_dates = set(dates.iloc[validation_start_index:validation_end_index])
        test_dates = set(dates.iloc[test_start_index:test_end_index])

        train_df = df[df["trading_date"].isin(train_dates)].copy()
        validation_df = df[df["trading_date"].isin(validation_dates)].copy()
        test_df = df[df["trading_date"].isin(test_dates)].copy()

        if train_df.empty or validation_df.empty or test_df.empty:
            raise ValueError("Walk-forward split produced an empty fold")

        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_df": train_df,
                "validation_df": validation_df,
                "test_df": test_df,
                "train_start_date": train_df["trading_date"].min(),
                "train_end_date": train_df["trading_date"].max(),
                "validation_start_date": validation_df["trading_date"].min(),
                "validation_end_date": validation_df["trading_date"].max(),
                "test_start_date": test_df["trading_date"].min(),
                "test_end_date": test_df["trading_date"].max(),
            }
        )

        if max_folds is not None and len(folds) >= max_folds:
            break

        train_end_index += step_size

    if not folds:
        raise ValueError("No walk-forward folds could be created")

    return folds


def run_walk_forward_backtest(
    df,
    features,
    params,
    initial_train_ratio,
    validation_ratio,
    test_ratio,
    step_ratio,
    early_stopping_rounds,
    backtest_kwargs,
    max_folds=None,
    train_model_fn=train_xgboost_model,
):
    folds = create_walk_forward_folds(
        df=df,
        initial_train_ratio=initial_train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        step_ratio=step_ratio,
        max_folds=max_folds,
    )

    predictions = []
    fold_metrics = []

    for fold in folds:
        train_df = fold["train_df"]
        validation_df = fold["validation_df"]
        test_df = fold["test_df"]

        model = train_model_fn(
            X_train=train_df[features],
            y_train=train_df["target_close"],
            params=params,
            X_val=validation_df[features],
            y_val=validation_df["target_close"],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        metrics, fold_prediction_df = evaluate_model(
            model=model,
            X_test=test_df[features],
            test_df=test_df,
        )
        fold_prediction_df = fold_prediction_df.copy()
        fold_prediction_df.insert(0, "fold_id", fold["fold_id"])
        predictions.append(fold_prediction_df)

        fold_metric_row = {
            "fold_id": fold["fold_id"],
            "train_start_date": fold["train_start_date"],
            "train_end_date": fold["train_end_date"],
            "validation_start_date": fold["validation_start_date"],
            "validation_end_date": fold["validation_end_date"],
            "test_start_date": fold["test_start_date"],
            "test_end_date": fold["test_end_date"],
            "best_iteration": getattr(model, "best_iteration", None),
        }
        fold_metric_row.update(metrics)
        fold_metrics.append(fold_metric_row)

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df = predictions_df.sort_values(["trading_date", "symbol"])
    fold_metrics_df = pd.DataFrame(fold_metrics)

    backtest_df, backtest_metrics = compute_top_k_backtest(
        predictions_df,
        **backtest_kwargs,
    )
    backtest_metrics["Walk_Forward_Folds"] = len(folds)

    return predictions_df, fold_metrics_df, backtest_df, backtest_metrics
