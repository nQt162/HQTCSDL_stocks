from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

DATA_PATH = PROJECT_ROOT / "data" / "clean" / "features_all.csv"
MODEL1_ROOT = PROJECT_ROOT / "models" / "model1"
MODEL_DIR = MODEL1_ROOT / "models"
REPORT_DIR = MODEL1_ROOT / "reports"

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT") or "8443")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")
CLICKHOUSE_SOURCE_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
CLICKHOUSE_TABLE = os.getenv("MODEL1_CLICKHOUSE_FEATURES_TABLE", os.getenv("CLICKHOUSE_TABLE", "features_all"))
CLICKHOUSE_SECURE = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}

MODEL_PATH = MODEL_DIR / "price_forecasting_xgb.pkl"

METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTION_PATH = REPORT_DIR / "predictions.csv"
PREDICTION_ACCURACY_PATH = REPORT_DIR / "prediction_accuracy.csv"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
BACKTEST_PATH = REPORT_DIR / "backtest.csv"
BACKTEST_METRICS_PATH = REPORT_DIR / "backtest_metrics.json"
BACKTEST_SWEEP_PATH = REPORT_DIR / "backtest_sweep.csv"
WALK_FORWARD_PREDICTION_PATH = REPORT_DIR / "walk_forward_predictions.csv"
WALK_FORWARD_FOLD_METRICS_PATH = REPORT_DIR / "walk_forward_fold_metrics.csv"
WALK_FORWARD_BACKTEST_PATH = REPORT_DIR / "walk_forward_backtest.csv"
WALK_FORWARD_BACKTEST_METRICS_PATH = REPORT_DIR / "walk_forward_backtest_metrics.json"
MART_MODEL1_PRICE_FORECAST_PATH = REPORT_DIR / "mart_model1_price_forecast.csv"
MART_MODEL1_TOP_EXPECTED_RETURN_PATH = REPORT_DIR / "mart_model1_top_expected_return.csv"
MART_MODEL1_BACKTEST_DAILY_PATH = REPORT_DIR / "mart_model1_backtest_daily.csv"
MART_MODEL1_METRICS_PATH = REPORT_DIR / "mart_model1_metrics.csv"
MODEL1_DAILY_INSIGHTS_PATH = REPORT_DIR / "model1_daily_insights.csv"

HORIZON = 5

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
EARLY_STOPPING_ROUNDS = 50
BACKTEST_TOP_K = 10
BACKTEST_MIN_VOLUME = 100000
BACKTEST_MIN_CLOSE = 5.0
BACKTEST_MIN_PREDICTED_RETURN = 0.004
BACKTEST_TOP_K_VALUES = [5, 10, 20]
BACKTEST_MIN_VOLUME_VALUES = [50000, 100000, 200000]
BACKTEST_MIN_CLOSE_VALUES = [5.0, 10.0]
BACKTEST_MIN_PREDICTED_RETURN_VALUES = [0.0, 0.004, 0.01, 0.02]
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.001
MAX_ABS_TARGET_RETURN = 0.2
WALK_FORWARD_INITIAL_TRAIN_RATIO = 0.5
WALK_FORWARD_VALIDATION_RATIO = 0.1
WALK_FORWARD_TEST_RATIO = 0.1
WALK_FORWARD_STEP_RATIO = 0.1
WALK_FORWARD_MAX_FOLDS = 4

FEATURES = [
    "encode_sector",

    "open", "high", "low", "close", "volume",

    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",

    "ma_5", "ma_20", "ma_50", "price_vs_ma20", "ma5_vs_ma20",

    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",

    "volatility_5d", "volatility_20d", "volatility_change",

    "rolling_max_20d", "drawdown_20d",

    "daily_range", "body_ratio", "close_position"
]

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
}
