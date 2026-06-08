
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


CLICKHOUSE_HOST = env_text(
    "CLICKHOUSE_HOST",
    "cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud",
)
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8443"))
CLICKHOUSE_USERNAME = os.getenv("CLICKHOUSE_USERNAME", os.getenv("CLICKHOUSE_USER", "default"))
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "stock")
CLICKHOUSE_FEATURES_TABLE = os.getenv("MODEL3_CLICKHOUSE_FEATURES_TABLE", "features_all")
MODEL3_CLICKHOUSE_START_DATE = os.getenv("MODEL3_CLICKHOUSE_START_DATE", "2021-01-01").strip()
MODEL3_CLICKHOUSE_END_DATE = os.getenv("MODEL3_CLICKHOUSE_END_DATE", "").strip()
MODEL3_CLICKHOUSE_LIMIT = int(os.getenv("MODEL3_CLICKHOUSE_LIMIT", "0") or "0")
CLICKHOUSE_SECURE = os.getenv("CLICKHOUSE_SECURE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
SECTOR_LABEL_ENCODING_PATH = os.getenv(
    "SECTOR_LABEL_ENCODING_PATH",
    "github_push/HQTCSDL_stocks/data/clean/sector_label_encoding.csv",
)

MODEL_PATH = "models/trading_signal_xgb_classifier.pkl"

METRICS_PATH = "reports/metrics.json"
PREDICTION_PATH = "reports/predictions.csv"
PREDICTION_ACCURACY_PATH = "reports/prediction_accuracy.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"
BACKTEST_PATH = "reports/backtest.csv"
BACKTEST_METRICS_PATH = "reports/backtest_metrics.json"
BACKTEST_SWEEP_PATH = "reports/backtest_sweep.csv"
MART_MODEL3_TRADING_SIGNALS_PATH = "reports/mart_model3_trading_signals.csv"
MART_MODEL3_SIGNAL_SUMMARY_PATH = "reports/mart_model3_signal_summary.csv"
MART_MODEL3_BACKTEST_DAILY_PATH = "reports/mart_model3_backtest_daily.csv"
MART_MODEL3_METRICS_PATH = "reports/mart_model3_metrics.csv"
MODEL3_DAILY_INSIGHTS_PATH = "reports/model3_daily_insights.csv"
WALK_FORWARD_PREDICTION_PATH = "reports/walk_forward_predictions.csv"
WALK_FORWARD_FOLD_METRICS_PATH = "reports/walk_forward_fold_metrics.csv"
WALK_FORWARD_BACKTEST_PATH = "reports/walk_forward_backtest.csv"
WALK_FORWARD_BACKTEST_METRICS_PATH = "reports/walk_forward_backtest_metrics.json"

HORIZON = 5

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
EARLY_STOPPING_ROUNDS = 50
BACKTEST_TOP_K = 5
BACKTEST_MIN_VOLUME = 100000
BACKTEST_MIN_CLOSE = 10.0
BACKTEST_MIN_BUY_PROBABILITY = 0.35
BACKTEST_MIN_BUY_SELL_MARGIN = 0.00
BACKTEST_TOP_K_VALUES = [5, 10, 20]
BACKTEST_MIN_VOLUME_VALUES = [50000, 100000, 200000]
BACKTEST_MIN_CLOSE_VALUES = [5.0, 10.0]
BACKTEST_MIN_BUY_PROBABILITY_VALUES = [0.35, 0.45, 0.55, 0.60, 0.65]
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.001
MAX_ABS_TARGET_RETURN = 0.2
SELL_RETURN_THRESHOLD = -0.025
BUY_RETURN_THRESHOLD = 0.025
SIGNAL_MIN_ACTION_PROBABILITY = 0.60
SIGNAL_MIN_ACTION_MARGIN = 0.00
SIGNAL_LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}
WALK_FORWARD_INITIAL_TRAIN_RATIO = 0.5
WALK_FORWARD_VALIDATION_RATIO = 0.1
WALK_FORWARD_TEST_RATIO = 0.1
WALK_FORWARD_STEP_RATIO = 0.1
WALK_FORWARD_MAX_FOLDS = 4

FEATURES = [
    "open", "high", "low", "close", "volume",

    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",

    "ma_5", "ma_20", "ma_50", "price_vs_ma20", "ma5_vs_ma20",

    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",

    "volatility_5d", "volatility_20d", "volatility_change",

    "rolling_max_20d", "drawdown_20d",

    "daily_range", "body_ratio", "close_position"
]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 3,
    "random_state": 42,
    "n_jobs": -1
}
