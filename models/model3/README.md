# HQTCSDL Model 3 - Trading Signal Classification

Model 3 phan loai tin hieu BUY/HOLD/SELL dua tren future return bang XGBoost Classifier.

Nguon du lieu:

- ClickHouse Cloud: `stock.features_all`
- Can set `CLICKHOUSE_PASSWORD` trong terminal/session truoc khi chay.

Mac dinh:

- SELL: future return <= -2.5%
- HOLD: -2.5% < future return < 2.5%
- BUY: future return >= 2.5%

Chay train va tao report:

```bash
python main.py
```

Chay walk-forward backtest:

```bash
python walk_forward.py
```

Tao dashboard HTML tu cac report da sinh:

```bash
python dashboard_model3.py
```

Output chinh:

- `models/trading_signal_xgb_classifier.pkl`
- `reports/predictions.csv`
- `reports/metrics.json`
- `reports/backtest.csv`
- `reports/backtest_metrics.json`
- `reports/model3_dashboard.html`
- `reports/mart_model3_trading_signals.csv`
- `reports/mart_model3_signal_summary.csv`
- `reports/mart_model3_backtest_daily.csv`
- `reports/mart_model3_metrics.csv`
- `reports/model3_daily_insights.csv`

Ghi chu schema/khoa:

- Xem `MODEL3_SCHEMA.md` de biet PK/FK logic cua tung report va mart.
