# Model 3 Schema Notes

Ghi chu khoa chinh/khoa ngoai logic cho cac report va mart cua Model 3.
CSV khong co constraint that nhu database, nen cac khoa duoi day la khoa de thiet ke bang ClickHouse, dashboard va API.

## Bang nguon lien quan

- `stock.features_all`
  - PK logic: `(trading_date, symbol)`
  - `symbol` FK logic toi bang danh muc ma co phieu, vi du `stock.stock_symbols(symbol)`.
  - `encode_sector` FK logic toi bang mapping nganh, vi du `stock.symbol_sector_encoding(encode_sector)`.

## Output chinh

### `reports/predictions.csv`

- Muc dich: prediction cap symbol/ngay tren tap test.
- PK logic: `(trading_date, symbol)`
- FK logic:
  - `(trading_date, symbol)` -> `stock.features_all(trading_date, symbol)`
  - `symbol` -> `stock.stock_symbols(symbol)`
  - `encode_sector` -> `stock.symbol_sector_encoding(encode_sector)` neu co bang mapping.
- Cot lien ket quan trong:
  - `future_trading_date` la ngay dung de so sanh ket qua that sau horizon 5 phien.
  - `target_signal_label` lien ket voi `target_signal`.
  - `predicted_signal_label` lien ket voi `predicted_signal`.
  - `adjusted_signal_label` lien ket voi `adjusted_signal`.

### `reports/prediction_accuracy.csv`

- Muc dich: bang doi chieu tin hieu that va tin hieu du doan.
- PK logic: `(date, symbol)`
- FK logic:
  - `(date, symbol)` -> `reports/predictions.csv(future_trading_date, symbol)`
  - `symbol` -> `stock.stock_symbols(symbol)`

### `reports/backtest.csv`

- Muc dich: ket qua backtest theo ngay.
- PK logic: `trading_date`
- FK logic:
  - `trading_date` -> ngay giao dich trong `stock.features_all(trading_date)`
  - `selected_symbols` la danh sach symbol dang chuoi, khong phai FK truc tiep. Neu dua len DB nen tach thanh bang chi tiet rieng co PK `(trading_date, symbol)`.

### `reports/metrics.json`

- Muc dich: metric danh gia phan loai.
- PK logic khi chuyen thanh bang: `(model_name, metric_group, metric_name, created_at)`
- Khong co FK truc tiep.

### `reports/backtest_metrics.json`

- Muc dich: metric tong hop backtest.
- PK logic khi chuyen thanh bang: `(model_name, metric_group, metric_name, created_at)`
- Khong co FK truc tiep.

## Mart Model 3

### `reports/mart_model3_trading_signals.csv`

- Muc dich: mart tin hieu giao dich cap symbol/ngay.
- PK logic: `(model_name, trading_date, symbol)`
- FK logic:
  - `(trading_date, symbol)` -> `stock.features_all(trading_date, symbol)`
  - `symbol` -> `stock.stock_symbols(symbol)`
  - `encode_sector` -> `stock.symbol_sector_encoding(encode_sector)` neu co bang mapping.
- Ghi chu:
  - Day la bang phu hop nhat de lam tab Trading Signals tren web dashboard.
  - `adjusted_signal` nen duoc uu tien de hien thi tin hieu hanh dong.

### `reports/mart_model3_signal_summary.csv`

- Muc dich: tong hop so BUY/HOLD/SELL theo ngay va nganh.
- PK logic: `(model_name, trading_date, sector)`
- FK logic:
  - `trading_date` -> ngay giao dich trong `stock.features_all(trading_date)`
  - `encode_sector` -> `stock.symbol_sector_encoding(encode_sector)` neu co bang mapping.
  - `sector` -> `stock.symbol_sector_encoding(sector)` hoac bang danh muc nganh tuong duong.
- Ghi chu:
  - Bang nay phuc vu card so luong BUY/HOLD/SELL va heatmap sector-signal.
  - `encode_sector` duoc giu lai de join ky thuat, con `sector` la ten nganh de hien thi tren dashboard.

### `reports/mart_model3_backtest_daily.csv`

- Muc dich: mart backtest theo ngay.
- PK logic: `(model_name, trading_date)`
- FK logic:
  - `trading_date` -> ngay giao dich trong `stock.features_all(trading_date)`
- Ghi chu:
  - `selected_symbols` la danh sach symbol dang chuoi. Neu can FK chuan, tao them bang `mart_model3_backtest_selected_symbols` voi PK `(model_name, trading_date, symbol)`.

### `reports/mart_model3_metrics.csv`

- Muc dich: metric dang long-format de dashboard/API doc truc tiep.
- PK logic: `(model_name, metric_group, metric_name, created_at)`
- FK logic:
  - `model_name` -> bang danh muc model neu co, vi du `mart_models(model_name)`.
- Ghi chu:
  - `metric_group` gom `classification` va `backtest`.

### `reports/model3_daily_insights.csv`

- Muc dich: insight co cau truc cho dashboard/Telegram/API.
- PK logic: `(insight_date, insight_type, source_model, metric_name, symbol, sector)`
- FK logic:
  - `source_model` -> bang danh muc model neu co, vi du `mart_models(model_name)`.
  - `symbol` -> `stock.stock_symbols(symbol)` khi khac rong.
  - `sector` hoac `encode_sector` -> bang mapping nganh khi khac rong.
  - `insight_date` -> ngay giao dich moi nhat trong `stock.features_all(trading_date)`.
- Ghi chu:
  - Cac cot `title` va `message` la noi dung hien thi; cac cot con lai giu insight co cau truc de filter/sort.

## Goi y DDL ClickHouse

Neu upload cac mart len ClickHouse, co the dung `MergeTree` va ORDER BY theo PK logic:

```sql
ORDER BY (model_name, trading_date, symbol)
```

cho `mart_model3_trading_signals`, va:

```sql
ORDER BY (model_name, trading_date, sector)
```

cho `mart_model3_signal_summary`.

ClickHouse khong enforce FK mac dinh nhu RDBMS, nen FK o tren la quan he logic de join va validate du lieu.
