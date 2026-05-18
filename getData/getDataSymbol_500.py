# Crawl dữ liệu 500 mã cổ phiếu
# Batch size = 10
# Sleep = 90s
# Source = KBS/VCI

import os
import time
import ast
import random
import pandas as pd

# API mới của vnstock
from vnstock.api.quote import Quote

# =========================
# CONFIG
# =========================

DATA_DIR = "ML/getData/data/data_clean"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "Data_500_stocks_2015-2026.csv"
    # "Data_500_stocks_01-17.csv"
)

SYMBOL_FILE = "ML/getData/symbol500.txt"

SOURCE = "KBS"

# START_DATE = "2026-05-01"
# END_DATE = "2026-05-17"
START_DATE = "2015-01-01"
END_DATE = "2026-04-30"

BATCH_SIZE = 10
SLEEP_TIME = 90

# =========================
# Đọc danh sách mã
# =========================

all_symbols = []

with open(SYMBOL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        symbols = ast.literal_eval(line)
        all_symbols.extend(symbols)

# lấy tối đa 500 mã
all_symbols = all_symbols[:500]
print(f"Tổng số mã: {len(all_symbols)}")

# =========================
# Chia batch
# =========================
batches = [
    all_symbols[i:i + BATCH_SIZE]
    for i in range(0, len(all_symbols), BATCH_SIZE)
]

print(f"Tổng batch: {len(batches)}")

# =========================
# XÓA FILE CŨ NẾU TỒN TẠI
# =========================
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# =========================
# CRAWL
# =========================
for batch_index, batch in enumerate(batches, start=1):
    print("\n" + "=" * 50)
    print(f"Batch {batch_index}/{len(batches)}")
    print("=" * 50)

    batch_df_list = []
    for symbol in batch:
        try:
            print(f"\nĐang lấy: {symbol}")

            # =========================
            # API mới
            # =========================
            q = Quote(
                symbol=symbol,
                source=SOURCE
            )

            df_history = q.history(
                start=START_DATE,
                end=END_DATE
            )

            # =========================
            # Check dữ liệu
            # =========================
            if df_history is None or df_history.empty:
                print(f"Không có dữ liệu: {symbol}")
                continue

            # thêm mã cổ phiếu
            df_history["symbol"] = symbol
            batch_df_list.append(df_history)
            print(f"Hoàn tất: {symbol}")

            # sleep nhỏ giữa từng request
            time.sleep(random.uniform(2, 5))
        except Exception as e:
            print(f"Lỗi với {symbol}: {e}")

    # =========================
    # GHI FILE THEO BATCH
    # =========================
    if batch_df_list:
        batch_df = pd.concat(
            batch_df_list,
            ignore_index=True
        )

        # batch đầu tiên -> ghi mới
        if batch_index == 1:
            batch_df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
                mode="w"
            )

        # batch sau -> append
        else:
            batch_df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
                mode="a",
                header=False
            )
        print(f"\nĐã ghi batch {batch_index} vào file")

    # =========================
    # Sleep giữa batch
    # =========================
    if batch_index < len(batches):
        random_sleep = random.randint(
            SLEEP_TIME - 10,
            SLEEP_TIME + 20
        )

        print(f"\nSleep {random_sleep}s...\n")
        time.sleep(random_sleep)

# =========================
# DONE
# =========================
print("\n" + "=" * 50)
print("HOÀN TẤT")
print(f"Lưu file tại: {OUTPUT_FILE}")
print("=" * 50)