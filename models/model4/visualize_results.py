# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: visualize_results.py
# Mục đích: Vẽ biểu đồ kết quả cho báo cáo
# ==============================================

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ==================
# PATHS
# ==================
BASE_DIR        = Path(__file__).resolve().parent
OUTPUT_DIR      = BASE_DIR / "output"
PREDICTIONS_CSV = OUTPUT_DIR / "benchmark_predictions.csv"
IMPORTANCE_CSV  = OUTPUT_DIR / "feature_importance.csv"
METRICS_JSON    = OUTPUT_DIR / "benchmark_metrics.json"
CHARTS_DIR      = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================
# ĐỌC DỮ LIỆU
# ==================
print("[model4] Đọc dữ liệu...")
df         = pd.read_csv(PREDICTIONS_CSV)
importance = pd.read_csv(IMPORTANCE_CSV)
with open(METRICS_JSON, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# ==================
# BIỂU ĐỒ 1 — Feature Importance
# ==================
print("[model4] Vẽ Feature Importance...")
top20 = importance.head(20)
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top20["feature"][::-1], top20["importance"][::-1],
               color="steelblue", edgecolor="white")
for bar, val in zip(bars, top20["importance"][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=9)
ax.set_title("Top 20 Feature Importance — Model4 Benchmark Outperformance",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "1_feature_importance.png", dpi=150)
plt.close()
print("[model4] Saved: 1_feature_importance.png")

# ==================
# BIỂU ĐỒ 2 — Confusion Matrix
# ==================
print("[model4] Vẽ Confusion Matrix...")
cm     = np.array(metrics["confusion_matrix"])
labels = ["Không Outperform (0)", "Outperform (1)"]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > thresh else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=10); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — Model4 Benchmark Outperformance",
             fontsize=13, fontweight="bold", pad=15)
textstr = (f"Accuracy:  {metrics['accuracy']:.4f}\n"
           f"Precision: {metrics['precision']:.4f}\n"
           f"Recall:    {metrics['recall']:.4f}\n"
           f"F1-score:  {metrics['f1']:.4f}")
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=10, va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("[model4] Saved: 2_confusion_matrix.png")

# ==================
# BIỂU ĐỒ 3 — ROC Curve
# ==================
print("[model4] Vẽ ROC Curve...")
fpr, tpr, _ = roc_curve(df["label"], df["outperform_probability"])
roc_auc     = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
        label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Model4 Benchmark Outperformance",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "3_roc_curve.png", dpi=150)
plt.close()
print("[model4] Saved: 3_roc_curve.png")

# ==================
# BIỂU ĐỒ 4 — Phân phối xác suất
# ==================
print("[model4] Vẽ phân phối xác suất...")
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(df[df["label"] == 0]["outperform_probability"],
        bins=50, alpha=0.6, color="tomato",
        label="Thực tế: Không Outperform (0)", density=True)
ax.hist(df[df["label"] == 1]["outperform_probability"],
        bins=50, alpha=0.6, color="steelblue",
        label="Thực tế: Outperform (1)", density=True)
ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Ngưỡng = 0.5")
ax.set_xlabel("Xác suất dự đoán Outperform", fontsize=11)
ax.set_ylabel("Mật độ", fontsize=11)
ax.set_title("Phân phối xác suất dự đoán — Model4",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "4_probability_distribution.png", dpi=150)
plt.close()
print("[model4] Saved: 4_probability_distribution.png")

print(f"\n[model4] Đã lưu 4 biểu đồ vào: {CHARTS_DIR}")
print("[model4] HOÀN THÀNH! ✅")