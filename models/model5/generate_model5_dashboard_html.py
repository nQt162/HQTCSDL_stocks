from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "output_model5"
HTML_PATH = OUTPUT_DIR / "model5_dashboard.html"

METRICS_PATH = OUTPUT_DIR / "risk_metrics.json"
RISK_PREDICTIONS_PATH = OUTPUT_DIR / "risk_predictions.csv"
RISK_ALERTS_PATH = OUTPUT_DIR / "mart_risk_alerts.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtest_risk_alerts.csv"

RISK_RED = "#dc2626"
RISK_DARK = "#7f1d1d"
RISK_ORANGE = "#ea580c"
LOW_GREEN = "#2a9d8f"
NAVY = "#1d3557"
MUTED = "#6c757d"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def as_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def date_key(value) -> str:
    if value is None:
        return ""
    return str(value)[:10]


def num(value, digits: int = 0) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:,.{digits}f}"


def pct(value, digits: int = 1, already_percent: bool = False) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "N/A"
    if not already_percent:
        numeric *= 100
    return f"{numeric:,.{digits}f}%"


def compact(value) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "N/A"
    sign = "-" if numeric < 0 else ""
    numeric = abs(numeric)
    if numeric >= 1_000_000:
        return f"{sign}{numeric / 1_000_000:.2f}M"
    if numeric >= 1_000:
        return f"{sign}{numeric / 1_000:.1f}K"
    return f"{sign}{numeric:.0f}"


def mean(values) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def metric_card(title: str, value: str, caption: str = "", color: str = RISK_RED) -> str:
    return (
        f'<section class="metric-card" style="border-left-color:{color}">'
        f"<p>{esc(title)}</p><strong>{esc(value)}</strong>"
        f"<span>{esc(caption)}</span></section>"
    )


def distribution_card(title: str, rows: list[tuple[str, int, float, str]]) -> str:
    items = []
    for label, count, ratio, color in rows:
        items.append(
            '<div class="dist-item">'
            f'<span><i style="background:{color}"></i>{esc(label)}</span>'
            f"<b>{count:,}</b><small>{pct(ratio)}</small></div>"
        )
    return f'<div class="dist-card"><h3>{esc(title)}</h3>{"".join(items)}</div>'


def probability_meter(value, color: str = RISK_RED) -> str:
    numeric = as_float(value, 0.0)
    numeric = max(0.0, min(1.0, numeric))
    return (
        '<div class="prob-meter">'
        f'<span style="width:{numeric * 100:.1f}%;background:{color}"></span>'
        f"<b>{pct(numeric)}</b></div>"
    )


def latest_rows(rows: list[dict[str, str]], date_col: str) -> list[dict[str, str]]:
    if not rows:
        return []
    latest = max(date_key(row.get(date_col)) for row in rows)
    result = [row for row in rows if date_key(row.get(date_col)) == latest]
    result.sort(key=lambda row: (as_float(row.get("risk_probability"), -1.0), row.get("symbol", "")), reverse=True)
    return result


def line_chart(
    rows: list[dict[str, str]],
    x_col: str,
    series: list[tuple[str, str, str]],
    *,
    height: int = 300,
    width: int = 780,
) -> str:
    chart_rows = [row for row in rows if date_key(row.get(x_col))]
    chart_rows.sort(key=lambda row: date_key(row.get(x_col)))
    if not chart_rows:
        return '<p class="empty">Line chart data is not available.</p>'

    available = []
    all_values = []
    for label, col, color in series:
        values = [as_float(row.get(col)) for row in chart_rows]
        values = [value for value in values if value is not None]
        if values:
            available.append((label, col, color))
            all_values.extend(values)

    if not available or not all_values:
        return '<p class="empty">Line chart series is not available.</p>'

    min_y = min(all_values)
    max_y = max(all_values)
    if math.isclose(min_y, max_y):
        min_y -= 1
        max_y += 1

    left, right, top, bottom = 42, 20, 28, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(chart_rows)

    def x_pos(index: int) -> float:
        if n <= 1:
            return left
        return left + (index / (n - 1)) * plot_w

    def y_pos(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_h

    polylines = []
    for label, col, color in available:
        points = []
        for index, row in enumerate(chart_rows):
            value = as_float(row.get(col))
            if value is None:
                continue
            points.append(f"{x_pos(index):.2f},{y_pos(value):.2f}")
        if points:
            polylines.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="3" stroke-linejoin="round" />'
            )

    legend = "".join(
        f'<span><i style="background:{color}"></i>{esc(label)}</span>'
        for label, _, color in available
    )
    top_label = pct(max_y) if max_y <= 2 else num(max_y)
    bottom_label = pct(min_y) if min_y <= 2 else num(min_y)

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Risk trend">'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#d8dee4" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#d8dee4" />'
        f'<text x="{left}" y="20">{top_label}</text>'
        f'<text x="{left}" y="{height-8}">{bottom_label}</text>'
        f'{"".join(polylines)}</svg><div class="legend">{legend}</div>'
    )


def bar_chart(rows: list[dict[str, str]], label_col: str, value_col: str, *, limit: int = 12, color: str = RISK_RED) -> str:
    chart_rows = [row for row in rows if as_float(row.get(value_col)) is not None]
    chart_rows.sort(key=lambda row: as_float(row.get(value_col), 0.0), reverse=True)
    chart_rows = chart_rows[:limit]
    if not chart_rows:
        return '<p class="empty">Bar chart data is not available.</p>'
    max_value = max(as_float(row.get(value_col), 0.0) for row in chart_rows) or 1.0
    html_rows = []
    for row in chart_rows:
        value = as_float(row.get(value_col), 0.0)
        width = max(3, value / max_value * 100)
        html_rows.append(
            '<div class="bar-row">'
            f'<span title="{esc(row.get(label_col))}">{esc(row.get(label_col))}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div></div>'
            f"<b>{num(value, 4)}</b></div>"
        )
    return f'<div class="bar-chart">{"".join(html_rows)}</div>'


def matrix_table(matrix: list[list[int]] | None) -> str:
    if not matrix or len(matrix) < 2 or len(matrix[0]) < 2 or len(matrix[1]) < 2:
        return '<p class="empty">Confusion matrix is not available.</p>'
    rows = [("Actual LOW_RISK", matrix[0]), ("Actual HIGH_RISK", matrix[1])]
    headers = ["Pred LOW_RISK", "Pred HIGH_RISK"]
    max_value = max(max(row) for _, row in rows) or 1
    body_rows = []
    for label, values in rows:
        row_sum = sum(values) or 1
        cells = []
        for value in values:
            opacity = 0.14 + (float(value) / max_value) * 0.66
            cells.append(
                f'<td style="background:rgba(220,38,38,{opacity:.3f})">'
                f"<b>{int(value):,}</b><small>{value / row_sum:.1%} row</small></td>"
            )
        body_rows.append(f"<tr><th>{esc(label)}</th>{''.join(cells)}</tr>")
    return (
        '<table class="confusion-matrix"><thead><tr><th>Actual / Pred</th>'
        + "".join(f"<th>{esc(header)}</th>" for header in headers)
        + f"</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def table_html(rows: list[dict[str, str]], columns: list[str], *, limit: int = 10) -> str:
    if not rows:
        return '<p class="empty">Table data is not available.</p>'
    available = [col for col in columns if any(col in row for row in rows)]
    if not available:
        return '<p class="empty">Table columns are not available.</p>'
    headers = "".join(f"<th>{esc(col)}</th>" for col in available)
    body_rows = []
    for row in rows[:limit]:
        cells = []
        for col in available:
            value = row.get(col)
            if col == "risk_probability":
                value_html = probability_meter(value, RISK_RED)
            elif col == "risk_label":
                color = RISK_RED if value == "HIGH_RISK" else LOW_GREEN
                value_html = f'<span class="risk-badge" style="--badge-color:{color}">{esc(value)}</span>'
            elif "probability" in col or col in {
                "precision_high_risk",
                "recall_high_risk",
                "avg_risk_probability",
                "avg_alert_probability",
                "return_5d",
                "drawdown_20d",
                "volatility_5d",
            }:
                value_html = pct(value)
            elif col in {
                "volume",
                "total_symbols",
                "alert_count",
                "actual_high_risk_count",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
            }:
                value_html = compact(value)
            else:
                value_html = esc(value)
            cells.append(f"<td>{value_html}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="table-wrap"><table class="data-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def make_dashboard() -> str:
    metrics = read_json(METRICS_PATH)
    predictions = read_csv_rows(RISK_PREDICTIONS_PATH)
    alerts = read_csv_rows(RISK_ALERTS_PATH)
    feature_importance = read_csv_rows(FEATURE_IMPORTANCE_PATH)
    backtest = read_csv_rows(BACKTEST_PATH)

    selected_model = metrics.get("selected_model", "N/A")
    model_metrics = metrics.get("models", {}).get(selected_model, {})
    threshold = metrics.get("threshold", 0.6)

    source = predictions or alerts
    total_rows = len(source)
    symbols = {row.get("symbol") for row in source if row.get("symbol")}
    symbol_count = len(symbols)
    high_risk_count = sum(1 for row in source if row.get("risk_label") == "HIGH_RISK")
    avg_risk = mean([as_float(row.get("risk_probability")) for row in source])
    max_risk = max([as_float(row.get("risk_probability"), 0.0) for row in source], default=None)
    latest_date = max([date_key(row.get("prediction_date")) for row in source], default="N/A")

    latest = latest_rows(alerts or predictions, "prediction_date")

    counts = {"HIGH_RISK": 0, "LOW_RISK": 0}
    for row in source:
        label = row.get("risk_label")
        if label in counts:
            counts[label] += 1
    total_label = max(sum(counts.values()), 1)
    distribution = distribution_card(
        "Risk Label Distribution",
        [
            ("HIGH_RISK", counts["HIGH_RISK"], counts["HIGH_RISK"] / total_label, RISK_RED),
            ("LOW_RISK", counts["LOW_RISK"], counts["LOW_RISK"] / total_label, LOW_GREEN),
        ],
    )

    comparison_cards = []
    for name, values in metrics.get("models", {}).items():
        comparison_cards.append(
            '<div class="dist-card model-card">'
            f"<h3>{esc(name)}</h3>"
            f'<div class="dist-item"><span>Accuracy</span><b>{pct(values.get("accuracy"))}</b></div>'
            f'<div class="dist-item"><span>Precision HIGH</span><b>{pct(values.get("precision_high_risk"))}</b></div>'
            f'<div class="dist-item"><span>Recall HIGH</span><b>{pct(values.get("recall_high_risk"))}</b></div>'
            f'<div class="dist-item"><span>ROC-AUC</span><b>{pct(values.get("roc_auc"))}</b></div>'
            "</div>"
        )

    top_alert_table = table_html(
        latest,
        [
            "prediction_date",
            "target_date",
            "symbol",
            "close",
            "volume",
            "return_5d",
            "drawdown_20d",
            "volatility_5d",
            "risk_probability",
            "risk_label",
        ],
        limit=12,
    )

    quality = list(backtest)
    quality.sort(key=lambda row: as_float(row.get("avg_alert_probability"), 0.0), reverse=True)
    quality_table = table_html(
        quality,
        [
            "prediction_date",
            "total_symbols",
            "alert_count",
            "actual_high_risk_count",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "precision_high_risk",
            "recall_high_risk",
            "avg_alert_probability",
        ],
        limit=8,
    )

    top_symbol = "N/A"
    top_prob = "N/A"
    if latest:
        top_symbol = latest[0].get("symbol", "N/A")
        top_prob = pct(latest[0].get("risk_probability"))

    insights = [
        f"Selected model is {selected_model} with threshold {pct(threshold)}.",
        f"Latest risk date is {latest_date}; highest-risk symbol is {top_symbol} with risk probability {top_prob}.",
        f"Test accuracy is {pct(model_metrics.get('accuracy'))}; ROC-AUC is {pct(model_metrics.get('roc_auc'))}.",
        f"Precision HIGH_RISK is {pct(model_metrics.get('precision_high_risk'))}; recall HIGH_RISK is {pct(model_metrics.get('recall_high_risk'))}.",
    ]
    insight_html = "".join(f"<li>{esc(item)}</li>" for item in insights)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model 5 Risk Alert Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #1d2530;
      --muted: #6c757d;
      --line: #d8dee4;
      --red: {RISK_RED};
      --dark-red: {RISK_DARK};
      --orange: {RISK_ORANGE};
      --green: {LOW_GREEN};
      --navy: {NAVY};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.4;
    }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    header {{
      padding: 24px 28px;
      border-radius: 16px;
      color: #fff;
      background: linear-gradient(135deg, #7f1d1d, #dc2626 58%, #f97316);
      margin-bottom: 16px;
      box-shadow: 0 12px 28px rgba(127, 29, 29, 0.18);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    header p {{ margin: 0; opacity: .9; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .metric-card, .panel, .dist-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 22px rgba(29, 37, 48, 0.07);
    }}
    .metric-card {{
      padding: 16px;
      min-height: 118px;
      border-left: 6px solid var(--red);
    }}
    .metric-card p {{ margin: 0 0 8px; color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    .metric-card strong {{ display: block; font-size: 28px; color: var(--dark-red); }}
    .metric-card span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; }}
    .panel {{ padding: 18px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .legend i {{ width: 11px; height: 11px; display: inline-block; border-radius: 999px; margin-right: 6px; }}
    .distribution-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .dist-card {{ padding: 14px; }}
    .dist-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .dist-item {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 0; border-top: 1px solid #eef1f4; }}
    .dist-item:first-of-type {{ border-top: 0; }}
    .dist-item span {{ color: var(--muted); font-size: 13px; }}
    .dist-item i {{ width: 10px; height: 10px; display: inline-block; border-radius: 999px; margin-right: 7px; }}
    .dist-item b {{ font-size: 16px; }}
    .dist-item small {{ color: var(--muted); }}
    .market-strip {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 14px;
      background: #fff;
    }}
    .market-strip span {{ padding: 12px; border-right: 1px solid var(--line); }}
    .market-strip span:last-child {{ border-right: 0; }}
    .market-strip b {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .table-wrap {{ overflow-x: auto; }}
    .data-table, .confusion-matrix {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table th, .data-table td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f2; text-align: left; white-space: nowrap; }}
    .data-table th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .confusion-matrix th, .confusion-matrix td {{ padding: 12px; border: 1px solid #fff; text-align: center; }}
    .confusion-matrix th {{ color: var(--muted); background: #f1f5f9; }}
    .confusion-matrix b, .confusion-matrix small {{ display: block; }}
    .confusion-matrix small {{ color: var(--muted); margin-top: 3px; }}
    .prob-meter {{ min-width: 110px; height: 22px; background: #f1f5f9; border-radius: 999px; overflow: hidden; position: relative; }}
    .prob-meter span {{ position: absolute; left: 0; top: 0; bottom: 0; opacity: .82; }}
    .prob-meter b {{ position: relative; z-index: 1; display: block; text-align: center; font-size: 12px; line-height: 22px; color: #111827; }}
    .risk-badge {{ background: color-mix(in srgb, var(--badge-color) 15%, white); color: var(--badge-color); border: 1px solid color-mix(in srgb, var(--badge-color) 45%, white); border-radius: 999px; font-weight: 800; padding: 4px 9px; }}
    .bar-row {{ display: grid; grid-template-columns: 160px 1fr 80px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-row span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; color: var(--muted); }}
    .bar-track {{ height: 18px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .bar-row b {{ font-size: 12px; text-align: right; }}
    .insight-list {{ margin: 0; padding-left: 18px; }}
    .insight-list li {{ margin: 8px 0; }}
    .empty {{ color: var(--muted); margin: 0; }}
    svg {{ width: 100%; height: auto; display: block; }}
    svg text {{ fill: var(--muted); font-size: 12px; }}
    @media (max-width: 1100px) {{
      .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .grid-2, .distribution-grid {{ grid-template-columns: 1fr; }}
      .market-strip {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Model 5 Risk Alert Dashboard</h1>
      <p>Risk alert model for HIGH_RISK / LOW_RISK stocks, generated from output_model5 CSV and JSON files.</p>
    </header>

    <section class="metric-grid">
      {metric_card("Selected Model", str(selected_model), "Risk classifier in use", RISK_DARK)}
      {metric_card("Threshold", pct(threshold), "HIGH_RISK if probability >= threshold", RISK_ORANGE)}
      {metric_card("Accuracy", pct(model_metrics.get("accuracy")), "Test classification accuracy", RISK_DARK)}
      {metric_card("Precision HIGH", pct(model_metrics.get("precision_high_risk")), "When model warns HIGH_RISK", RISK_RED)}
      {metric_card("Recall HIGH", pct(model_metrics.get("recall_high_risk")), "Actual HIGH_RISK captured", RISK_ORANGE)}
      {metric_card("ROC-AUC", pct(model_metrics.get("roc_auc")), "Ranking quality", NAVY)}
    </section>

    <section class="market-strip">
      <span><b>Latest date</b>{esc(latest_date)}</span>
      <span><b>Symbols</b>{symbol_count:,}</span>
      <span><b>HIGH_RISK rows</b>{high_risk_count:,}</span>
      <span><b>Avg risk</b>{pct(avg_risk)}</span>
      <span><b>Max risk</b>{pct(max_risk)}</span>
    </section>

    <section class="grid-2">
      <section class="panel">
        <h2>Daily Risk Monitoring</h2>
        {line_chart(backtest, "prediction_date", [
            ("Precision HIGH", "precision_high_risk", RISK_RED),
            ("Recall HIGH", "recall_high_risk", RISK_ORANGE),
            ("Avg alert probability", "avg_alert_probability", NAVY),
        ])}
      </section>
      <section class="panel">
        <h2>Risk Distribution & Model Comparison</h2>
        <div class="distribution-grid">
          {distribution}
          {''.join(comparison_cards)}
        </div>
      </section>
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Top HIGH_RISK Alerts - Latest Date</h2>
      {top_alert_table}
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Model 5 Insights</h2>
      <ul class="insight-list">{insight_html}</ul>
    </section>

    <section class="grid-2" style="margin-top:14px">
      <section class="panel">
        <h2>Confusion Matrix</h2>
        {matrix_table(model_metrics.get("confusion_matrix"))}
      </section>
      <section class="panel">
        <h2>Top Feature Importance</h2>
        {bar_chart(feature_importance, "feature", "importance", limit=12, color=RISK_RED)}
      </section>
    </section>

    <section class="panel" style="margin-top:14px">
      <h2>Best / Latest Risk Alert Backtest Days</h2>
      {quality_table}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(make_dashboard(), encoding="utf-8")
    print(f"Saved {HTML_PATH}")


if __name__ == "__main__":
    main()
