from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VN_TZ = timezone(timedelta(hours=7))
SUMMARY_PATH = PROJECT_ROOT / "data" / "pipeline_run_summary.json"

PIPELINE = [
	# (
	# 	"merge_01_17_to_dirty",
	# 	PROJECT_ROOT / "tmp_merge_01_17_to_dirty.py",
	# ),
	(
		"merge_daily_csv_to_dirty",
		PROJECT_ROOT / "ingestion" / "merge_daily_csv_to_dirty.py",
	),
	(
		"survey_data",
		PROJECT_ROOT / "etl" / "xu_li_du_lieu" / "kiemtradl_ghiralog_extract.py",
	),
	(
		"clean_data",
		PROJECT_ROOT / "etl" / "xu_li_du_lieu" / "clean_db_ghiralog_transform.py",
	),
	(
		"load_prices",
		PROJECT_ROOT / "connect_clickhouse" / "load_prices_to_click_house.py",
	),
	(
		"load_symbols",
		PROJECT_ROOT / "connect_clickhouse" / "load_symbols_to_clickhouse.py",
	),
	(
		"upload_features_all",
		PROJECT_ROOT / "connect_clickhouse" / "features_all.py",
	),
	("train_model1", PROJECT_ROOT / "models" / "model1" / "main.py"),
	("train_model2", PROJECT_ROOT / "models" / "model2" / "main.py"),
	("train_model3", PROJECT_ROOT / "models" / "model3" / "main.py"),
	(
		"train_model4",
		PROJECT_ROOT / "models" / "model4" / "train_benchmark_model.py",
	),
	(
		"upload_model4_outputs",
		PROJECT_ROOT / "models" / "model4" / "upload_predictions.py",
	),
	("train_model5", PROJECT_ROOT / "models" / "model5" / "run_pipeline.py"),
	(
		"upload_model5_outputs",
		PROJECT_ROOT / "models" / "model5" / "upload_outputs_to_clickhouse.py",
	),
	(
		"upload_all_tables",
		PROJECT_ROOT / "upload_all.py",
		["--layers", "mart,audit"],
	),
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the full stock data, model, and ClickHouse pipeline."
	)
	return parser.parse_args()


def unpack_step(step):
	if len(step) == 2:
		name, script_path = step
		args = []
	elif len(step) == 3:
		name, script_path, args = step
	else:
		raise ValueError(f"Invalid pipeline step: {step}")

	return name, script_path, list(args)


def display_script_path(script_path: Path) -> str:
	try:
		return script_path.relative_to(PROJECT_ROOT).as_posix()
	except ValueError:
		return script_path.name


def run_step(name: str, script_path: Path, args: list[str] | None = None) -> None:
	if not script_path.exists():
		raise FileNotFoundError(f"Missing script: {script_path}")

	print(f"[pipeline] Running {name}: {display_script_path(script_path)}")
	command = [sys.executable, str(script_path), *(args or [])]
	subprocess.run(
		command,
		check=True,
		cwd=PROJECT_ROOT,
	)
	print(f"[pipeline] Finished {name}")


def write_pipeline_summary(
	status: str,
	completed_steps: list[str],
	failed_step: str | None = None,
) -> None:
	SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"status": status,
		"updated_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
		"completed_steps": completed_steps,
		"failed_step": failed_step,
	}
	SUMMARY_PATH.write_text(
		json.dumps(payload, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)


def main() -> None:
	parse_args()
	print("[pipeline] Starting full stock pipeline")
	completed_steps: list[str] = []
	write_pipeline_summary("running", completed_steps)

	try:
		for step in PIPELINE:
			name, script_path, args = unpack_step(step)
			run_step(name, script_path, args=args)
			completed_steps.append(name)
			write_pipeline_summary("running", completed_steps)
	except Exception:
		failed_step = next(
			(
				unpack_step(step)[0]
				for step in PIPELINE
				if unpack_step(step)[0] not in completed_steps
			),
			"unknown",
		)
		write_pipeline_summary("failed", completed_steps, failed_step=failed_step)
		raise

	write_pipeline_summary("success", completed_steps)
	print("[pipeline] All steps completed")


if __name__ == "__main__":
	main()
