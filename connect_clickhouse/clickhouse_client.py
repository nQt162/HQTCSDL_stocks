import os
import time
from pathlib import Path

import clickhouse_connect
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def get_clickhouse_client():
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port_raw = os.getenv("CLICKHOUSE_PORT")
    username = os.getenv("CLICKHOUSE_USER", os.getenv("CLICKHOUSE_USERNAME", "default"))
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    database = os.getenv("CLICKHOUSE_DATABASE", "default")

    if not port_raw or port_raw.strip() == "":
        port = 8443 if "clickhouse.cloud" in host else 8123
    else:
        port = int(port_raw)

    default_secure = port == 8443 or "clickhouse.cloud" in host
    secure = _get_bool_env("CLICKHOUSE_SECURE", default_secure)
    connect_timeout = _get_int_env("CLICKHOUSE_CONNECT_TIMEOUT", 60)
    send_receive_timeout = _get_int_env("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", 600)
    query_retries = _get_int_env("CLICKHOUSE_QUERY_RETRIES", 3)
    client_retries = _get_int_env("CLICKHOUSE_CLIENT_RETRIES", 3)
    retry_sleep_seconds = _get_int_env("CLICKHOUSE_RETRY_SLEEP_SECONDS", 10)

    last_error = None
    for attempt in range(1, client_retries + 1):
        try:
            client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
                secure=secure,
                connect_timeout=connect_timeout,
                send_receive_timeout=send_receive_timeout,
                query_retries=query_retries,
            )
            print(
                "Connected to ClickHouse: "
                f"{host}:{port} (Secure: {secure}, Database: {database})"
            )
            return client
        except Exception as exc:
            last_error = exc
            if attempt >= client_retries:
                break
            print(
                "[clickhouse] Connection attempt "
                f"{attempt}/{client_retries} failed: {exc}. "
                f"Retrying in {retry_sleep_seconds}s..."
            )
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(
        "Cannot connect to ClickHouse after "
        f"{client_retries} attempts. Host={host}, port={port}, "
        f"secure={secure}, connect_timeout={connect_timeout}s"
    ) from last_error
