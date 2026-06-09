from src.config import (
    CLICKHOUSE_DATABASE,
    MODEL3_CLICKHOUSE_END_DATE,
    CLICKHOUSE_FEATURES_TABLE,
    CLICKHOUSE_HOST,
    MODEL3_CLICKHOUSE_LIMIT,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    MODEL3_CLICKHOUSE_START_DATE,
    CLICKHOUSE_USERNAME,
    FEATURES,
)


def quote_identifier(identifier):
    return "`" + str(identifier).replace("`", "``") + "`"


def quote_literal(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def load_data_from_clickhouse():
    if not CLICKHOUSE_HOST:
        raise ValueError("CLICKHOUSE_HOST is required")

    if not CLICKHOUSE_PASSWORD:
        raise ValueError("CLICKHOUSE_PASSWORD is required")

    try:
        import clickhouse_connect
    except ImportError as exc:
        raise ImportError(
            "clickhouse-connect is required for ClickHouse loading. "
            "Install it with: pip install clickhouse-connect"
        ) from exc

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )

    table_name = (
        f"{quote_identifier(CLICKHOUSE_DATABASE)}."
        f"{quote_identifier(CLICKHOUSE_FEATURES_TABLE)}"
    )

    selected_columns = list(
        dict.fromkeys(["trading_date", "symbol", "encode_sector"] + FEATURES)
    )
    columns_sql = ",\n            ".join(
        quote_identifier(column) for column in selected_columns
    )

    filters = []
    if MODEL3_CLICKHOUSE_START_DATE:
        filters.append(
            "toDate(trading_date) >= "
            f"toDate({quote_literal(MODEL3_CLICKHOUSE_START_DATE)})"
        )
    if MODEL3_CLICKHOUSE_END_DATE:
        filters.append(
            "toDate(trading_date) <= "
            f"toDate({quote_literal(MODEL3_CLICKHOUSE_END_DATE)})"
        )

    where_sql = "WHERE " + " AND ".join(filters) if filters else ""
    limit_sql = f"LIMIT {MODEL3_CLICKHOUSE_LIMIT}" if MODEL3_CLICKHOUSE_LIMIT > 0 else ""

    query = f"""
        SELECT
            {columns_sql}
        FROM {table_name}
        {where_sql}
        ORDER BY symbol, trading_date
        {limit_sql}
    """
    df = client.query_df(query)
    df.columns = df.columns.str.strip()
    return df


def load_data(path=None):
    print(
        "[model3] Loading data from ClickHouse table "
        f"{CLICKHOUSE_DATABASE}.{CLICKHOUSE_FEATURES_TABLE}"
    )
    if MODEL3_CLICKHOUSE_START_DATE or MODEL3_CLICKHOUSE_END_DATE:
        print(
            "[model3] Date filter: "
            f"{MODEL3_CLICKHOUSE_START_DATE or 'START'} -> "
            f"{MODEL3_CLICKHOUSE_END_DATE or 'END'}"
        )
    if MODEL3_CLICKHOUSE_LIMIT > 0:
        print(f"[model3] Row limit: {MODEL3_CLICKHOUSE_LIMIT:,}")
    return load_data_from_clickhouse()
