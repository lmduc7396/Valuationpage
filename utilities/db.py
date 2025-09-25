"""Database utilities using pymssql for SQL Server access."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pymssql

# Environment variables that should contain SQL Server connection strings
SOURCE_DB_ENV = "SOURCE_DB_CONNECTION_STRING"
TARGET_DB_ENV = "TARGET_DB_CONNECTION_STRING"

# Default schema to use when callers pass an unqualified table name
DEFAULT_SCHEMA = "dbo"


def _get_connection_string(db: str) -> str:
    env_var = TARGET_DB_ENV if db == "target" else SOURCE_DB_ENV
    conn_str = os.getenv(env_var)
    if conn_str:
        return conn_str

    # Streamlit Cloud exposes secrets via st.secrets, not environment vars
    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if secrets is not None and env_var in secrets:
            return secrets[env_var]
    except Exception:
        pass

    raise RuntimeError(
        f"Connection information for '{env_var}' is not set. "
        "Please define it as an environment variable or Streamlit secret."
    )


def _parse_connection_string(conn_str: str) -> Dict[str, str]:
    tokens: List[str] = []
    current = []
    in_quotes = False
    for ch in conn_str.strip():
        if ch == '"':
            in_quotes = not in_quotes
        if ch == ';' and not in_quotes:
            token = ''.join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(ch)
    if current:
        token = ''.join(current).strip()
        if token:
            tokens.append(token)

    params: Dict[str, str] = {}
    for token in tokens:
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        key = key.strip().upper()
        if key == 'DRIVER':
            continue
        params[key] = value.strip().strip('"')
    return params


def _build_connection_kwargs(params: Dict[str, str], autocommit: bool) -> Dict[str, object]:
    server_val = params.get('SERVER') or params.get('HOST') or params.get('ADDRESS')
    if not server_val:
        raise RuntimeError("Connection string is missing SERVER information")

    server = server_val
    port = None
    if server.lower().startswith('tcp:'):
        server = server[4:]
    if ',' in server:
        host_part, port_part = server.split(',', 1)
        server = host_part
        try:
            port = int(port_part)
        except ValueError:
            port = None

    user = params.get('UID') or params.get('USER ID') or params.get('USERNAME')
    password = params.get('PWD') or params.get('PASSWORD')
    database = params.get('DATABASE') or params.get('DB') or params.get('INITIAL CATALOG')

    if not user or not password or not database:
        raise RuntimeError("Connection string must include UID, PWD, and DATABASE values")

    kwargs: Dict[str, object] = {
        'server': server,
        'user': user,
        'password': password,
        'database': database,
        'autocommit': autocommit,
        'charset': 'UTF-8',
    }
    if port:
        kwargs['port'] = port

    timeout = params.get('CONNECTION TIMEOUT') or params.get('TIMEOUT')
    if timeout:
        try:
            kwargs['timeout'] = int(timeout)
        except ValueError:
            pass

    login_timeout = params.get('LOGIN TIMEOUT')
    if login_timeout:
        try:
            kwargs['login_timeout'] = int(login_timeout)
        except ValueError:
            pass

    return kwargs


def create_connection_from_string(conn_str: str, autocommit: bool = False) -> pymssql.Connection:
    params = _parse_connection_string(conn_str)
    kwargs = _build_connection_kwargs(params, autocommit)
    return pymssql.connect(**kwargs)


@contextmanager
def get_connection(db: str = "target", autocommit: bool = False) -> Iterable[pymssql.Connection]:
    params = _parse_connection_string(_get_connection_string(db))
    try:
        connection = pymssql.connect(**_build_connection_kwargs(params, autocommit))
    except pymssql.Error as exc:
        message = getattr(exc, 'args', [str(exc)])
        raise RuntimeError(f"Failed to connect to SQL Server using '{db}' credentials: {message}") from exc

    try:
        yield connection
    finally:
        connection.close()


def read_sql(query: str, params: Optional[Sequence] = None, db: str = "target") -> pd.DataFrame:
    with get_connection(db=db) as conn:
        return pd.read_sql(query, conn, params=params)


def execute(query: str, params: Optional[Sequence] = None, db: str = "target") -> None:
    with get_connection(db=db, autocommit=False) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params) if params is not None else cursor.execute(query)
            conn.commit()
        finally:
            cursor.close()


def _qualify_table(table: str, schema: Optional[str]) -> str:
    if schema:
        return f"[{schema}].[{table}]"
    if "." in table:
        return table
    return f"[{DEFAULT_SCHEMA}].[{table}]"


def _prepare_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[Tuple]]:
    selected = df.loc[:, columns]
    sanitized = selected.replace({np.nan: None})
    sanitized = sanitized.where(pd.notnull(sanitized), None)
    return list(selected.columns), list(map(tuple, sanitized.to_numpy()))


def _executemany(cursor, sql: str, rows: List[Tuple]) -> None:
    if not rows:
        return
    if hasattr(cursor, "fast_executemany"):
        cursor.fast_executemany = True
    cursor.executemany(sql, rows)


def _delete_conflicts(
    cursor,
    qualified_table: str,
    key_columns: Sequence[str],
    rows: List[Tuple],
) -> None:
    if not key_columns or not rows:
        return

    placeholders = " AND ".join(f"[{col}] = %s" for col in key_columns)
    delete_sql = f"DELETE FROM {qualified_table} WHERE {placeholders}"
    _executemany(cursor, delete_sql, rows)


def _chunk_iterable(items: List[Tuple], chunk_size: int) -> Iterable[List[Tuple]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    *,
    key_columns: Sequence[str],
    batch_size: int = 1000,
    schema: Optional[str] = None,
    db: str = "target",
) -> int:
    if df.empty:
        return 0

    qualified_table = _qualify_table(table, schema)
    all_columns = list(df.columns)
    insert_columns, prepared_rows = _prepare_dataframe(df, all_columns)

    with get_connection(db=db, autocommit=False) as conn:
        cursor = conn.cursor()
        try:
            for chunk in _chunk_iterable(prepared_rows, batch_size):
                if key_columns:
                    key_indices = [insert_columns.index(col) for col in key_columns]
                    key_rows = [[row[idx] for idx in key_indices] for row in chunk]
                    _delete_conflicts(cursor, qualified_table, key_columns, key_rows)

                placeholders = ", ".join("%s" for _ in insert_columns)
                column_list = ", ".join(f"[{col}]" for col in insert_columns)
                insert_sql = f"INSERT INTO {qualified_table} ({column_list}) VALUES ({placeholders})"
                _executemany(cursor, insert_sql, chunk)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return len(prepared_rows)


def table_exists(table: str, schema: Optional[str] = None, db: str = "target") -> bool:
    qualified_table = _qualify_table(table, schema)
    if "." in qualified_table:
        schema_name, table_name = qualified_table.replace("[", "").replace("]", "").split(".")
    else:
        schema_name, table_name = DEFAULT_SCHEMA, qualified_table.strip('[]')

    query = (
        "SELECT COUNT(*) AS cnt FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
    )
    with get_connection(db=db) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (schema_name, table_name))
        result = cursor.fetchone()
        cursor.close()
    return bool(result and result[0])


def fetch_table(table: str, *, schema: Optional[str] = None, db: str = "target") -> pd.DataFrame:
    qualified_table = _qualify_table(table, schema)
    return read_sql(f"SELECT * FROM {qualified_table}", db=db)


__all__ = [
    "get_connection",
    "read_sql",
    "execute",
    "upsert_dataframe",
    "table_exists",
    "fetch_table",
    "create_connection_from_string",
]
