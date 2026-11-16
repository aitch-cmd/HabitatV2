import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("habitat_db")

# Connection Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "habitatos")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)


def get_connection_string() -> str:
    """Return database connection URI."""
    return DATABASE_URL


@contextmanager
def get_db_connection():
    """
    Context manager that yields a psycopg connection configured with dict_row.
    Automatically commits on success or rollbacks on exception.
    """
    conn = None
    try:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"PostgreSQL error: {e}")
        raise
    finally:
        if conn and not conn.closed:
            conn.close()


@contextmanager
def get_db_cursor():
    """
    Context manager that yields a dictionary-based cursor.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            yield cur


class PostgresClient:
    """
    PostgresClient handles interactions with the HabitatOS PostgreSQL database.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.db_url = database_url or DATABASE_URL

    def check_connection(self) -> bool:
        """Check if database server is reachable."""
        try:
            with psycopg.connect(self.db_url, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return True
        except Exception as e:
            logger.warning(f"PostgreSQL connection check failed: {e}")
            return False

    def fetch_all(self, query: str, params: Optional[tuple | dict] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return all matching rows as dictionaries."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.fetchall()

    def fetch_one(self, query: str, params: Optional[tuple | dict] = None) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return a single row as a dictionary."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.fetchone()

    def execute(self, query: str, params: Optional[tuple | dict] = None) -> int:
        """Execute an INSERT, UPDATE, or DELETE query and return affected row count."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.rowcount

    def execute_and_return_id(self, query: str, params: Optional[tuple | dict] = None) -> Optional[int]:
        """Execute an INSERT query with RETURNING id and return the generated ID."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                row = cur.fetchone()
                return row["id"] if row and "id" in row else None


# Module-level client instance
postgres_client = PostgresClient()
