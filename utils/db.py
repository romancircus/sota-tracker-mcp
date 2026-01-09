"""Database utilities for SOTA Tracker."""

import sqlite3
from pathlib import Path
from typing import Union


def get_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Get a database connection with row factory.

    Use as context manager to auto-close:
        with get_db(path) as db:
            rows = db.execute(...).fetchall()

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection with Row factory enabled
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db
