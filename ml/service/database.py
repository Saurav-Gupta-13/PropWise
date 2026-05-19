"""
PropWise — Database Layer (SQLite)
====================================
Stores predictions + user feedback for continuous learning.

Schema:
  - predictions: every prediction made (input features + output)
  - feedback: user-submitted actual prices (used for retraining)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "propwise.db"


@contextmanager
def get_db():
    """Context manager for SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_json TEXT NOT NULL,
            predicted_price INTEGER NOT NULL,
            price_low INTEGER NOT NULL,
            price_high INTEGER NOT NULL,
            location TEXT NOT NULL,
            bhk INTEGER NOT NULL,
            area_sqft REAL NOT NULL,
            model_version TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prediction_id INTEGER,
            actual_price INTEGER,
            rating INTEGER,
            comment TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_location ON predictions(location);
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
        """)
    print(f"✅ Database ready at {DB_PATH}")


def log_prediction(input_dict, predicted_price, price_low, price_high, model_version="v0.1"):
    """Log a prediction to the database. Returns prediction ID."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions
            (timestamp, input_json, predicted_price, price_low, price_high,
             location, bhk, area_sqft, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            json.dumps(input_dict),
            predicted_price,
            price_low,
            price_high,
            input_dict.get("location", "unknown"),
            input_dict.get("bhk", 0),
            input_dict.get("area_sqft", 0),
            model_version,
        ))
        return cursor.lastrowid


def log_feedback(prediction_id, actual_price=None, rating=None, comment=None):
    """User feedback on a prediction."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO feedback (timestamp, prediction_id, actual_price, rating, comment)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            prediction_id,
            actual_price,
            rating,
            comment,
        ))


def get_stats():
    """Get usage statistics for the dashboard."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as n FROM predictions").fetchone()["n"]
        feedback_count = conn.execute("SELECT COUNT(*) as n FROM feedback").fetchone()["n"]
        feedback_with_price = conn.execute(
            "SELECT COUNT(*) as n FROM feedback WHERE actual_price IS NOT NULL"
        ).fetchone()["n"]

        top_locations = conn.execute("""
            SELECT location, COUNT(*) as count
            FROM predictions
            GROUP BY location
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

        return {
            "total_predictions": total,
            "total_feedback": feedback_count,
            "feedback_with_actual_price": feedback_with_price,
            "top_locations": [dict(r) for r in top_locations],
        }


def get_training_supplements():
    """
    Get user-submitted actual prices for retraining the model.
    Joins predictions with feedback where actual_price is provided.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT p.input_json, f.actual_price
            FROM predictions p
            JOIN feedback f ON f.prediction_id = p.id
            WHERE f.actual_price IS NOT NULL
        """).fetchall()
        return [{"input": json.loads(r["input_json"]), "price": r["actual_price"]} for r in rows]


if __name__ == "__main__":
    init_db()
    print("Stats:", get_stats())
