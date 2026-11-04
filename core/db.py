# core/db.py
"""SQLite Database engine and persistence layer for DustGuard-Vision.
Tables:
- dust_events: id, timestamp, camera_id, opacity_level, misting_status, snapshot_filename
- water_savings: timestamp, baseline_usage_liters, dustguard_usage_liters, liters_saved
- daily_scorecard: date, site_id, hours_operated, high_dust_incidents, compliance_rating
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from constants import DB_PATH
except ImportError:
    from ..constants import DB_PATH


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables with the specified schemas."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dust_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        opacity_level REAL NOT NULL,
        misting_status TEXT NOT NULL,
        snapshot_filename TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS water_savings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        baseline_usage_liters REAL NOT NULL,
        dustguard_usage_liters REAL NOT NULL,
        liters_saved REAL NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_scorecard (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL UNIQUE,
        site_id TEXT NOT NULL,
        hours_operated REAL NOT NULL,
        high_dust_incidents INTEGER NOT NULL,
        compliance_rating TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def log_dust_event(
    camera_id: str,
    opacity_level: float,
    misting_status: str,
    snapshot_filename: Optional[str] = None,
    db_path: str = DB_PATH
) -> int:
    """Log an incident or snapshot event into dust_events table."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO dust_events (timestamp, camera_id, opacity_level, misting_status, snapshot_filename)
        VALUES (?, ?, ?, ?, ?)
    """, (now_str, camera_id, round(opacity_level, 2), misting_status, snapshot_filename))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def log_water_savings(
    baseline_usage: float,
    dustguard_usage: float,
    liters_saved: float,
    db_path: str = DB_PATH
) -> None:
    """Record a snapshot of water savings calculation."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO water_savings (timestamp, baseline_usage_liters, dustguard_usage_liters, liters_saved)
        VALUES (?, ?, ?, ?)
    """, (now_str, round(baseline_usage, 2), round(dustguard_usage, 2), round(liters_saved, 2)))

    conn.commit()
    conn.close()


def get_recent_events(limit: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent dust event violation records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dust_events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_summary_metrics(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculate summary statistics for dashboard display."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_violations FROM dust_events WHERE opacity_level >= 35.0")
    total_violations = cursor.fetchone()["total_violations"]

    cursor.execute("SELECT AVG(opacity_level) as avg_opacity FROM dust_events")
    avg_row = cursor.fetchone()["avg_opacity"]
    avg_opacity = round(avg_row, 1) if avg_row is not None else 0.0

    cursor.execute("SELECT liters_saved, baseline_usage_liters, dustguard_usage_liters FROM water_savings ORDER BY id DESC LIMIT 1")
    savings_row = cursor.fetchone()
    if savings_row:
        liters_saved = savings_row["liters_saved"]
        baseline = savings_row["baseline_usage_liters"]
        efficiency_pct = round((liters_saved / baseline * 100), 1) if baseline > 0 else 0.0
    else:
        liters_saved = 0.0
        efficiency_pct = 85.0  # Typical expected efficiency

    conn.close()

    # Determine compliance grade
    if total_violations < 5:
        rating = "A (Exemplary)"
    elif total_violations < 15:
        rating = "B (Compliant)"
    elif total_violations < 30:
        rating = "C (Marginal)"
    else:
        rating = "D (Action Required)"

    return {
        "total_violations": total_violations,
        "avg_opacity": avg_opacity,
        "liters_saved": round(liters_saved, 1),
        "efficiency_pct": efficiency_pct,
        "compliance_rating": rating
    }
