# logger.py — ThreadVision AI Data Logger
# ==========================================
# SQLite3 database for inspection records + CSV export + SPC statistics.
# Auto-creates the database and table on first run.
#
# Usage:
#   from logger import save_inspection, export_csv, get_stats

import os
import sqlite3
import datetime
import logging
import pandas as pd

from config import DB_PATH, CSV_PATH, LOG_DIR, THREAD_STANDARD

logger = logging.getLogger("ThreadVision.logger")


def _ensure_db():
    """
    Ensure the database and inspection table exist.

    Creates the logs/ directory, database file, and table if they
    don't already exist. Safe to call multiple times.

    Returns:
        sqlite3.Connection: Open connection to the inspection database.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            thread_standard TEXT    NOT NULL,
            major_diameter  REAL,
            minor_diameter  REAL,
            pitch           REAL,
            thread_depth    REAL,
            flank_angle     REAL,
            overall_result  TEXT    NOT NULL,
            major_status    TEXT,
            minor_status    TEXT,
            pitch_status    TEXT,
            depth_status    TEXT,
            angle_status    TEXT,
            notes           TEXT
        )
    """)
    conn.commit()

    return conn


def save_inspection(measurements, result, overall_str, notes=""):
    """
    Save an inspection record to the SQLite database.

    Args:
        measurements (dict): Output from measurement.measure().
        result (InspectionResult): Output from inspector.check().
        overall_str (str): 'PASS', 'FAIL', or 'UNMEASURED'.
        notes (str): Optional operator notes.

    Returns:
        int: The row ID of the saved record.
    """
    conn = _ensure_db()
    cursor = conn.cursor()

    per_dim = result.per_dimension if hasattr(result, 'per_dimension') else {}

    try:
        cursor.execute("""
            INSERT INTO inspections (
                timestamp, thread_standard,
                major_diameter, minor_diameter, pitch, thread_depth, flank_angle,
                overall_result,
                major_status, minor_status, pitch_status, depth_status, angle_status,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(),
            THREAD_STANDARD,
            measurements.get('major_diameter'),
            measurements.get('minor_diameter'),
            measurements.get('pitch'),
            measurements.get('thread_depth'),
            measurements.get('flank_angle'),
            overall_str,
            per_dim.get('major_diameter', {}).get('status', 'N/A'),
            per_dim.get('minor_diameter', {}).get('status', 'N/A'),
            per_dim.get('pitch', {}).get('status', 'N/A'),
            per_dim.get('thread_depth', {}).get('status', 'N/A'),
            per_dim.get('flank_angle', {}).get('status', 'N/A'),
            notes,
        ))
        conn.commit()
        row_id = cursor.lastrowid
        logger.info(f"Inspection #{row_id} saved: {overall_str}")
        return row_id
    except Exception as e:
        logger.error(f"Failed to save inspection: {e}")
        raise
    finally:
        conn.close()


def export_csv(output_path=None):
    """
    Export all inspection records to a CSV file.

    Uses pandas for clean, formatted CSV output with headers.

    Args:
        output_path (str): Path for the CSV file. Defaults to CSV_PATH from config.

    Returns:
        str: Path to the exported CSV file.
    """
    if output_path is None:
        output_path = CSV_PATH

    conn = _ensure_db()

    try:
        df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id", conn)

        if df.empty:
            logger.warning("No inspection records to export.")
            # Still create the file with headers
            df.to_csv(output_path, index=False)
            return output_path

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise
    finally:
        conn.close()


def get_stats(date=None):
    """
    Get inspection statistics for a specific date (default: today).

    Args:
        date (str): Date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        dict: Statistics dictionary with keys:
            - 'total':     int — total inspections for the day
            - 'pass':      int — number of PASS results
            - 'fail':      int — number of FAIL results
            - 'unmeasured': int — number of UNMEASURED results
            - 'pass_rate': float — pass rate as percentage (0–100)
            - 'date':      str — the date queried
    """
    if date is None:
        date = datetime.date.today().isoformat()

    conn = _ensure_db()

    try:
        cursor = conn.cursor()

        # Count by result type for the given date
        cursor.execute("""
            SELECT overall_result, COUNT(*) 
            FROM inspections 
            WHERE timestamp LIKE ?
            GROUP BY overall_result
        """, (f"{date}%",))

        rows = cursor.fetchall()
        counts = {row[0]: row[1] for row in rows}

        total = sum(counts.values())
        pass_count = counts.get('PASS', 0)
        fail_count = counts.get('FAIL', 0)
        unmeasured = counts.get('UNMEASURED', 0)
        pass_rate = (pass_count / total * 100) if total > 0 else 0.0

        stats = {
            'total':      total,
            'pass':       pass_count,
            'fail':       fail_count,
            'unmeasured': unmeasured,
            'pass_rate':  round(pass_rate, 1),
            'date':       date,
        }

        logger.info(
            f"Stats for {date}: {total} total, "
            f"{pass_count} pass, {fail_count} fail, {pass_rate:.1f}%"
        )
        return stats

    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        return {
            'total': 0, 'pass': 0, 'fail': 0,
            'unmeasured': 0, 'pass_rate': 0.0, 'date': date,
        }
    finally:
        conn.close()


def get_history(limit=50):
    """
    Get recent inspection records for the history view.

    Args:
        limit (int): Maximum number of records to return.

    Returns:
        list: List of dicts, each representing one inspection record.
    """
    conn = _ensure_db()

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, major_diameter, minor_diameter,
                   pitch, thread_depth, flank_angle,
                   overall_result
            FROM inspections 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))

        columns = [
            'id', 'timestamp', 'major_diameter', 'minor_diameter',
            'pitch', 'thread_depth', 'flank_angle', 'overall_result',
        ]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]

        logger.info(f"Retrieved {len(records)} history records.")
        return records
    except Exception as e:
        logger.error(f"History query failed: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Self-test: run `python logger.py` to verify database operations
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Logger Self-Test")
    print("=" * 60)

    # Simulated inspection data
    mock_measurements = {
        'major_diameter': 7.995,
        'minor_diameter': 6.552,
        'pitch':          1.249,
        'thread_depth':   0.644,
        'flank_angle':    60.1,
    }

    # Simulated result object
    class MockResult:
        per_dimension = {
            'major_diameter': {'status': 'PASS'},
            'minor_diameter': {'status': 'PASS'},
            'pitch':          {'status': 'PASS'},
            'thread_depth':   {'status': 'PASS'},
            'flank_angle':    {'status': 'PASS'},
        }

    # Test 1: Save an inspection
    print("\n--- Test 1: Save inspection ---")
    row_id = save_inspection(mock_measurements, MockResult(), 'PASS')
    print(f"  Saved as record #{row_id}")

    # Test 2: Save a failing inspection
    print("\n--- Test 2: Save failing inspection ---")
    mock_measurements['major_diameter'] = 7.900
    MockResult.per_dimension['major_diameter']['status'] = 'FAIL'
    row_id = save_inspection(mock_measurements, MockResult(), 'FAIL')
    print(f"  Saved as record #{row_id}")

    # Test 3: Get stats
    print("\n--- Test 3: Get today's stats ---")
    stats = get_stats()
    print(f"  Total: {stats['total']}, Pass: {stats['pass']}, "
          f"Fail: {stats['fail']}, Rate: {stats['pass_rate']}%")

    # Test 4: Export CSV
    print("\n--- Test 4: Export CSV ---")
    csv_path = export_csv()
    file_size = os.path.getsize(csv_path)
    print(f"  Exported to: {csv_path} ({file_size} bytes)")

    # Test 5: Get history
    print("\n--- Test 5: Get history ---")
    history = get_history(limit=10)
    print(f"  Records: {len(history)}")
    for rec in history:
        print(f"    #{rec['id']}: {rec['overall_result']} at {rec['timestamp']}")

    print(f"\n  Database: {DB_PATH}")
    print(f"  CSV file: {CSV_PATH}")
    print("\n✓ Logger self-test passed.")
