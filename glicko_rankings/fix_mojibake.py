"""
Repair double-encoded athlete names.
"""

import sqlite3
import sys

# Characters that only appear in text that has been through the bad round trip.
MARKERS = ("Ã", "Å", "Â", "â€", "Ä", "Ð", "Ñ", "Ø", "Þ")

# Tables and columns holding athlete names.
NAME_COLUMNS = [
    ("results", "name"),
    ("athlete_ratings", "name"),
    ("rating_history", "name"),
]

def repair(text):
    """Undo one latin-1/UTF-8 round trip; return the input unchanged on failure."""
    if not text or not any(marker in text for marker in MARKERS):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def repair_database(db_path):
    """
    Rewrite every corrupted name in place.

    """
    conn = sqlite3.connect(db_path)
    total = 0
    for table, column in NAME_COLUMNS:
        try:
            rows = conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()
        except sqlite3.OperationalError:
            continue                      # table absent in this schema variant

        updates = [(v, repair(v)) for (v,) in rows if v and repair(v) != v]
        if not updates:
            print(f"  {table}.{column}: nothing to repair")
            continue

        conn.execute("DROP TABLE IF EXISTS _name_fix")
        conn.execute("CREATE TEMP TABLE _name_fix (bad TEXT PRIMARY KEY, good TEXT)")
        conn.executemany("INSERT OR REPLACE INTO _name_fix VALUES (?, ?)", updates)
        conn.execute(f"""
            UPDATE {table}
            SET {column} = (SELECT good FROM _name_fix WHERE bad = {table}.{column})
            WHERE {column} IN (SELECT bad FROM _name_fix)
        """)
        conn.execute("DROP TABLE _name_fix")
        total += len(updates)
        print(f"  {table}.{column}: {len(updates)} distinct names repaired")
    conn.commit()
    conn.close()
    return total

if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        print(__doc__)
        raise SystemExit(1)
    for db in targets:
        print(db)
        n = repair_database(db)
        print(f"  -> {n} name values rewritten\n")
