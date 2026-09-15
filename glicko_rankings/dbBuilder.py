"""
Builds Glicko ratings for every athlete in the dataset, keyed on athleteId.

Pipeline:
  1. Load the CSV into a `results` table in a SQLite database (schema.sql).
  2. Ask SQLite for the distinct races (venue, date, Round), ordered
     chronologically (ties broken by round stage: Heat < Quarterfinal <
     Semifinal < Final, since those can share the same date/venue).
  3. Walk the races in that order. For each race:
       - SQL: fetch the field (name, time-corrected, season/personal bests).
       - SQL: fetch each athlete's current rating/RD (or 1000/350 if new).
       - Python (rankingSystem.py): compute the Glicko update for every athlete
         in that race, using the other athletes in the SAME race as
         opponents, weighted by SeasonBest_Corrected/PersonalBest_Corrected.
       - SQL: upsert athlete_ratings and append to rating_history.

Usage:
    py build_ratings.py [--csv PATH] [--db PATH] [--option NAME] [--rebuild]
"""

import argparse
import csv
import sqlite3
from datetime import date, datetime
from pathlib import Path

# Novelty is a script to account for the first time a major performance is ran, e.g. Maurice Greene's 9.79
# is the first 9.79, so it is weighted more than a 9.79 today

# fix_mojibake is a script to fix names that were broken due to stuff like accent characters
# glicko_corrections is the script that does the glicko calculations
import fix_mojibake
import rankingSystem as glicko
import novelty

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "INSERT rankingData file here"
DEFAULT_DB = HERE / "INSERT output database here"
SCHEMA_PATH = HERE / "INSERT schema here"

CSV_COLUMNS = [
    "venue", "date", "athleteId", "name", "time-corrected",
    "SeasonBest_Corrected", "PersonalBest_Corrected", "Round",
    "PreviousPlacement", "PreviousTime-Corrected",
    "PerformanceMovingAverage_Time-Corrected", "PlacementAverage",
]


def optional_float(raw):
    """CSV cell -> float, or None for the blanks the v9 form columns carry."""
    if raw is None or raw.strip() == "" or raw.strip().upper() in {"NA", "NAN"}:
        return None
    return float(raw)

BATCH_SIZE = 5000

# Converts dates to ISO Format for later
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y")


def to_iso(raw):
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date format: {raw!r}")


def load_csv(conn, csv_path, limit=None):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM results")
    if cur.fetchone()[0] > 0:
        print("results table already populated, skipping CSV load")
        return

    insert_sql = """
        INSERT INTO results (venue, date, athleteId, name, time_corrected,
                              SeasonBest_Corrected, PersonalBest_Corrected, Round,
                              PreviousPlacement, PreviousTime_Corrected,
                              PerformanceMovingAverage_Time_Corrected, PlacementAverage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    total = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append((
                row["venue"],
                to_iso(row["date"]),
                row["athleteId"],
                # Names arrive double-encoded from the CSV; repair on the way in
                # so every database built from here is clean.
                fix_mojibake.repair(row["name"]),
                float(row["time-corrected"]),
                float(row["SeasonBest_Corrected"]),
                float(row["PersonalBest_Corrected"]),
                row["Round"],
                optional_float(row["PreviousPlacement"]),
                optional_float(row["PreviousTime-Corrected"]),
                optional_float(row["PerformanceMovingAverage_Time-Corrected"]),
                optional_float(row["PlacementAverage"]),
            ))
            if len(batch) >= BATCH_SIZE:
                cur.executemany(insert_sql, batch)
                total += len(batch)
                batch.clear()
            if limit and total + len(batch) >= limit:
                break
        if batch:
            cur.executemany(insert_sql, batch)
            total += len(batch)

    conn.commit()
    print(f"loaded {total} result rows into SQLite")


def fetch_race_order(conn):
    sql = """
        SELECT venue, date, Round
        FROM results
        GROUP BY venue, date, Round
        ORDER BY
            date ASC,
            CASE
                WHEN Round LIKE 'Heat%'         THEN 0
                WHEN Round LIKE 'Quarterfinal%' THEN 1
                WHEN Round LIKE 'Semifinal%'    THEN 2
                WHEN Round LIKE 'Final%'        THEN 3
                ELSE 4
            END ASC,
            venue ASC,
            Round ASC
    """
    return conn.execute(sql).fetchall()


def fetch_race_field(conn, venue, race_date, round_name):
    sql = """
        SELECT athleteId, name, time_corrected, SeasonBest_Corrected, PersonalBest_Corrected,
               PerformanceMovingAverage_Time_Corrected, PreviousPlacement,
               PlacementAverage, PreviousTime_Corrected,
               prior_faster_count, prior_total_count, prior_window_count
        FROM results
        WHERE venue = ? AND date = ? AND Round = ?
    """
    return conn.execute(sql, (venue, race_date, round_name)).fetchall()


def fetch_current_ratings(conn, athlete_ids):
    placeholders = ",".join("?" * len(athlete_ids))
    sql = f"""
        SELECT athlete_id, rating, rd, last_race_date
        FROM athlete_ratings
        WHERE athlete_id IN ({placeholders})
    """
    rows = conn.execute(sql, athlete_ids).fetchall()
    return {aid: (rating, rd, last_date) for aid, rating, rd, last_date in rows}


def days_between(earlier_iso, later_iso):
    if not earlier_iso:
        return None
    d1 = date.fromisoformat(earlier_iso)
    d2 = date.fromisoformat(later_iso)
    return (d2 - d1).days


def annotate_novelty(conn):
    """Fill results.prior_faster_count, walking races in rating order."""
    cur = conn.execute("SELECT COUNT(*) FROM results WHERE prior_faster_count IS NOT NULL")
    if cur.fetchone()[0] > 0:
        print("prior_faster_count already populated, skipping")
        return

    races = fetch_race_order(conn)
    fields = [
        conn.execute(
            "SELECT rowid, time_corrected FROM results "
            "WHERE venue = ? AND date = ? AND Round = ?", race
        ).fetchall()
        for race in races
    ]
    counts, totals, sizes = novelty.prior_faster_counts(
        [[t for _, t in f] for f in fields],
        [race[1] for race in races],
        glicko.NOVELTY_WINDOW_YEARS,
    )

    updates = []
    for field, field_counts, field_totals, field_sizes in zip(fields, counts, totals, sizes):
        for (rowid, _), count, total, size in zip(field, field_counts, field_totals, field_sizes):
            updates.append((count, total, size, rowid))
    conn.executemany(
        "UPDATE results SET prior_faster_count = ?, prior_total_count = ?, "
        "prior_window_count = ? WHERE rowid = ?",
        updates)
    conn.commit()
    print(f"annotated {len(updates)} rows with prior_faster_count")


def process_races(conn):
    # Running bests as of BEFORE the race currently being scored. Updated only
    # after a race has been processed, so the anchor can never see the mark it is
    # being asked to bound. personal_best is all-time; season_best is per calendar
    # year, matching how SeasonBest_Corrected is defined in the CSV.
    prior_pb = {}                # athlete_id -> best time_corrected so far
    prior_sb = {}                # (athlete_id, year) -> best time_corrected so far

    races = fetch_race_order(conn)
    print(f"{len(races)} distinct races to process")

    upsert_sql = """
        INSERT INTO athlete_ratings (athlete_id, name, rating, rd, last_race_date)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(athlete_id) DO UPDATE SET
            name = excluded.name,
            rating = excluded.rating,
            rd = excluded.rd,
            last_race_date = excluded.last_race_date
    """
    history_sql = """
        INSERT INTO rating_history (
            athlete_id, name, venue, date, round, time_corrected, opponent_count,
            opponent_quality_weight, self_consistency_weight,
            placement_surprise_multiplier, novelty_multiplier,
            rating_before, rd_before, rating_after, rd_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cur = conn.cursor()

    for i, (venue, race_date, round_name) in enumerate(races):
        field = fetch_race_field(conn, venue, race_date, round_name)
        athlete_ids = [row[0] for row in field]
        current = fetch_current_ratings(conn, athlete_ids)

        participants = []
        before_state = {}
        for (athlete_id, name, time_corrected, season_best, personal_best,
             form_mean, prev_placement, placement_avg, prev_time,
             prior_faster, prior_total, prior_window) in field:
            rating, rd, last_date = current.get(
                athlete_id, (glicko.INITIAL_RATING, glicko.INITIAL_RD, None)
            )
            idle_days = days_between(last_date, race_date)
            # Absence widens RD and, past the grace period, pulls the rating back
            # toward the baseline. Both happen before the race is scored, so
            # rating_history records the prior actually used.
            rating = glicko.apply_inactivity_decay(rating, idle_days)
            rd = glicko.apply_inactivity_growth(rd, idle_days)
            # Option 4 widens RD for an athlete whose form is visibly moving. Applied
            # here, next to the inactivity growth, so rating_history.rd_before records
            # the prior actually used for this race.
            rd = glicko.apply_trend_inflation(rd, prev_time, form_mean)
            before_state[athlete_id] = (rating, rd)
            participants.append({
                "athlete_id": athlete_id,
                "name": name,
                "rating": rating,
                "rd": rd,
                "time_corrected": time_corrected,
                "season_best_corrected": season_best,
                "personal_best_corrected": personal_best,
                "form_mean_corrected": form_mean,
                "previous_placement": prev_placement,
                "placement_average": placement_avg,
                "previous_time_corrected": prev_time,
                "prior_faster_count": prior_faster,
                "prior_total_count": prior_total,
                "prior_window_count": prior_window,
                # None on a debut / first race of a season; glicko_corrections
                # falls back to the row's own bests in that case.
                "prior_season_best_corrected": prior_sb.get((athlete_id, race_date[:4]),
                                                            prior_pb.get(athlete_id)),
                "prior_personal_best_corrected": prior_pb.get(athlete_id),
            })

        updates = glicko.update_ratings_for_race(participants)

        rating_rows = []
        history_rows = []
        for p in participants:
            athlete_id = p["athlete_id"]
            result = updates[athlete_id]
            rating_before, rd_before = before_state[athlete_id]
            rating_rows.append((athlete_id, p["name"], result["rating"], result["rd"], race_date))
            history_rows.append((
                athlete_id, p["name"], venue, race_date, round_name, p["time_corrected"],
                len(participants) - 1,
                result["opponent_quality_weight"], result["self_consistency_weight"],
                result["placement_surprise_multiplier"], result["novelty_multiplier"],
                rating_before, rd_before, result["rating"], result["rd"],
            ))

        # Roll the running bests forward only now that the race is scored.
        year = race_date[:4]
        for p in participants:
            aid, t = p["athlete_id"], p["time_corrected"]
            if aid not in prior_pb or t < prior_pb[aid]:
                prior_pb[aid] = t
            key = (aid, year)
            if key not in prior_sb or t < prior_sb[key]:
                prior_sb[key] = t

        cur.executemany(upsert_sql, rating_rows)
        cur.executemany(history_sql, history_rows)

        if (i + 1) % 2000 == 0:
            conn.commit()
            print(f"  processed {i + 1}/{len(races)} races")

    conn.commit()
    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--option", default="baseline",
                        choices=sorted(glicko.OPTIONS),
                        help="which recent-form weighting preset to apply")
    parser.add_argument("--rebuild", action="store_true", help="delete existing DB first")
    parser.add_argument("--limit", type=int, default=None, help="only load first N csv rows (testing)")
    args = parser.parse_args()

    settings = glicko.apply_option(args.option)
    print("option " + args.option + ": "
          + ", ".join(k + "=" + str(v) for k, v in settings.items()))

    db_path = Path(args.db)
    if args.rebuild and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())

    load_csv(conn, args.csv, limit=args.limit)
    annotate_novelty(conn)
    process_races(conn)

    top = conn.execute(
        "SELECT athlete_id, name, rating, rd FROM athlete_ratings "
        "ORDER BY rating DESC LIMIT 10"
    ).fetchall()
    print("\nTop 10 by rating:")
    for athlete_id, name, rating, rd in top:
        print(f"  {name:28s} {athlete_id:16s} rating={rating:8.2f}  rd={rd:6.2f}")

    conn.close()


if __name__ == "__main__":
    main()
