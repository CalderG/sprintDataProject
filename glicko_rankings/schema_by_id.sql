-- Glicko rating system schema (SQLite)
--
-- `results` holds the raw race data imported from rankingData_v9.csv.
-- `athlete_ratings` holds each athlete's current (most recent) rating/RD,
-- keyed on athleteId so that two people sharing a name stay separate. The
-- name column is carried for display only.
-- `rating_history` is an append-only audit log of every rating update,
-- one row per athlete per race, so ratings over time can be reconstructed.

CREATE TABLE IF NOT EXISTS results (
    result                  REAL,
    country                 TEXT,
    pos                     TEXT,
    venue                   TEXT    NOT NULL,
    venueCountry            TEXT,
    date                    TEXT    NOT NULL,   -- ISO 'YYYY-MM-DD'
    competitionId           TEXT,
    athleteId               TEXT,
    name                    TEXT    NOT NULL,
    dateOfBirth             TEXT,
    yearOfBirth             INTEGER,
    wind                    REAL,
    competition             TEXT,
    height                  REAL,
    weight                  REAL,
    altitude                REAL,
    time_corrected          REAL    NOT NULL,
    PersonalBest            REAL,
    yearOfResult            INTEGER,
    SeasonBest              REAL,
    ageDuringResult         REAL,
    monthOfResult           INTEGER,
    PersonalBest_Corrected  REAL    NOT NULL,
    SeasonBest_Corrected    REAL    NOT NULL,
    dayOfResult             INTEGER,
    Round                   TEXT    NOT NULL,
    -- v9 recent-form columns; NULL on an athlete's first race, and the moving
    -- average is also NULL on each season's first race.
    PreviousPlacement                       REAL,
    PreviousTime_Corrected                  REAL,
    PerformanceMovingAverage_Time_Corrected REAL,
    PlacementAverage                        REAL,
    -- performances at least this fast inside the trailing window (novelty.py)
    prior_faster_count                      INTEGER,
    -- performances of any speed preceding this race, for the novelty ramp
    prior_total_count                       INTEGER,
    -- performances of any speed live in the trailing window, the era-quantile
    -- denominator
    prior_window_count                      INTEGER
);

-- Every race is uniquely identified by (venue, date, Round); this index
-- makes grouping/fetching one race's field fast.
CREATE INDEX IF NOT EXISTS idx_results_race ON results(venue, date, Round);
CREATE INDEX IF NOT EXISTS idx_results_athlete ON results(athleteId);

CREATE TABLE IF NOT EXISTS athlete_ratings (
    athlete_id      TEXT PRIMARY KEY,
    name            TEXT,               -- display only; NOT the identity
    rating          REAL    NOT NULL DEFAULT 1000.0,
    rd              REAL    NOT NULL DEFAULT 350.0,
    last_race_date  TEXT
);

CREATE TABLE IF NOT EXISTS rating_history (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    athlete_id              TEXT    NOT NULL,
    name                    TEXT,
    venue                   TEXT    NOT NULL,
    date                    TEXT    NOT NULL,
    round                   TEXT    NOT NULL,
    time_corrected          REAL,
    opponent_count          INTEGER,
    opponent_quality_weight REAL,   -- this athlete's own weight when acting as an opponent
    self_consistency_weight REAL,   -- this athlete's dampener for this specific result
    placement_surprise_multiplier REAL, -- option 3 delta multiplier (1.0 when disabled)
    novelty_multiplier      REAL,   -- precedence discount (1.0 when disabled)
    rating_before           REAL,
    rd_before               REAL,
    rating_after            REAL,
    rd_after                REAL
);

CREATE INDEX IF NOT EXISTS idx_history_athlete ON rating_history(athlete_id);
CREATE INDEX IF NOT EXISTS idx_history_race ON rating_history(venue, date, round);
