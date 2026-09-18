# Three recent developments are each aimed at a failure mode found by inspecting individual
# rating moves rather than by any aggregate metric:
#
# 1. RATING_DELTA_CAP - a hard bound on how far one race can move a rating.
#
# 2. Anchor computed from PRIOR bests. apply_time_anchor previously read
#    SeasonBest_Corrected / PersonalBest_Corrected straight from the row, and
#    those INCLUDE the race being scored. A breakthrough therefore raised its own
#    ceiling in the same update it was being bounded by. 
#    The driver now tracks each athlete's bests as of BEFORE the
#    race and passes them in.
#
# 3. apply_inactivity_decay - ratings drift back toward the baseline after an
#    absence longer than INACTIVITY_GRACE_DAYS. Previously only RD responded to
#    inactivity; the rating itself was preserved indefinitely.
# ---------------------------------------------------------------------------

"""

Coded with Claude (AI)

Potential Changes:
    - Try to incorporate  winstreaks into the rating update
    - Try to incorporate a peaking ability feature into the ratings
Changelog: 
    - Modified SB_Weight, PB_Weight, and Scale constants to 0.7, 0.3, and 0.15 respectively
    - Normalized weighted_score_sum by the total opponent weight (sum of w_j*g_j),
      making it a weighted mean bounded to [-1, 1] instead of a field-size-scaled sum;
      the divisor is clamped at 1.0 so small fields are never amplified
    - Replaced the ordinal win/loss score with margin_score(), a logistic on the
      time gap, so the size of a winning margin reaches the rating update
    - Added a one-sided absolute-time anchor: ratings are shrunk toward the level
      an athlete's own season/personal bests imply, bounding drift inside closed pools
    - Added a novelty multiplier to weight the first occurrences of given performances higher than later attempts,
    with sparseness of data accounted for
    - Anchors computed from prior bests so that PBs or SBs are not considered expected just because the columns
    were constructed that way
    - Added inactivity decay so sprinters lose ratings for not racing for longer than 182 days, except in the case of 
    the offseason typically going from the end of September to March/April

Glicko-1 rating math for the sprint ranking system, extended with two
explicit weighting terms tied to each athlete's SeasonBest_Corrected and
PersonalBest_Corrected columns:

1. Opponent-quality weight (w_j): when athlete i races athlete j, j's
   pull on i's rating (in both the information sum and the mean-shift
   sum of the standard Glicko formula) is scaled by how fast j's own
   season/personal bests are. Racing (and beating, or losing closely
   to) athletes with fast recent/lifetime bests moves a rating more
   than racing athletes with slow bests.

2. Self-consistency weight (c_i): after the ordinary Glicko update is
   computed for athlete i, the applied rating/RD change is scaled down
   the further i's actual time-corrected in this race falls short of
   i's own expected time (derived from i's own season/personal bests).
   A result at or better than an athlete's known level is trusted
   fully; a result well off that level (bad day, tactical race,
   etc.) is treated as weaker evidence and only partially applied.

"""

import math

# --- Core Glicko-1 constants -------------------------------------------------

Q = math.log(10) / 400.0

INITIAL_RATING = 1000.0
INITIAL_RD = 350.0
MIN_RD = 30.0
MAX_RD = 350.0

# Hard bound on |rating_after - rating_before| for a single race, applied last of
# all - after the multipliers and after the anchor - so it is a guarantee rather
# than a tendency. Glicko has no such bound natively: the update is limited only
# by RD, and an athlete returning from a long layoff has RD at its 350 maximum
# precisely when the model knows least about them. Set to None to disable.
RATING_DELTA_CAP = 300.0

# RD grows back toward MAX_RD while an athlete is inactive. RD_INACTIVITY_C is
# chosen so that an athlete who hasn't raced in exactly one year (365 days)
# has their RD grow all the way from MIN_RD back to MAX_RD.
RD_INACTIVITY_C = math.sqrt((MAX_RD ** 2 - MIN_RD ** 2) / 365.0)

# --- Opponent-quality weighting (uses SeasonBest_Corrected / PersonalBest_Corrected
#     of the OPPONENT) ---------------------------------------------------------

QUALITY_SB_WEIGHT = 0.7          # season best reflects current form -> weighted higher
QUALITY_PB_WEIGHT = 0.3          # personal best reflects lifetime ceiling
QUALITY_REFERENCE_TIME = 10.00   # implied time (seconds) that maps to weight == 1.0
QUALITY_SCALE = 0.15              # seconds; smaller = weight reacts more sharply, may need to decrease
QUALITY_MIN_WEIGHT = 0.5
QUALITY_MAX_WEIGHT = 2.0

# --- Self-consistency weighting (uses SeasonBest_Corrected / PersonalBest_Corrected
#     of the ATHLETE BEING UPDATED) -------------------------------------------

CONSISTENCY_SB_WEIGHT = 0.7
CONSISTENCY_PB_WEIGHT = 0.3
CONSISTENCY_SCALE = 0.15         # seconds; smaller = harsher penalty for off performances

# --- Margin-sensitive scoring -------------------------------------------------
# How many seconds of winning margin it takes for the head-to-head score to move
# meaningfully away from 0.5. Smaller = closer to the old ordinal win/loss step;
# larger = margins matter more but every result carries less weight.
MARGIN_SCALE = 0.10              # seconds

# --- Absolute-time anchor -----------------------------------------------------
# Everything else in this file is relative: s_ij compares two athletes in the same
# race, so a rating is only ever defined against the field it raced. Inside a
# partially closed pool (a domestic circuit whose members mostly race each other)
# nothing ties that pool back to the global time scale, and an athlete can
# accumulate rating indefinitely by beating locally-rated opponents.
#
# The anchor maps an athlete's own season/personal bests onto the rating scale and
# pulls a fraction of the way toward it each race. ANCHOR_ONE_SIDED makes it a
# CEILING rather than a magnet: it only bites when a rating has climbed above what
# the athlete's times justify, so it corrects upward drift without dragging down
# athletes who are already rated at or below their time-implied level.
#
# ANCHOR_POINTS_PER_SECOND is the empirical slope of rating against implied time in
# the competitive 10.0-10.5s band; ANCHOR_REFERENCE_RATING is the observed median
# rating at the 10.00s reference. Set ANCHOR_WEIGHT to 0.0 to disable entirely.
ANCHOR_WEIGHT = 0.35             # fraction of the gap closed per race
ANCHOR_REFERENCE_TIME = 10.00    # seconds
ANCHOR_REFERENCE_RATING = 1450.0 # rating an athlete implying 10.00s anchors to
ANCHOR_POINTS_PER_SECOND = 1300.0
ANCHOR_TOLERANCE = 750.0         # headroom above the anchor before the ceiling bites
ANCHOR_ONE_SIDED = True          # True = ceiling only; False = symmetric shrinkage

# Four per-row columns describe an athlete's immediate history BEFORE the race
# being scored, so none of them leak the current result:
#
#   PreviousPlacement                        - finishing position last race
#   PreviousTime-Corrected                   - time-corrected mark last race
#   PerformanceMovingAverage_Time-Corrected  - mean of the last <=3 marks (resets
#                                              each calendar year)
#   PlacementAverage                         - mean of the last <=3 placements
#
# All four columns are missing on an athlete's first race (and the moving average
# is missing on each season's first race), so every function here
# falls back to the season/personal-best behaviour when its input is None.


# OPTION parameters:

# The expected time c_i measures against is currently 0.7*SB + 0.3*PB. Season best
# is an extreme-value statistic - the single fastest run of the year - so a typical
# race looks "off" against it and c is depressed almost everywhere. The trailing
# 3-race mean is a central estimate of what the athlete has actually been running,
# so blending it in makes c reflect a bad day rather than a non-peak day.
FORM_CONSISTENCY_WEIGHT = 1.00 # 0 = bests only, 1 = moving average only
FORM_QUALITY_WEIGHT = 1.00 # 0 = bests only, 1 = moving average only
# PlacementAverage says where this athlete usually finishes; their position in this
# race says where they actually finished. Beating that expectation is stronger
# evidence than confirming it, so the rating delta is scaled by a bounded
# multiplier. tanh keeps one freak result from exploding the update.
SURPRISE_WEIGHT = 0.60 # multiplier reaches 1 +/- this at large surprise
SURPRISE_SCALE = 2.0 # places of surprise for tanh to reach ~0.76
# The gap between the last race and the 3-race mean says whether an athlete is
# moving. A sprinter mid-breakthrough has a genuinely less certain current ability
# than their RD implies, and RD is exactly the parameter for "we are unsure", so
# inflate it and let the update move them faster.
TREND_SCALE = 0.15 # seconds of trend that saturates the inflation
TREND_RD_WEIGHT = 0.10 # max fractional RD inflation
# Every term above is relative or form-based; none of them care whether a mark is
# historically remarkable. A mark nobody had run before is stronger evidence about
# an athlete than the two-hundredth running of the same time.
#
# Novelty is measured by novelty.py as prior_faster_count: performances at least
# this fast inside a TRAILING WINDOW (default 8 years), not all-time. Two all-time
# formulations were built and rejected first, both because an all-time count
# measures position in the dataset rather than novelty - see novelty.py for the
# measurements. The trailing window makes the count track the standard of the era.
#
# NOVELTY_RAMP closes the remaining hole. Even windowed, the opening years of the
# data are too sparse for "unprecedented" to mean anything - 84 rows precede 1961
# and 9,223 precede 1995 - so every early mark looks novel. The bonus is ramped in
# with prior_total_count, the number of performances of any speed preceding the
# race, and is therefore near-zero until real history exists.
#
# Windowing alone still left a gradient, because racing volume grows by three
# orders of magnitude: with an absolute precedent count the mean bonus was 1.15 in
# the 1970s against 1.001 in the 2020s. The default "relative" mode therefore
# scores the mark's QUANTILE within its window - its rank as a fraction of what the
# era was actually running - which is invariant to how much racing there was.
#
#     relative: multiplier = 1 + w * ramp * exp(-(count / window_size) / NOVELTY_QUANTILE)
#     bonus:    multiplier = 1 + w * ramp * exp(-count / NOVELTY_SCALE)
#     ramp    = 1 - exp(-prior_total_count / NOVELTY_RAMP)
#
# "bonus" (absolute count) and "penalty" are retained so the rejected results stay
# reproducible.
# Set NOVELTY_WEIGHT to 0.0 to disable (the default).
NOVELTY_WEIGHT = 0.50
NOVELTY_QUANTILE = 0.02 # era-quantile at which a bonus decays by 1/e
NOVELTY_WINDOW_YEARS = 8.0
NOVELTY_RAMP = 2000.0 # performances of history before the bonus is fully on

# In dictionary format
OPTIONS = {
    # corrected - option12's tuned weights with NOVELTY_WEIGHT reduced to 0.5.
    #             Used with RATING_DELTA_CAP and the prior-bests anchor above.
    "corrected": {"FORM_CONSISTENCY_WEIGHT": 1.00, "FORM_QUALITY_WEIGHT": 1.00,
                  "SURPRISE_WEIGHT": 0.60, "TREND_RD_WEIGHT": 0.10,
                  "NOVELTY_WEIGHT": 0.50, "NOVELTY_MODE": "relative",
                  "NOVELTY_QUANTILE": 0.02, "NOVELTY_WINDOW_YEARS": 8.0,
                  "NOVELTY_RAMP": 2000.0},

}

def _blend(best_implied_time, form_time, weight):
    """Mix a bests-derived implied time with the trailing-form mean."""
    if form_time is None or weight <= 0.0:
        return best_implied_time
    return (1.0 - weight) * best_implied_time + weight * form_time


def placement_surprise_multiplier(actual_placement, placement_average, previous_placement):
    """
    Scale the rating delta by how far this finish beat the athlete's usual one.

    Lower placement numbers are better, so (expected - actual) is positive when the
    athlete outperformed. Falls back to last race's placement when the 3-race
    average is missing, and to a neutral 1.0 when both are.
    """
    if SURPRISE_WEIGHT <= 0.0:
        return 1.0
    expected = placement_average if placement_average is not None else previous_placement
    if expected is None or actual_placement is None:
        return 1.0
    surprise = expected - actual_placement
    return max(0.0, 1.0 + SURPRISE_WEIGHT * math.tanh(surprise / SURPRISE_SCALE))


def apply_trend_inflation(rd, previous_time_corrected, form_mean_corrected):
    """
    Widen RD when the athlete's last race sits far from their 3-race mean.

    Direction is deliberately ignored: a sharp improvement and a sharp decline are
    both evidence that current ability is unsettled.
    """
    if TREND_RD_WEIGHT <= 0.0:
        return rd
    if previous_time_corrected is None or form_mean_corrected is None:
        return rd
    trend = abs(form_mean_corrected - previous_time_corrected)
    inflated = rd * (1.0 + TREND_RD_WEIGHT * math.tanh(trend / TREND_SCALE))
    return min(MAX_RD, inflated)


def novelty_multiplier(prior_faster_count, prior_total_count=None, window_size=None):
    """Weight a result by how unprecedented the mark was for its era."""
    if NOVELTY_WEIGHT <= 0.0 or prior_faster_count is None:
        return 1.0

    ramp = 1.0
    if prior_total_count is not None and NOVELTY_RAMP > 0.0:
        ramp = 1.0 - math.exp(-prior_total_count / NOVELTY_RAMP)
    if not window_size:
            # No era to compare against yet; the ramp is already suppressing this.
        return 1.0 + NOVELTY_WEIGHT * ramp
    quantile = prior_faster_count / window_size
    decay = math.exp(-quantile / NOVELTY_QUANTILE)
    return 1.0 + NOVELTY_WEIGHT * ramp * decay

def apply_option(name):
    """Set the module-level weights for a named preset. Returns the settings used."""
    if name not in OPTIONS:
        raise KeyError(f"unknown option {name!r}; choose from {sorted(OPTIONS)}")
    defaults = {"FORM_CONSISTENCY_WEIGHT": 0.0, "FORM_QUALITY_WEIGHT": 0.0,
                "SURPRISE_WEIGHT": 0.0, "TREND_RD_WEIGHT": 0.0,
                "NOVELTY_WEIGHT": 0.0, 
                "NOVELTY_SCALE": 25.0, "NOVELTY_RAMP": 2000.0,
                "NOVELTY_WINDOW_YEARS": 8.0, "NOVELTY_QUANTILE": 0.002}
    settings = {**defaults, **OPTIONS[name]}
    for key, value in settings.items():
        globals()[key] = value
    return settings


# --- Inactivity rating decay --------------------------------------------------
# RD already grows while an athlete is idle (apply_inactivity_growth), but that
# only widens the uncertainty around a rating - it never moves the rating itself.
# A sprinter who has not raced in three years keeps their peak number intact and
# re-enters the field carrying it, which is why athlete_ratings is populated with
# stale peaks from athletes long since retired.
#
# This decays the rating itself once an absence passes a grace period. Form in a
# 100m is not preserved across a layoff, so the rating should drift back toward
# the population baseline rather than sit where it was.
#
#     idle   = days_since_last_race - INACTIVITY_GRACE_DAYS
#     factor = 0.5 ** (idle / RATING_DECAY_HALFLIFE_DAYS)
#     rating = FLOOR + (rating - FLOOR) * factor
#
# Properties chosen deliberately:
#   - Nothing happens inside the grace period, so a normal off-season (a European
#     sprinter racing September then April, ~200 days) is barely touched.
#   - Exponential in the EXCESS over the floor, so decay is proportional: a 2400
#     rating loses far more absolute rating than an 1100 one, and nobody is
#     dragged below the baseline by absence alone.
#   - One-sided. A rating already at or below the floor is left untouched; time
#     away is not evidence of improvement.
#
# Applied by the driver at the athlete's NEXT race, where the elapsed time is
# known, so it is recorded in rating_history.rating_before. Note this means a
# retired athlete's stored rating in athlete_ratings is not decayed - the decay
# materializes only when they race again.
INACTIVITY_GRACE_DAYS = 182.0      # ~6 months; no decay before this
RATING_DECAY_HALFLIFE_DAYS = 730.0  # excess over the floor halves per 2 years idle
RATING_DECAY_FLOOR = INITIAL_RATING  # 1000.0; the level absence decays toward


def apply_inactivity_decay(rating, days_since_last_race):
    """Pull a rating back toward RATING_DECAY_FLOOR after a long layoff."""
    if days_since_last_race is None:
        return rating
    idle = days_since_last_race - INACTIVITY_GRACE_DAYS
    if idle <= 0 or rating <= RATING_DECAY_FLOOR:
        return rating
    factor = 0.5 ** (idle / RATING_DECAY_HALFLIFE_DAYS)
    return RATING_DECAY_FLOOR + (rating - RATING_DECAY_FLOOR) * factor


def g(rd):
    """Glicko's g(RD): shrinks an opponent's effective influence as their RD grows."""
    return 1.0 / math.sqrt(1.0 + 3.0 * Q ** 2 * rd ** 2 / math.pi ** 2)


def expected_score(rating, opponent_rating, opponent_rd):
    """Probability that `rating` beats `opponent_rating`, given the opponent's RD."""
    return 1.0 / (1.0 + 10 ** (-g(opponent_rd) * (rating - opponent_rating) / 400.0))


def margin_score(time_i, time_j):
    """
    Graded head-to-head score for athlete i against athlete j, in (0, 1).

    Standard Glicko uses an ordinal score in {1.0, 0.0, 0.5}, which discards the
    size of the gap: edging someone by 0.01s scores exactly the same as beating
    them by 0.40s. time_corrected is the only absolute measure of speed in the
    data, so that collapse is what lets a rating drift away from the time scale
    inside a closed pool of similarly-fast athletes - an athlete can accumulate
    rating indefinitely by narrowly winning races among slow opponents.

    A logistic on the time gap restores the margin while preserving the two
    properties the Glicko update depends on:

        margin_score(t, t) == 0.5                       (a dead heat is a draw)
        margin_score(a, b) + margin_score(b, a) == 1    (antisymmetry)

    Lower time wins, so the gap is (time_j - time_i): when i is faster the
    exponent is positive and the score exceeds 0.5.
    """
    # Times are ~9-12s so the exponent stays small, but clamp anyway: at +/-40
    # the score is already within 5e-18 of a clean win/loss.
    x = max(-40.0, min(40.0, (time_j - time_i) / MARGIN_SCALE))
    return 1.0 / (1.0 + math.exp(-x))


def opponent_quality_weight(season_best_corrected, personal_best_corrected,
                            form_mean_corrected=None):
    """
    How much racing THIS athlete (as an opponent) should count for the other
    competitors in the race. Faster implied ability -> weight > 2 (up to
    QUALITY_MAX_WEIGHT); slower -> weight < 2 (down to QUALITY_MIN_WEIGHT).
    """
    implied_time = _blend(
        QUALITY_SB_WEIGHT * season_best_corrected
        + QUALITY_PB_WEIGHT * personal_best_corrected,
        form_mean_corrected, FORM_QUALITY_WEIGHT,
    )
    raw_weight = math.exp((QUALITY_REFERENCE_TIME - implied_time) / QUALITY_SCALE)
    return min(QUALITY_MAX_WEIGHT, max(QUALITY_MIN_WEIGHT, raw_weight))


def self_consistency_weight(time_corrected, season_best_corrected, personal_best_corrected,
                            form_mean_corrected=None):
    """
    How much confidence to place in THIS specific result for the athlete who
    ran it. Running at or better than your own expected time -> weight == 1.0.
    Running slower than expected -> weight decays toward 0.
    """
    expected_time = _blend(
        CONSISTENCY_SB_WEIGHT * season_best_corrected
        + CONSISTENCY_PB_WEIGHT * personal_best_corrected,
        form_mean_corrected, FORM_CONSISTENCY_WEIGHT,
    )
    shortfall = max(0.0, time_corrected - expected_time)
    return math.exp(-shortfall / CONSISTENCY_SCALE)


def time_anchor_rating(season_best_corrected, personal_best_corrected):
    """The rating an athlete's own bests imply, on the global time scale."""
    implied_time = (
        CONSISTENCY_SB_WEIGHT * season_best_corrected
        + CONSISTENCY_PB_WEIGHT * personal_best_corrected
    )
    return ANCHOR_REFERENCE_RATING + (
        (ANCHOR_REFERENCE_TIME - implied_time) * ANCHOR_POINTS_PER_SECOND
    )


def apply_time_anchor(rating, season_best_corrected, personal_best_corrected):
    """
    Shrink `rating` toward the level its owner's times justify.

    With ANCHOR_ONE_SIDED this is a soft ceiling: a rating at or below
    anchor + ANCHOR_TOLERANCE is returned untouched, so the anchor only ever
    removes upward drift and never inflates a rating toward the anchor.

    The bests passed in MUST exclude the race being scored - see
    update_ratings_for_race, which prefers the driver-supplied prior bests. If the
    race's own mark is allowed in, a breakthrough run raises the very ceiling that
    is supposed to bound it.
    """
    if ANCHOR_WEIGHT <= 0.0:
        return rating
    ceiling = time_anchor_rating(season_best_corrected, personal_best_corrected) + ANCHOR_TOLERANCE
    if ANCHOR_ONE_SIDED and rating <= ceiling:
        return rating
    return rating + ANCHOR_WEIGHT * (ceiling - rating)


def apply_inactivity_growth(rd, days_since_last_race):
    """RD drifts back up toward MAX_RD the longer an athlete has been idle."""
    if days_since_last_race is None or days_since_last_race <= 0:
        return rd
    grown = math.sqrt(rd ** 2 + (RD_INACTIVITY_C ** 2) * days_since_last_race)
    return min(MAX_RD, grown)


def update_ratings_for_race(participants):
    """
    Compute updated (rating, rd) for every athlete in a single race.

    `participants` is a list of dicts, each with:
        athlete_id              - athlete key (str); the unique identity
        name                    - athlete display name (str), not used as a key
        rating, rd              - current Glicko rating/RD (already
                                   inactivity-adjusted for this race's date)
        time_corrected          - this race's time-corrected value (lower wins)
        season_best_corrected   - athlete's SeasonBest_Corrected
        personal_best_corrected - athlete's PersonalBest_Corrected
        form_mean_corrected     - PerformanceMovingAverage_Time-Corrected, or None
        previous_placement      - PreviousPlacement, or None
        placement_average       - PlacementAverage, or None
        previous_time_corrected - PreviousTime-Corrected, or None
        prior_faster_count      - performances at least this fast inside the
                                  trailing window (see novelty.py), or None
        prior_total_count       - performances of any speed preceding this race,
                                  used by the novelty history ramp, or None
        prior_window_count      - performances of any speed live in the trailing
                                  window, the denominator of the era quantile
        prior_season_best_corrected     - the athlete's season best BEFORE this
                                  race, or None on the first race of a season
        prior_personal_best_corrected   - the athlete's personal best BEFORE this
                                  race, or None on a debut

    Returns a dict: athlete_id -> {
        "rating": new_rating, "rd": new_rd,
        "opponent_quality_weight": w,   # this athlete's own weight as an opponent
        "self_consistency_weight": c,   # dampener applied to this athlete's update
    }
    """
    # Precompute each participant's own opponent-quality weight once, since
    # it only depends on their bests and gets reused by every other participant.
    for p in participants:
        p["_quality_weight"] = opponent_quality_weight(
            p["season_best_corrected"], p["personal_best_corrected"],
            p.get("form_mean_corrected"),
        )

    # Finishing position in THIS race, needed for the placement-surprise term.
    # Ranked on time_corrected so it agrees with the margin_score ordering; ties
    # share the better position, as they do in the results themselves.
    ordered = sorted(participants, key=lambda p: p["time_corrected"])
    placements = {}
    for rank, p in enumerate(ordered, start=1):
        if rank > 1 and p["time_corrected"] == ordered[rank - 2]["time_corrected"]:
            placements[p["athlete_id"]] = placements[ordered[rank - 2]["athlete_id"]]
        else:
            placements[p["athlete_id"]] = rank

    results = {}
    for i, pi in enumerate(participants):
        opponents = participants[:i] + participants[i + 1:]

        c = self_consistency_weight(
            pi["time_corrected"], pi["season_best_corrected"], pi["personal_best_corrected"],
            pi.get("form_mean_corrected"),
        )
        m = placement_surprise_multiplier(
            placements[pi["athlete_id"]], pi.get("placement_average"),
            pi.get("previous_placement")
        )
        nov = novelty_multiplier(pi.get("prior_faster_count"),
                                 pi.get("prior_total_count"),
                                 pi.get("prior_window_count"))
        # if there are no other ppl in the race (aka 1 man race)
        # then instantiate a results object for the given participant
        if not opponents:
            results[pi["athlete_id"]] = {
                "rating": pi["rating"],
                "rd": pi["rd"],
                "opponent_quality_weight": pi["_quality_weight"],
                "self_consistency_weight": c,
                "placement_surprise_multiplier": 1.0,
                "novelty_multiplier": nov,
            }
            continue

        d2_denominator = 0.0
        weighted_score_sum = 0.0
        total_weight = 0.0
        for pj in opponents:
            w = pj["_quality_weight"]
            gj = g(pj["rd"])
            e_ij = expected_score(pi["rating"], pj["rating"], pj["rd"])

            s_ij = margin_score(pi["time_corrected"], pj["time_corrected"])
            # think of this as a recurrence relation of the previous denominator plus
            # the product of the opponent's weight and their reduced influence on rating deviation squared
            # times the bernoulli trial of the probability that runner "i" wins over the opponent
            d2_denominator += (w * gj) ** 2 * e_ij * (1.0 - e_ij)
            weighted_score_sum += w * gj * (s_ij - e_ij)
            total_weight += w * gj

        # Normalize the score sum into a WEIGHTED MEAN over the field.
        #
        # Unnormalized, weighted_score_sum grows linearly with the number of
        # opponents while d2_denominator collapses toward 0 whenever the rating
        # gap is extreme (E_ij -> 0 makes E_ij*(1 - E_ij) -> 0). The two move in
        # opposite directions, so a stale low-rated athlete who beats a deep,
        # much stronger field gets an unbounded mean-shift: that pairing produced
        # the four largest single-race gains in the dataset (+2400 to +2930).
        #
        # Dividing by the same sum of w_j*g_j that builds the numerator turns the
        # term into an average of (s_ij - e_ij), bounded to [-1, 1] regardless of
        # field size. One dominant upset can still move a rating, but 30 of them
        # can no longer stack additively.
        #
        # The divisor is clamped at 1.0 so normalization can only ever SHRINK the
        # update. In a small field the sum of w_j*g_j is well below 1 (a lone
        # opponent at the 0.5 weight floor with a high RD contributes only about
        # 0.335), and dividing by that would amplify head-to-head races instead of
        # damping them - the opposite of the intent, and ~37% of races in the data
        # have exactly one opponent.
        if total_weight > 1.0:
            weighted_score_sum /= total_weight

        if d2_denominator <= 0.0:
            # Every expected score was exactly 0 or 1 (extreme rating gap) -
            # fall back to leaving the rating/RD unchanged for this race.
            new_rating, new_rd = pi["rating"], pi["rd"]
        else:
            d2 = 1.0 / (Q ** 2 * d2_denominator)
            precision = (1.0 / pi["rd"] ** 2) + (1.0 / d2)

            full_new_rating = pi["rating"] + (Q / precision) * weighted_score_sum
            full_new_rd = math.sqrt(1.0 / precision)

            # Self-consistency blends between "no update" (c=0) and the full
            # Glicko update (c=1) based on how expected this result was.
            # c scales the update by how trustworthy this result is, m by how
            # surprising the finish was, nov by how unprecedented the mark was. RD is
            # deliberately left on c alone - neither a surprising nor a historically
            # novel result is by itself a reason to claim more precision.
            new_rating = pi["rating"] + c * m * nov * (full_new_rating - pi["rating"])
            new_rd = pi["rd"] - c * (pi["rd"] - full_new_rd)
            new_rd = max(MIN_RD, min(MAX_RD, new_rd))

        # Absolute-time anchor: stop the rating drifting above what this
        # athlete's own bests justify. Only applied when the athlete actually
        # raced someone - a solo run is not evidence either way.
        #
        # Use the bests as they stood BEFORE this race when the driver supplies
        # them, so this result cannot lift its own ceiling. Falls back to the row's
        # bests on a true debut, where no prior mark exists.
        anchor_sb = pi.get("prior_season_best_corrected") or pi["season_best_corrected"]
        anchor_pb = pi.get("prior_personal_best_corrected") or pi["personal_best_corrected"]
        new_rating = apply_time_anchor(new_rating, anchor_sb, anchor_pb)

        # Hard delta cap, applied last so nothing downstream can reopen it.
        if RATING_DELTA_CAP is not None:
            low = pi["rating"] - RATING_DELTA_CAP
            high = pi["rating"] + RATING_DELTA_CAP
            new_rating = max(low, min(high, new_rating))

        results[pi["athlete_id"]] = {
            "rating": new_rating,
            "rd": new_rd,
            "opponent_quality_weight": pi["_quality_weight"],
            "self_consistency_weight": c,
            "placement_surprise_multiplier": m,
            "novelty_multiplier": nov,
        }

    return results
