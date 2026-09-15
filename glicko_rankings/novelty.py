"""
Precedence counting for the novelty term.

The intent: the first time a mark is run it is evidence about the limits of the
event; the two-hundredth time it is evidence about an ordinary good day. Maurice
Greene's 9.79 in 1999 should move a rating more than a 9.79 in 2015.

1. TRAILING WINDOW. Precedents are counted only within the preceding
   `window_years`, so the count reflects the standard of the era rather than
   accumulating forever.

2. ERA-RELATIVE PRECEDENCE. Reporting the window's size lets the multiplier use a QUANTILE
  - the mark's rank as a
   fraction of what its era was running - which is density-invariant.

3. HISTORY RAMP. `prior_total_count` reports how many performances of any speed
   precede the race at all. glicko.novelty_multiplier ramps the bonus in with it,
   so the sparse opening years cannot collect a bonus for novelty where 
   the data is too thin to establish.

"""

from collections import deque
from datetime import date

# Quantization bounds. time_corrected spans roughly 9.62..11.57; the window is
# deliberately wider so corrections at the extremes cannot fall outside it.
MIN_TIME = 8.0
MAX_TIME = 14.0
STEP = 0.001

N_BINS = int(round((MAX_TIME - MIN_TIME) / STEP)) + 2

DEFAULT_WINDOW_YEARS = 8.0


def _bin(time_value):
    """Map a time to a 1-based Fenwick index, clamped to the tracked window."""
    idx = int(round((time_value - MIN_TIME) / STEP)) + 1
    return max(1, min(N_BINS - 1, idx))


class _Fenwick:
    __slots__ = ("tree",)

    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, idx, delta=1):
        tree = self.tree
        while idx < len(tree):
            tree[idx] += delta
            idx += idx & -idx

    def prefix(self, idx):
        """Number of live values in bins 1..idx (i.e. at least this fast)."""
        total = 0
        tree = self.tree
        while idx > 0:
            total += tree[idx]
            idx -= idx & -idx
        return total


def prior_faster_counts(races, dates, window_years=DEFAULT_WINDOW_YEARS):
    """
    races:  iterable of races in chronological order, each a sequence of
            time_corrected floats
    dates:  matching iterable of ISO 'YYYY-MM-DD' race dates

    Returns (windowed_counts, total_counts, window_sizes):
      windowed_counts - per row, performances at least this fast within the
                        preceding `window_years`
      total_counts    - per row, ALL performances of any speed that precede this
                        race (used by the history ramp)
      window_sizes    - per row, how many performances of ANY speed are live in
                        the window. 

    A race's own marks enter the tally only after that race has been counted, so
    a mark never counts itself or its heat-mates as precedent, and nothing from
    the future reaches the count.
    """
    tree = _Fenwick(N_BINS)
    live = deque()          # (ordinal, bin) still inside the window
    window_days = window_years * 365.2425

    windowed, totals, sizes = [], [], []
    running_total = 0

    for field, race_date in zip(races, dates):
        today = date.fromisoformat(race_date).toordinal()

        # Expire precedents that have fallen out of the trailing window.
        cutoff = today - window_days
        while live and live[0][0] < cutoff:
            _, old_bin = live.popleft()
            tree.add(old_bin, -1)

        bins = [_bin(t) for t in field]
        windowed.append([tree.prefix(b) for b in bins])
        totals.append([running_total] * len(bins))
        sizes.append([len(live)] * len(bins))

        for b in bins:
            tree.add(b, 1)
            live.append((today, b))
        running_total += len(bins)

    return windowed, totals, sizes
