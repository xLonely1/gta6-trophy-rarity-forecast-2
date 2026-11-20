"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Python 3.13, standard library only, no network, deterministic. Reads every
<id>.in in the input directory and writes <id>.out: for each of the case's
55 trophies, five quantiles (0.05/0.25/0.5/0.75/0.95) of its Exophase earn
rate (percent) at the case's target.

How it forecasts, per trophy, in logit space:

1. The history is deduplicated first. Exophase refreshes its counts on its
   own schedule, so the page can sit frozen for hours: consecutive reads
   showing the same players-tracked count and the same rates collapse to a
   single observation. A frozen plateau must never count as many
   independent observations, or it drags every trend to zero.

2. If a trophy's deduplicated history is flat (at most two distinct rates),
   the page has not refreshed inside the window. The next refresh dilutes
   every rate: the tracked-player count jumps while earners grow slowly, so
   the median is a fraction of the last rate, with a wide spread.

3. Otherwise three estimators are blended in logit space, the trend
   carrying the largest weight:
     A. Saturating curves fitted to the trophy's earners and to the tracked
        players against hours since release; rate = earners* / players*.
        Both curves are amp * (1 - exp(-t / tau)), tau picked by grid
        search, amplitude by least squares.
     B. A logit-linear trend of logit(rate) against ln(1 + hours since
        release), fitted on deduplicated reads from the last 24 hours
        before the checkpoint. When fewer than three distinct points fall
        in that window it widens step by step (48h, 96h, whole history)
        instead of jumping straight to a slope the early steep phase would
        poison. The slope is clamped to a sane range.
     C. Persistence: the rate in the last read, the anchor when both fits
        misbehave.

4. The median is capped per tier (platinum 3, gold 15, silver 40,
   bronze 95). A trophy already at 99.5% or more is forecast near 99.9
   with a narrow spread. Quantiles come from a logit-normal spread that
   grows with the horizon, with fit residuals and with how far the three
   estimators disagree.

Release anchor: 2026-11-19T00:00:00Z (the game's launch day).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# shipped beside this file would still import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Blend weights in logit space; the recent trend carries the most.
W_SATURATING = 0.30
W_TREND = 0.55
W_PERSIST = 0.15

# Flat-history fallback: the next refresh dilutes the rate to a fraction
# of the last read (measured on the launch-night refresh), wide spread.
FLAT_FACTOR = 0.20
FLAT_SIGMA = 0.75

# Median caps per tier, in percent.
TIER_CAPS = {"platinum": 3.0, "gold": 15.0, "silver": 40.0, "bronze": 95.0}
# Last-resort medians when a trophy has no usable read at all.
TIER_PRIORS = {"platinum": 0.5, "gold": 1.0, "silver": 3.0, "bronze": 10.0}

# Saturating-curve grid for the time constant, in hours.
TAU_GRID = (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0)
# Trend windows, in hours before the checkpoint, widened step by step.
TREND_WINDOWS = (24.0, 48.0, 96.0, None)
MAX_SLOPE = 3.0
MIN_POINTS = 3


def parse_time(text):
    """ISO-8601 timestamp (with Z) to an aware datetime; naive means UTC."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(moment):
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p == 0.5:
        return 0.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def dedupe_reads(history):
    """Collapse consecutive reads showing the identical page to one."""
    out = []
    seen = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != seen:
            out.append(read)
            seen = key
    return out


def fit_saturating(times, values):
    """Least-squares amp*(1-exp(-t/tau)) over a tau grid.

    Returns (amplitude, tau) for the best tau, or None when the data
    cannot support a two-parameter fit.
    """
    pts = [(t, y) for t, y in zip(times, values) if y is not None]
    if len(pts) < MIN_POINTS:
        return None
    best = None
    for tau in TAU_GRID:
        f = [1.0 - math.exp(-t / tau) for t, _ in pts]
        denom = sum(v * v for v in f)
        if denom <= 0.0:
            continue
        amp = sum(y * fv for (_, y), fv in zip(pts, f)) / denom
        if amp <= 0.0:
            continue
        sse = sum((y - amp * fv) ** 2 for (_, y), fv in zip(pts, f))
        if best is None or sse < best[0]:
            best = (sse, amp, tau)
    if best is None:
        return None
    return best[1], best[2]


def fit_trend(points):
    """Least-squares line of logit(rate) on ln(1 + hours since release).

    points: list of (hours, rate). Returns (intercept, slope, rmse) or None.
    """
    pts = [(math.log1p(max(0.0, t)), logit(r)) for t, r in points if r is not None]
    if len(pts) < MIN_POINTS:
        return None
    n = float(len(pts))
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
    intercept = (sy - slope * sx) / n
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in pts) / n)
    return intercept, slope, rmse


def refresh_factor(series):
    """Dilution factor for the next refresh, from the last observed one.

    series: [(hours, rate, players_tracked), ...] in time order, already
    deduplicated. When the page refreshed once inside the window, the
    ratio of the two distinct rates is direct evidence of how a refresh
    dilutes the rate; shrink it toward the launch-night prior
    (FLAT_FACTOR) with a geometric mean so one noisy refresh cannot
    dominate. When the page never refreshed, the prior stands.
    """
    vals = []
    for _, r, _ in series:
        if not vals or round(r, 6) != vals[-1]:
            vals.append(round(r, 6))
    if len(vals) < 2:
        return FLAT_FACTOR
    first, last = vals[0], vals[-1]
    if first <= 0:
        return FLAT_FACTOR
    ratio = last / first
    if not (0.05 <= ratio <= 0.8):
        return FLAT_FACTOR
    return math.sqrt(ratio * FLAT_FACTOR)


def forecast_trophy(idx, trophy, deduped, checkpoint, target, horizon_days):
    """Five quantiles for one trophy. Returns a list of five numbers."""
    tier = trophy.get("tier") or "bronze"
    cap = TIER_CAPS.get(tier, 95.0)

    series = []  # (hours since release, rate, players_tracked)
    for read in deduped:
        rates = read.get("rates") or []
        rate = rates[idx] if idx < len(rates) else None
        if rate is None:
            continue
        when = read.get("at") or read.get("slot")
        series.append((hours_since_release(parse_time(when)), rate,
                       read.get("players_tracked")))

    if not series:
        med = TIER_PRIORS.get(tier, 10.0)
        return quantiles_from(med, 1.5)

    last_rate = series[-1][1]
    if last_rate >= 99.5:
        return quantiles_from(99.9, 0.15)

    distinct = {round(r, 6) for _, r, _ in series}
    if len(distinct) <= 2:
        # Frozen page: the next refresh dilutes the rate. When exactly
        # one refresh was seen, its ratio informs the factor. The spread
        # widens with the horizon, like every other spread here: a
        # 90-minute practice horizon stays near FLAT_SIGMA, a week-long
        # final horizon is honestly wider.
        sigma_flat = math.sqrt(FLAT_SIGMA ** 2 + (0.5 * math.sqrt(max(0.0, horizon_days))) ** 2)
        sigma_flat = min(1.3, max(0.75, sigma_flat))
        return quantiles_from(refresh_factor(series) * last_rate, sigma_flat)

    target_h = hours_since_release(target)
    estimates = {}  # name -> (logit, weight)

    # A: saturating curves for earners and for tracked players.
    times = [t for t, _, _ in series]
    earners = [(r / 100.0 * n) if n else None for _, r, n in series]
    players = [n for _, _, n in series]
    fit_e = fit_saturating(times, earners)
    fit_n = fit_saturating(times, players)
    if fit_e is not None and fit_n is not None:
        amp_e, tau_e = fit_e
        amp_n, tau_n = fit_n
        e_star = amp_e * (1.0 - math.exp(-target_h / tau_e))
        n_star = amp_n * (1.0 - math.exp(-target_h / tau_n))
        if e_star > 0.0 and n_star > 0.0:
            estimates["A"] = (logit(100.0 * e_star / n_star), W_SATURATING)

    # B: logit-linear trend on the last 24h (widening when points are few).
    fit_sigma = 0.5
    trend = None
    checkpoint_h = hours_since_release(checkpoint)
    for window in TREND_WINDOWS:
        pts = [(t, r) for t, r, _ in series
               if window is None or (checkpoint_h - t) <= window]
        trend = fit_trend(pts)
        if trend is not None:
            break
    if trend is not None:
        intercept, slope, rmse = trend
        fit_sigma = min(1.0, max(0.25, rmse))
        pred = intercept + slope * math.log1p(target_h)
        estimates["B"] = (pred, W_TREND)

    # C: persistence, the anchor.
    estimates["C"] = (logit(last_rate), W_PERSIST)

    total_w = sum(w for _, w in estimates.values())
    med_logit = sum(lg * w for lg, w in estimates.values()) / total_w
    med = min(cap, 100.0 * sigmoid(med_logit))

    logits = [lg for lg, _ in estimates.values()]
    mean_lg = sum(logits) / len(logits)
    disagree = math.sqrt(sum((x - mean_lg) ** 2 for x in logits) / len(logits))
    disagree = min(0.8, disagree)
    horizon_term = 0.5 * math.sqrt(max(0.0, horizon_days))
    sigma = math.sqrt(fit_sigma ** 2 + disagree ** 2 + horizon_term ** 2 + 0.4 ** 2)
    sigma = min(1.3, max(0.5, sigma))
    return quantiles_from(med, sigma)


def quantiles_from(median_pct, sigma):
    lg = logit(median_pct)
    out = []
    prev = None
    for level in (0.05, 0.25, 0.5, 0.75, 0.95):
        q = 100.0 * sigmoid(lg + norm_ppf(level) * sigma)
        q = clamp_pct(round4(q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case, name):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    deduped = dedupe_reads(history)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        qs = forecast_trophy(i, trophy, deduped, checkpoint, target, horizon_days)
        if len(qs) != len(levels):  # pragma: no cover - levels are fixed
            raise ValueError(f"{name}: wrong quantile count")
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
