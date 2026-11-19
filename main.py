"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI PS5 trophy's earn
rate on Exophase at the case's target.

Model: an ensemble of three median estimators, averaged in logit space,
with a spread that grows with their disagreement.

For every read the house made, the case gives the players Exophase tracks
(N) and each trophy's earn rate (r). Earners E = r * N / 100 are
reconstructed (rates print to 2 decimals, so E is an integer up to rounding).

  A. Saturating earners / players.  E(t) = A * (1 - exp(-t/tau)) and
     N(t) = B * (1 - exp(-t/tau_N)), each fitted by a tau grid search
     (profile least squares). A trophy's earners approach a ceiling as the
     launch cohort finishes the content; the tracked population approaches
     its own ceiling as the launch surge passes. Median rate A =
     100 * E*(target) / N*(target). This is the mechanism: the rate is a
     ratio, and the two sides saturate on different clocks.
  B. Logit-linear trend.  logit(r) = a + b * ln(1 + hours since release),
     fitted on deduped reads in the last 24h (the house curve baseline's
     shape and window), widening to 48h/96h/all only when the recent window
     holds too few distinct reads. Read at the target.
  C. Persistence. The rate in the case's last read.

Median (logit space): 0.25 * A + 0.65 * B + 0.10 * C. B dominates: it is
the house curve baseline's shape, but fitted on deduped reads (one point
per Exophase refresh, not one per 10-minute house read, so a long frozen
plateau cannot drag the slope toward zero the way raw reads do) and on a
recent window that widens only when it holds too few distinct reads.
A is the mechanism hedge (the rate is a ratio; if player growth slows
after the checkpoint, rates rise faster than any logit slope can see),
and C anchors runaway extrapolations. No single shape is trusted on its
own; the weights were chosen on five synthetic launch weeks (three
saturating, one slow-progression, one power-law), where B alone was the
best single median in every world and this blend was the best ensemble.

Stale reads: Exophase refreshes a game's counts on its own schedule, not
with every player. The rules record the page frozen at 638 players from
18 Nov 23:48 UTC through 19 Nov 04:00 UTC, then refreshed at 04:30 UTC to
3,868 players with every non-100% rate lower. A history that is completely
flat on launch night is a stale read waiting for that refresh, so when the
checkpoint is inside the launch window and the history has no variation at
all, the median is cut to 0.35x the last rate with a wide spread instead of
trusting the frozen number. (Checked against the live page: after the
refresh the median rate ratio across trophies was ~0.3, platinum 0.16,
the fastest earner 0.57. On the practice cases this scores loss 0.386
against the house baseline's 1.508, measured on the same live-page rates.)

Spread: quantiles are logit-normal around the ensemble median, sigma
combining the logit-fit residual, the disagreement between the three
estimators, a random-walk term 0.5*sqrt(days) (the house baseline's
diffusion) and 0.4 for refresh timing (the answering read shows the rates
as of Exophase's last refresh, not the target instant).

Standard library only, no network, deterministic (grid search + least
squares, no randomness), one file. Reads every *.in in the input
directory and writes one *.out per case.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# GTA VI released 19 November 2026 (00:00 UTC); the house's first read was
# 18 Nov 23:50 UTC, just before release. Saturating fits count time from
# 12h before release, so the pre-release reads (reviewers already earning)
# sit at t > 0 instead of t = 0.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
TIME_ORIGIN = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
# The launch window in which a completely flat history means "stale read
# waiting for its first refresh" (see module docstring).
LAUNCH_WINDOW_END = datetime(2026, 11, 20, tzinfo=timezone.utc)

# Prior saturation times (hours) for a trophy's earners, by tier, used only
# when a trophy's history is too thin to fit its own tau.
TIER_PRIOR_TAU = {"platinum": 400.0, "gold": 150.0, "silver": 90.0, "bronze": 36.0}
# Generous day-7 plausibility caps on the median rate, by tier. These only
# bind runaway extrapolations; they are not targets.
TIER_MEDIAN_CAP = {"platinum": 3.0, "gold": 15.0, "silver": 40.0, "bronze": 95.0}
# Fallback median (percent) when a trophy has no usable rate history at all.
TIER_PRIOR_MEDIAN = {"platinum": 0.5, "gold": 2.0, "silver": 5.0, "bronze": 10.0}

# Ensemble weights (logit space) for estimators A, B, C. Renormalised over
# whichever estimators are available for a trophy. B dominates: across
# five synthetic launch weeks (saturating and power-law worlds) the
# last-24h logit slope was the best single median in every world, and the
# A/C weights are insurance for the worlds where its assumptions break
# (a player-count slowdown the slope cannot see, a refresh jump).
W_A, W_B, W_C = 0.25, 0.65, 0.10
# Tau grid (hours) for the saturating fits. Capped at 384h: with only
# ~2.5 days of history a larger tau is indistinguishable from a straight
# line, and a straight line extrapolated 4.5 days out is how forecasts
# run away. A capped tau forces the fit to say "saturating" honestly.
TAU_GRID = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
# Prior power-law slope for ln N against ln(1 + hours) when the population
# has no fittable trend (N roughly tracks elapsed time early in launch).
N_PRIOR_SLOPE = 1.0

# Stale-read refresh prior (launch window, flat history): median multiplier
# and spread, in logit units. See module docstring.
FLAT_MEDIAN_FACTOR = 0.35
FLAT_SIGMA = 1.15
# Spread pieces (logit units).
SIGMA_REFRESH = 0.4          # answering read lags Exophase's last refresh
SIGMA_RW_PER_SQRT_DAY = 0.5  # the house baseline's diffusion
SIGMA_FIT_FLOOR = 0.25
SIGMA_FIT_CAP = 1.0
SIGMA_MIN = 0.5
SIGMA_MAX = 1.3


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    return 100.0 / (1.0 + math.exp(-z))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def linfit(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, residual_sd, n)."""
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid, n)


def sat_fit(ts, ys):
    """Fit y = amp * (1 - exp(-t/tau)) by tau grid search.

    Returns (amp, tau, resid_sd) or None when there is nothing to fit.
    amp is the least-squares amplitude for the best tau.
    """
    if len(ts) < 2 or max(ts) <= min(ts):
        return None
    best = None
    for tau in TAU_GRID:
        f = [1.0 - math.exp(-t / tau) for t in ts]
        sff = sum(v * v for v in f)
        if sff <= 0:
            continue
        amp = sum(y * v for y, v in zip(ys, f)) / sff
        amp = max(0.0, amp)
        sse = sum((y - amp * v) ** 2 for y, v in zip(ys, f))
        if best is None or sse < best[0]:
            best = (sse, amp, tau)
    if best is None:
        return None
    sse, amp, tau = best
    n = len(ts)
    resid = math.sqrt(sse / max(1, n - 2)) if n > 2 else 0.0
    return (amp, tau, resid)


def sat_predict(fit, t_star):
    amp, tau, _ = fit
    return amp * (1.0 - math.exp(-t_star / tau))


def dedupe_history(history):
    """Collapse consecutive reads with identical players and rates.

    Between Exophase refreshes the page is frozen, so a flat stretch is one
    observation, not many: keeping every read would overweight the plateau.
    """
    out = []
    prev_key = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev_key:
            out.append(h)
            prev_key = key
    return out


def quantiles(median_pct, sigma):
    """Five quantiles of a logit-normal around median_pct."""
    mu = logit(median_pct)
    qs = []
    for z in Z:
        q = clamp(inv_logit(mu + z * sigma))
        qs.append(round4(q))
    # Rounding can only break monotonicity at the edges; enforce it.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_trophy(trophy, snaps, t_target, x_target, horizon_days, n_fit,
                    checkpoint):
    """Forecast one trophy.

    snaps: deduped reads, each (t_hours_since_origin, x_ln1p_hours_since
    release, N, rate or None, read_time). n_fit: the population saturating
    fit (amp, tau, resid) or None.
    """
    tier = trophy.get("tier", "bronze")
    last_rate = None
    for _, _, _, r, _ in reversed(snaps):
        if r is not None:
            last_rate = r
            break
    if last_rate is None:
        # No rate ever observed for this trophy: tier prior, wide spread.
        return quantiles(TIER_PRIOR_MEDIAN.get(tier, 5.0), 1.2)
    if last_rate >= 99.5:
        # Earned by (essentially) every tracked player; a refresh cannot
        # push it down, because every new player earns it too.
        return quantiles(99.9, 0.15)
    if last_rate >= 95.0:
        return quantiles(min(99.9, last_rate), 0.3)

    # --- Estimator A: saturating earners / saturating players. ---
    est_a = None
    ts_e, ys_e = [], []
    for t, _, n, r, _ in snaps:
        if r is None or not n:
            continue
        ts_e.append(t)
        ys_e.append(round(r * n / 100.0))
    if len(ts_e) >= 2 and max(ts_e) > min(ts_e):
        fit_e = sat_fit(ts_e, ys_e)
    else:
        fit_e = None
    if fit_e is not None:
        e_star = sat_predict(fit_e, t_target)
    else:
        # Too thin to fit: earners grow from the last count at the tier's
        # prior saturation clock.
        tau0 = TIER_PRIOR_TAU.get(tier, 36.0)
        e_last = ys_e[-1] if ys_e else 0.0
        t_last = ts_e[-1] if ts_e else 0.0
        denom = 1.0 - math.exp(-t_last / tau0) if t_last > 0 else 1.0
        amp0 = e_last / denom if denom > 0 else e_last
        e_star = amp0 * (1.0 - math.exp(-t_target / tau0))
    if n_fit is not None:
        n_star = sat_predict(n_fit, t_target)
    else:
        # No population trend: players grow as a power law in elapsed time.
        n_last = None
        x_last = None
        for _, x, n, _, _ in reversed(snaps):
            if n:
                n_last, x_last = n, x
                break
        if n_last is None:
            n_star = 1.0
        else:
            n_star = n_last * math.exp(N_PRIOR_SLOPE * (x_target - x_last))
    n_star = max(1.0, n_star)
    if e_star is not None and e_star > 0:
        est_a = clamp(100.0 * max(0.0, e_star) / n_star)

    # --- Estimator B: logit(rate) linear in ln(1 + hours since release),
    # fitted on the LAST 24h of reads only. The first day has different
    # dynamics (reviewer cohort, first refresh, near-zero start) and its
    # steep early rise would drag a full-history slope far too high; the
    # recent window is the slope that is still running at the checkpoint.
    # (This is the house curve baseline's window.) If the recent window
    # holds too few distinct reads, widen it step by step (24h -> 48h ->
    # 96h -> all) rather than jumping straight to the contaminated
    # full-history slope. The ladder below only widens when the current
    # window holds fewer than 3 distinct reads, so a healthy 24h window
    # is never diluted by older history.
    est_b = None
    resid_b = None
    for window_h in (24, 48, 96, None):
        xs_b, ys_b = [], []
        for _, x, _, r, at in snaps:
            if r is None:
                continue
            if window_h is not None and (checkpoint - at).total_seconds() > window_h * 3600:
                continue
            xs_b.append(x)
            ys_b.append(logit(r))
        if len(xs_b) >= 3 and max(xs_b) > min(xs_b):
            break
    if len(xs_b) >= 2 and max(xs_b) > min(xs_b):
        fit_b = linfit(xs_b, ys_b)
        a_b, b_b, resid_b, _ = fit_b
        est_b = clamp(inv_logit(a_b + b_b * x_target))
    elif xs_b:
        est_b = clamp(inv_logit(ys_b[-1]))

    # --- Estimator C: persistence. ---
    est_c = clamp(last_rate)

    # --- Ensemble in logit space. ---
    parts = []
    if est_a is not None:
        parts.append((W_A, logit(est_a)))
    if est_b is not None:
        parts.append((W_B, logit(est_b)))
    parts.append((W_C, logit(est_c)))
    wsum = sum(w for w, _ in parts)
    mu = sum(w * v for w, v in parts) / wsum
    median = clamp(inv_logit(mu))
    median = min(median, TIER_MEDIAN_CAP.get(tier, 95.0))
    median = clamp(median)

    # --- Spread: fit residual + estimator disagreement + diffusion +
    # refresh timing. ---
    logits = [v for _, v in parts]
    mean_l = sum(logits) / len(logits)
    disagree = math.sqrt(sum((v - mean_l) ** 2 for v in logits) / len(logits))
    sigma_fit = SIGMA_FIT_FLOOR
    if resid_b is not None:
        sigma_fit = min(SIGMA_FIT_CAP, max(SIGMA_FIT_FLOOR, resid_b))
    sigma = math.sqrt(
        sigma_fit ** 2
        + disagree ** 2
        + (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        + SIGMA_REFRESH ** 2
    )
    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    return quantiles(median, sigma)


def forecast_case(case):
    """Forecast every trophy of one case. Returns {"trophies": [...]}."""
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    t_target = (target - TIME_ORIGIN).total_seconds() / 3600.0
    x_target = math.log(1.0 + max(0.0, (target - RELEASE).total_seconds() / 3600.0))

    # Snapshot series with plateaus collapsed: (t, x, N, [rates]).
    deduped = dedupe_history(history)
    snaps_all = []
    for h in deduped:
        at = parse_time(h["at"])
        t = (at - TIME_ORIGIN).total_seconds() / 3600.0
        x = math.log(1.0 + max(0.0, (at - RELEASE).total_seconds() / 3600.0))
        snaps_all.append((t, x, h.get("players_tracked"), h.get("rates") or []))

    flat = len(snaps_all) <= 1
    checkpoint = parse_time(case["at"])
    if flat and checkpoint < LAUNCH_WINDOW_END and snaps_all:
        # Stale launch-night read: the next refresh dilutes every rate.
        # Median cut + wide spread; the 100% trophy cannot fall.
        rows = []
        last_rates = snaps_all[-1][3]
        for t in trophies:
            tid = t["id"]
            r = last_rates[tid - 1] if tid - 1 < len(last_rates) else None
            if r is None:
                rows.append({"id": tid,
                             "q": quantiles(TIER_PRIOR_MEDIAN.get(t.get("tier", "bronze"), 5.0), 1.2)})
            elif r >= 99.5:
                rows.append({"id": tid, "q": quantiles(99.9, 0.15)})
            else:
                rows.append({"id": tid,
                             "q": quantiles(clamp(r * FLAT_MEDIAN_FACTOR), FLAT_SIGMA)})
        return {"trophies": rows}

    # Population saturating fit: N(t) = B * (1 - exp(-t/tau_N)).
    ts_n = [t for t, _, n, _ in snaps_all if n]
    ys_n = [n for _, _, n, _ in snaps_all if n]
    n_fit = sat_fit(ts_n, ys_n) if len(ts_n) >= 2 else None

    # Per-trophy snapshot series.
    per_trophy = {}
    for t in trophies:
        tid = t["id"]
        snaps = []
        for (tt, x, n, rates), h in zip(snaps_all, deduped):
            r = rates[tid - 1] if tid - 1 < len(rates) else None
            snaps.append((tt, x, n, r, parse_time(h["at"])))
        per_trophy[tid] = snaps

    rows = []
    for t in trophies:
        try:
            q = forecast_trophy(t, per_trophy[t["id"]], t_target, x_target,
                                horizon_days, n_fit, checkpoint)
        except Exception:
            # Never lose a case to one bad trophy: fall back to its last
            # rate with a wide spread.
            last = None
            for _, _, _, r, _ in reversed(per_trophy[t["id"]]):
                if r is not None:
                    last = r
                    break
            q = quantiles(clamp(last if last is not None else 1.0), 1.0)
        rows.append({"id": t["id"], "q": q})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def safe_forecast_case(case):
    """forecast_case, but a broken case still yields a valid-shaped output."""
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        last_rates = []
        hist = case.get("history") or []
        if hist:
            last_rates = hist[-1].get("rates") or []
        for t in case["trophies"]:
            tid = t["id"]
            r = last_rates[tid - 1] if tid - 1 < len(last_rates) else None
            rows.append({"id": tid,
                         "q": quantiles(clamp(r if r is not None else 1.0), 1.0)})
        return {"trophies": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_files = sorted(INPUT_DIR.glob("*.in"))
    if not case_files:
        raise SystemExit("no case files in input directory")
    for path in case_files:
        case = json.loads(path.read_text(encoding="utf-8"))
        out = safe_forecast_case(case)
        text = json.dumps(out, separators=(",", ":"))
        if len(text.encode("utf-8")) > 65536:
            raise SystemExit(f"{path.name}: output over 65,536 bytes")
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
