"""Self-exciting (Hawkes) burst layer.

Why a Hawkes process: Elon doesn't tweet as independent Poisson draws. When he starts, one
post tends to trigger a cluster ("batch") within minutes-to-hours. A Hawkes process models
exactly this -- each event temporarily raises the intensity of further events:

    lambda(t) = mu(t) + sum_{t_j < t} alpha * beta * exp(-beta * (t - t_j))

  * mu(t)            : seasonal background = mu_scale * g(t), with g the day/hour shape (mean 1).
  * alpha            : branching ratio = expected directly-triggered offspring per post (0<alpha<1).
                       This is what produces the over-dispersion (bursts) the data shows.
  * beta             : decay rate (1/beta = characteristic burst timescale in hours).

We fit (mu_scale, alpha, beta) by EM (exponential-kernel Hawkes) on recent history. For
forecasting we keep the *burst structure* (alpha, beta) and re-tie the level to the
bootstrapped weekly level from ``intensity.py``, so the volatile "how much overall" is handled
by the level distribution and the stable "how clustered" by the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HawkesParams:
    mu_scale: float   # background scale (tweets/hour at g==1) from the fit window
    alpha: float      # branching ratio (0<alpha<1)
    beta: float       # decay rate (per hour); 1/beta = burst timescale
    loglik: float
    n_events: int

    @property
    def burst_timescale_h(self) -> float:
        return 1.0 / self.beta if self.beta > 0 else float("nan")


def fit_hawkes(
    event_hours: np.ndarray,   # event times in hours, ascending, within [0, T]
    T: float,                  # horizon in hours
    g_event: np.ndarray,       # seasonal background weight g(t_i) at each event (mean ~1)
    g_integral: float,         # integral of g over [0, T]
    max_iter: int = 200,
    tol: float = 1e-6,
    beta0: float = 1.0,
    alpha0: float = 0.5,
) -> HawkesParams:
    """EM fit of an exponential-kernel Hawkes process with seasonal background g."""
    t = np.asarray(event_hours, dtype=float)
    n = len(t)
    if n < 20:  # not enough to fit a kernel; degenerate to near-Poisson
        mu = n / max(g_integral, 1e-9)
        return HawkesParams(mu, 1e-3, beta0, float("nan"), n)

    g = np.asarray(g_event, dtype=float)
    dt = np.diff(t, prepend=t[0])  # dt[0]=0
    alpha, beta = float(alpha0), float(beta0)
    mu_scale = max(0.5 * n / max(g_integral, 1e-9), 1e-9)
    prev_ll = -np.inf

    for _ in range(max_iter):
        # --- O(N) recursions for R = sum e^{-b dt}, R2 = sum dt e^{-b dt} ---
        R = np.zeros(n)
        R2 = np.zeros(n)
        for i in range(1, n):
            e = np.exp(-beta * dt[i])
            R[i] = e * (1.0 + R[i - 1])
            R2[i] = e * (R2[i - 1] + dt[i] * (R[i - 1] + 1.0))
        mu_i = mu_scale * g
        trig_i = alpha * beta * R                      # triggered intensity at each event
        lam = mu_i + trig_i + 1e-12

        # E-step responsibilities (aggregated)
        p_bg = mu_i / lam
        sum_bg = p_bg.sum()
        A = float((trig_i / lam).sum())                # expected # triggered events
        num_beta = float((alpha * beta * R2 / lam).sum())

        # M-step
        decay_tail = 1.0 - np.exp(-beta * (T - t))     # offspring horizon truncation per parent
        denom_alpha = decay_tail.sum()
        new_alpha = np.clip(A / max(denom_alpha, 1e-9), 1e-4, 0.95)
        new_beta = float(np.clip(A / max(num_beta, 1e-9), 1e-3, 50.0))
        new_mu = max(sum_bg / max(g_integral, 1e-9), 1e-12)

        # log-likelihood (for convergence / reporting)
        compensator = new_mu * g_integral + new_alpha * decay_tail.sum()
        ll = float(np.log(lam).sum() - compensator)

        alpha, beta, mu_scale = new_alpha, new_beta, new_mu
        if abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll

    return HawkesParams(mu_scale, alpha, beta, prev_ll, n)


def simulate_remaining(
    rng: np.random.Generator,
    g_window: np.ndarray,    # (H,) background weight per integer hour-offset over the sim window
    T_remaining: float,      # hours left until window close
    levels: np.ndarray,      # (n_sims,) bootstrapped weekly totals
    alpha: float,
    beta: float,
    seed_decay_sum: np.ndarray,  # (n_sims,) initial Z = sum_seeds e^{-beta*(now - t_j)} (per sim or scalar-broadcast)
    week_hours: float = 168.0,
) -> np.ndarray:
    """Simulate the number of *additional* tweets in the remaining window via Ogata thinning.

    Each sim uses its own bootstrapped ``level`` -> mu_scale = level*(1-alpha)/week_hours, so the
    background integrates (with self-excitation) to the intended weekly volume. Real recent tweets
    enter through ``seed_decay_sum`` (current self-excitation momentum at window time ``now``).
    """
    n_sims = len(levels)
    g_window = np.asarray(g_window, dtype=float)
    Gmax = float(g_window.max()) if g_window.size else 1.0
    mu_scale = np.maximum(levels * (1.0 - alpha) / week_hours, 0.0)  # (n_sims,)
    seed = np.broadcast_to(seed_decay_sum, (n_sims,)).astype(float).copy()

    out = np.zeros(n_sims, dtype=np.int64)
    ab = alpha * beta
    for s in range(n_sims):
        ms = mu_scale[s]
        bg_max = ms * Gmax          # background upper bound over the window for this sim
        t = 0.0
        Z = seed[s]
        count = 0
        # Ogata thinning: lam_bar bounds lambda on the step because Z only decays between events
        while True:
            lam_bar = bg_max + ab * Z + 1e-12
            w = rng.exponential(1.0 / lam_bar)
            t_new = t + w
            if t_new >= T_remaining:
                break
            Z *= np.exp(-beta * w)            # decay self-excitation to new time
            hour_idx = int(t_new)
            g_now = g_window[hour_idx] if hour_idx < g_window.size else g_window[-1]
            lam_true = ms * g_now + ab * Z
            if rng.random() <= lam_true / lam_bar:
                count += 1
                Z += 1.0                       # accepted event raises future intensity
            t = t_new
        out[s] = count
    return out


def simulate_remaining_daily(
    rng: np.random.Generator,
    g_window: np.ndarray,
    T_remaining: float,
    levels: np.ndarray,
    alpha: float,
    beta: float,
    seed_decay_sum: np.ndarray,
    day_edges_h: np.ndarray,   # ascending hour-offsets where the ET day rolls over; last == T_remaining
    week_hours: float = 168.0,
) -> np.ndarray:
    """Like ``simulate_remaining`` but returns per-day counts: (n_sims, n_day_buckets).

    Each accepted event is binned into the day bucket whose upper edge it falls under, so the
    columns sum (per row) to the total remaining count. Used only for the per-day forecast chart.
    """
    n_sims = len(levels)
    edges = np.asarray(day_edges_h, dtype=float)
    n_buckets = len(edges)
    g_window = np.asarray(g_window, dtype=float)
    Gmax = float(g_window.max()) if g_window.size else 1.0
    mu_scale = np.maximum(levels * (1.0 - alpha) / week_hours, 0.0)
    seed = np.broadcast_to(seed_decay_sum, (n_sims,)).astype(float).copy()
    out = np.zeros((n_sims, n_buckets), dtype=np.int64)
    ab = alpha * beta
    for s in range(n_sims):
        ms = mu_scale[s]
        bg_max = ms * Gmax
        t = 0.0
        Z = seed[s]
        while True:
            lam_bar = bg_max + ab * Z + 1e-12
            w = rng.exponential(1.0 / lam_bar)
            t_new = t + w
            if t_new >= T_remaining:
                break
            Z *= np.exp(-beta * w)
            hour_idx = int(t_new)
            g_now = g_window[hour_idx] if hour_idx < g_window.size else g_window[-1]
            lam_true = ms * g_now + ab * Z
            if rng.random() <= lam_true / lam_bar:
                bucket = int(np.searchsorted(edges, t_new, side="right"))
                if bucket >= n_buckets:
                    bucket = n_buckets - 1
                out[s, bucket] += 1
                Z += 1.0
            t = t_new
    return out


def seed_decay_sum(now_minus_event_hours: np.ndarray, beta: float) -> float:
    """Z(now) = sum over recent real events of exp(-beta * age_in_hours)."""
    if len(now_minus_event_hours) == 0:
        return 0.0
    return float(np.sum(np.exp(-beta * np.asarray(now_minus_event_hours, dtype=float))))
