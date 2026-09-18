"""Exposure design with outer-borough never-takers.

The obvious specification -- in-cordon versus Manhattan above 60th St -- is
biased by construction, because diverted traffic lands in the control group and
differences away part of the effect being measured. Streets in Brooklyn, Queens
and the Bronx are far enough from the cordon that diversion does not reach them,
so they serve as never-takers with exposure E = 0.

Three exposure levels are estimated jointly:

    E = 1   inside the cordon                      (treated)
    E = s   Manhattan above 60th St                (spillover; sign not signable
                                                    a priori, so estimated, not
                                                    assumed away)
    E = 0   Brooklyn / Queens / Bronx              (never-takers, omitted base)

Specification, with two-way fixed effects and errors clustered on the unit:

    log(speed_it) = a_i + g_t + d1 (CRZ_i x Post_t) + d2 (Above60_i x Post_t) + e_it

Unit fixed effects absorb permanent speed differences between streets; period
fixed effects absorb anything hitting the whole city in a given month, which is
what the naive pre/post comparison could not handle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class DiDResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    n_obs: int
    n_units: int
    r2_within: float

    def summary(self) -> str:
        lines = [f"{'term':<28} {'coef':>10} {'se':>9} {'t':>7}"]
        lines.append("-" * 57)
        for n, c, s in zip(self.names, self.coef, self.se):
            t = c / s if s > 0 else float("nan")
            star = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
            lines.append(f"{n:<28} {c:>+10.5f} {s:>9.5f} {t:>7.2f} {star}")
        lines.append(f"n={self.n_obs:,}  units={self.n_units:,}  within-R2={self.r2_within:.4f}")
        return "\n".join(lines)


def _codes(s: pl.Series) -> np.ndarray:
    """Dense integer codes for a grouping column of any dtype."""
    return s.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy().astype(int)


def _demean_two_way(
    y: np.ndarray, X: np.ndarray, unit: np.ndarray, period: np.ndarray,
    w: np.ndarray | None = None,
    tol: float = 1e-10, max_iter: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Absorb unit and period fixed effects by alternating projections.

    Building explicit dummies for hundreds of units times dozens of periods is
    wasteful; iterated within-transformation on each dimension converges to the
    same two-way-demeaned residuals.

    Weights must enter the *means*, not be applied to the data beforehand.
    Scaling y and X by sqrt(w) and then subtracting unweighted group means does
    not absorb the fixed effects, and the unabsorbed group levels leak into the
    coefficients -- which shows up as a large constant offset across every
    event-time estimate.
    """
    if w is None:
        w = np.ones_like(y, dtype=float)

    def demean(v: np.ndarray, g: np.ndarray) -> np.ndarray:
        ng = g.max() + 1
        shape = (ng,) + v.shape[1:]
        wv = v * w.reshape((-1,) + (1,) * (v.ndim - 1))
        sums = np.zeros(shape)
        np.add.at(sums, g, wv)
        wsum = np.zeros(ng)
        np.add.at(wsum, g, w)
        means = sums / np.maximum(wsum, 1e-12).reshape((-1,) + (1,) * (v.ndim - 1))
        return v - means[g]

    yd, Xd = y.copy(), X.copy()
    for _ in range(max_iter):
        y_prev, X_prev = yd.copy(), Xd.copy()
        yd = demean(demean(yd, unit), period)
        Xd = demean(demean(Xd, unit), period)
        # Both must be checked. On an unbalanced panel the response converges
        # in a couple of sweeps while the regressors need many more, so testing
        # only `yd` exits early and leaves the design matrix un-absorbed --
        # which barely moves a two-regressor DiD but badly biases an event
        # study with dozens of event-time dummies.
        moved = max(np.max(np.abs(yd - y_prev)),
                    np.max(np.abs(Xd - X_prev)) if Xd.size else 0.0)
        if moved < tol:
            break
    return yd, Xd


def _cluster_se(X: np.ndarray, resid: np.ndarray, cluster: np.ndarray, xtx_inv: np.ndarray) -> np.ndarray:
    """Cluster-robust sandwich, clustering on the unit dimension."""
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cluster):
        m = cluster == c
        Xc = X[m]
        uc = resid[m]
        s = Xc.T @ uc
        meat += np.outer(s, s)
    n_c = len(np.unique(cluster))
    adj = n_c / max(n_c - 1, 1)
    V = xtx_inv @ meat @ xtx_inv * adj
    return np.sqrt(np.maximum(np.diag(V), 0))


def fit(
    df: pl.DataFrame,
    *,
    y_col: str = "log_speed",
    unit_col: str = "unit",
    period_col: str = "period",
    treat_cols: tuple[str, ...] = ("crz_post", "above60_post"),
    weight_col: str | None = None,
) -> DiDResult:
    """Two-way fixed-effects regression with cluster-robust errors."""
    d = df.drop_nulls([y_col, unit_col, period_col, *treat_cols])
    y = d[y_col].to_numpy().astype(float)
    X = np.column_stack([d[c].to_numpy().astype(float) for c in treat_cols])
    unit = _codes(d[unit_col])
    period = _codes(d[period_col])

    w = d[weight_col].to_numpy().astype(float) if weight_col else None
    beta, se, resid, r2 = _wls_fe(y, X, unit, period, w)
    return DiDResult(list(treat_cols), beta, se, len(y), len(np.unique(unit)), r2)


def _wls_fe(y, X, unit, period, w=None):
    """Weighted least squares with two-way fixed effects absorbed."""
    yd, Xd = _demean_two_way(y, X, unit, period, w)
    sw = np.sqrt(w) if w is not None else np.ones_like(y)
    yw, Xw = yd * sw, Xd * sw[:, None]
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = xtx_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    se = _cluster_se(Xw, resid, unit, xtx_inv)
    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    r2 = 1 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, resid, r2


def event_study(
    df: pl.DataFrame,
    *,
    y_col: str = "log_speed",
    unit_col: str = "unit",
    period_col: str = "period",
    treat_col: str = "is_crz",
    rel_col: str = "rel_period",
    omit: int = -1,
    weight_col: str | None = None,
) -> DiDResult:
    """Period-by-period treatment effects, to expose pre-trends.

    A design whose 'effect' is already present before the policy started is not
    measuring the policy, so the pre-period coefficients matter as much as the
    post-period ones.
    """
    d = df.drop_nulls([y_col, unit_col, period_col, treat_col, rel_col])
    rels = sorted(r for r in d[rel_col].unique().to_list() if r != omit)
    cols, names = [], []
    treat = d[treat_col].to_numpy().astype(float)
    rel = d[rel_col].to_numpy()
    for r in rels:
        cols.append(treat * (rel == r))
        names.append(f"rel[{r:+d}]")
    X = np.column_stack(cols)
    y = d[y_col].to_numpy().astype(float)
    unit = _codes(d[unit_col])
    period = _codes(d[period_col])
    w = d[weight_col].to_numpy().astype(float) if weight_col else None
    beta, se, _, r2 = _wls_fe(y, X, unit, period, w)
    return DiDResult(names, beta, se, len(y), len(np.unique(unit)), r2)
