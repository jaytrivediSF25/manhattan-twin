"""Macroscopic fundamental diagram: estimation and the invariance test.

The fundamental diagram is the physics block of the twin -- the relation
between how many vehicles are in a reservoir and how fast they move. It is
assumed policy-invariant: congestion pricing changes how many vehicles choose
to be there, not how a street behaves at a given density.

That assumption is testable, and testing it is the point of the project:

  * post-policy points land ON the pre-policy curve, at lower accumulation
      -> the physics is complete; the entire speed change is demand moving
         along a fixed curve, and all the policy content is behavioural.
  * post-policy points land OFF the curve
      -> the physics itself shifted, and the residual says where: fleet
         composition, curb friction, signal retiming, street redesign.

Either way the question is answered, which is what makes this the experiment
worth building toward.

Accumulation is observed only up to a per-reservoir scale constant (see
`panel.build.reservoir_panel`), so each reservoir is normalised by its own
pre-period 95th percentile before curves are compared.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class MFDCurve:
    reservoir: str
    x_bins: np.ndarray       # normalised accumulation, bin centres
    v_pre: np.ndarray        # pre-period median speed in each bin
    v_post: np.ndarray       # post-period median speed in each bin
    n_pre: np.ndarray        # observation counts
    n_post: np.ndarray
    scale: float             # pre-period p95 of raw accumulation proxy

    @property
    def shift(self) -> np.ndarray:
        """Post minus pre speed at matched accumulation."""
        return self.v_post - self.v_pre


def normalise(panel: pl.DataFrame, pre_mask: pl.Expr) -> pl.DataFrame:
    """Scale each reservoir's accumulation proxy by its own pre-period p95.

    Common-scale normalisation. Valid only if taxi share of traffic is stable
    across the policy date, which it is not -- see `normalise_by_period`.
    """
    scales = (
        panel.filter(pre_mask)
        .group_by("reservoir")
        .agg(pl.col("accum_proxy").quantile(0.95).alias("scale"))
    )
    return panel.join(scales, on="reservoir").with_columns(
        (pl.col("accum_proxy") / pl.col("scale")).alias("x")
    )


def normalise_by_period(panel: pl.DataFrame, pre_mask: pl.Expr, post_mask: pl.Expr) -> pl.DataFrame:
    """Scale each reservoir separately within each period.

    The accumulation proxy is taxi trips divided by speed, so it tracks true
    accumulation only up to the taxi share of traffic -- and that share is not
    constant across the policy date. Yellow-taxi volumes were still recovering
    over 2023-2025, and the toll made taxis relatively cheaper than driving, so
    a common scale would read rising taxi share as rising congestion.

    Normalising within each period removes any level factor that is constant
    inside the period, which is what a share shift is to first order. What
    survives is the *shape* of the speed-accumulation relation, and shape is
    what the invariance test needs: the accumulation scale was never identified
    from speed data anyway.
    """
    out = []
    for mask, label in ((pre_mask, "pre"), (post_mask, "post")):
        d = panel.filter(mask)
        scales = d.group_by("reservoir").agg(pl.col("accum_proxy").quantile(0.95).alias("scale"))
        out.append(
            d.join(scales, on="reservoir")
            .with_columns([(pl.col("accum_proxy") / pl.col("scale")).alias("x"),
                           pl.lit(label).alias("era")])
        )
    return pl.concat(out, how="diagonal_relaxed")


def build_curves(
    panel: pl.DataFrame,
    pre_mask: pl.Expr,
    post_mask: pl.Expr,
    *,
    n_bins: int = 12,
    x_max: float = 1.3,
    min_obs: int = 30,
    per_period_scale: bool = True,
) -> list[MFDCurve]:
    """Bin accumulation and compare pre- and post-period speed within bins.

    Comparing speeds *at matched accumulation* is what separates the two
    stories. An unconditional speed comparison cannot tell "fewer cars on the
    same streets" from "the same cars on different streets".
    """
    d = (normalise_by_period(panel, pre_mask, post_mask) if per_period_scale
         else normalise(panel, pre_mask))
    edges = np.linspace(0, x_max, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    out: list[MFDCurve] = []

    for res in sorted(d["reservoir"].unique().to_list()):
        sub = d.filter(pl.col("reservoir") == res)
        pre = sub.filter(pl.col("era") == "pre") if per_period_scale else sub.filter(pre_mask)
        post = sub.filter(pl.col("era") == "post") if per_period_scale else sub.filter(post_mask)
        scale = float(sub["scale"][0]) if sub.height else float("nan")

        v_pre, v_post, n_pre, n_post = [], [], [], []
        for lo, hi in itertools.pairwise(edges):
            p = pre.filter((pl.col("x") >= lo) & (pl.col("x") < hi))["speed_mph"]
            q = post.filter((pl.col("x") >= lo) & (pl.col("x") < hi))["speed_mph"]
            v_pre.append(float(p.median()) if p.len() >= min_obs else np.nan)
            v_post.append(float(q.median()) if q.len() >= min_obs else np.nan)
            n_pre.append(p.len())
            n_post.append(q.len())

        out.append(
            MFDCurve(res, centres, np.array(v_pre), np.array(v_post),
                     np.array(n_pre), np.array(n_post), scale)
        )
    return out


def invariance_summary(curves: list[MFDCurve]) -> pl.DataFrame:
    """Per-reservoir summary of how far post-period speed sits off the curve."""
    rows = []
    for c in curves:
        ok = ~np.isnan(c.v_pre) & ~np.isnan(c.v_post)
        if ok.sum() == 0:
            continue
        w = (c.n_pre + c.n_post)[ok].astype(float)
        shift = c.shift[ok]
        rel = shift / c.v_pre[ok]
        rows.append(
            {
                "reservoir": c.reservoir,
                "bins_compared": int(ok.sum()),
                "mean_shift_mph": float(np.average(shift, weights=w)),
                "mean_shift_pct": float(np.average(rel, weights=w)),
                "max_abs_shift_mph": float(np.max(np.abs(shift))),
            }
        )
    return pl.DataFrame(rows)


def accumulation_change(panel: pl.DataFrame, pre_mask: pl.Expr, post_mask: pl.Expr) -> pl.DataFrame:
    """Change in mean normalised accumulation, i.e. movement ALONG the curve."""
    d = normalise(panel, pre_mask)
    pre = d.filter(pre_mask).group_by("reservoir").agg(pl.col("x").mean().alias("x_pre"))
    post = d.filter(post_mask).group_by("reservoir").agg(pl.col("x").mean().alias("x_post"))
    return (
        pre.join(post, on="reservoir")
        .with_columns(((pl.col("x_post") / pl.col("x_pre") - 1) * 100).alias("accum_pct_change"))
        .sort("reservoir")
    )
