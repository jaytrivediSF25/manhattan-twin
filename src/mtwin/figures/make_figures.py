"""Publication figures.

Palette is the validated three-slot categorical set (blue / orange / aqua);
the aqua slot fails the 3:1 contrast check against the surface, so every series
carries a direct label rather than relying on the legend swatch alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parents[3] / "outputs" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddddd"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_bunching(profile_cars, profile_taxi, r_cars, r_taxi) -> Path:
    """Entries by 10-minute block around the 21:00 toll step."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    for ax, prof, r, colour, label in (
        (axes[0], profile_cars, r_cars, BLUE, "Cars, pickups, vans"),
        (axes[1], profile_taxi, r_taxi, ORANGE, "TLC taxi / FHV"),
    ):
        m = prof["minute_of_day"].to_numpy() / 60.0
        v = prof["entries"].to_numpy()
        w = (m >= 18) & (m <= 24)
        ax.plot(m[w], v[w], color=colour, lw=2, marker="o", ms=3.5)
        ax.axvline(21, color=INK, lw=1.2, ls="--")
        ax.annotate("toll drops\n\\$9 → \\$2.25", xy=(21, ax.get_ylim()[1]),
                    xytext=(19.2, max(v[w]) * 0.92), color=MUTED, fontsize=8)
        ax.set_title(f"{label}   jump {r.jump_pct:+.1%}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("hour of day")
        ax.set_xticks([18, 19, 20, 21, 22, 23, 24])
    axes[0].set_ylabel("entries per 10-min block")
    fig.suptitle("Entries bunch at the toll threshold — but only for vehicles facing a time-varying toll",
                 fontsize=11, x=0.01, ha="left")
    return _save(fig, "fig1_bunching.png")


def fig_mfd(curves) -> Path:
    """Speed against normalised accumulation, pre- vs post-policy."""
    n = len(curves)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.1 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, curves):
        ok_pre = ~np.isnan(c.v_pre)
        ok_post = ~np.isnan(c.v_post)
        ax.plot(c.x_bins[ok_pre], c.v_pre[ok_pre], color=BLUE, lw=2, marker="o", ms=3.5)
        ax.plot(c.x_bins[ok_post], c.v_post[ok_post], color=ORANGE, lw=2, marker="s", ms=3.5)
        ax.set_title(c.reservoir.replace("_", " "), fontsize=9, loc="left", color=INK)
        if ok_pre.any():
            ax.annotate("pre", (c.x_bins[ok_pre][-1], c.v_pre[ok_pre][-1]),
                        color=BLUE, fontsize=8, xytext=(3, 2), textcoords="offset points")
        if ok_post.any():
            # Offset the second label away from the first when the curves end
            # close together, or the two annotations overprint.
            gap = (c.v_post[ok_post][-1] - c.v_pre[ok_pre][-1]) if ok_pre.any() else -1.0
            dy = -10 if gap > -0.4 else 2
            ax.annotate("post", (c.x_bins[ok_post][-1], c.v_post[ok_post][-1]),
                        color=ORANGE, fontsize=8, xytext=(3, dy), textcoords="offset points")
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - cols):n]:
        ax.set_xlabel("normalised accumulation")
    axes[0].set_ylabel("speed (mph)")
    fig.suptitle("The fundamental diagram is invariant: post-policy speed tracks the pre-policy curve",
                 fontsize=11, x=0.01, ha="left")
    return _save(fig, "fig2_mfd_invariance.png")


def fig_event_study(es, null_band: float | None = None) -> Path:
    """Event-time coefficients with the pre-period spread drawn on."""
    ks = np.array([int(n[4:-1]) for n in es.names])
    order = np.argsort(ks)
    ks, coef, se = ks[order], es.coef[order], es.se[order]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    pre, post = ks < 0, ks >= 0
    if null_band is not None:
        ax.axhspan(-null_band, null_band, color=GRID, alpha=0.7, lw=0)
        ax.annotate("pre-period spread", (ks.min(), null_band), color=MUTED,
                    fontsize=8, xytext=(2, 3), textcoords="offset points")
    for m, colour, lab in ((pre, MUTED, "pre-policy"), (post, BLUE, "post-policy")):
        ax.errorbar(ks[m], coef[m], yerr=1.96 * se[m], fmt="o", ms=4,
                    color=colour, ecolor=colour, elinewidth=1, capsize=0)
    ax.axvline(-0.5, color=INK, lw=1.2, ls="--")
    ax.axhline(0, color=MUTED, lw=1)
    ax.annotate("policy start", (-0.5, ax.get_ylim()[1]), color=MUTED, fontsize=8,
                xytext=(3, -12), textcoords="offset points")
    ax.set_xlabel("months relative to congestion pricing")
    ax.set_ylabel("log speed vs never-takers")
    fig.suptitle("Event study: a visible jump at policy start, but pre-trends are not flat",
                 fontsize=11, x=0.01, ha="left")
    return _save(fig, "fig3_event_study.png")


def fig_excluded_share(share: pl.DataFrame) -> Path:
    """Untolled share of zone entries over the post-policy period."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(share.height)
    y = share["excluded_pct"].to_numpy()
    ax.plot(x, y, color=AQUA, lw=2, marker="o", ms=3.5)
    ax.annotate("excluded-roadway share", (x[-1], y[-1]), color=AQUA, fontsize=8,
                xytext=(-140, 8), textcoords="offset points")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(share["ym"].to_list()[::3], rotation=45, ha="right")
    ax.set_ylabel("% of all zone entries")
    ax.set_ylim(0, max(y) * 1.4)
    fig.suptitle("No growth in diversion onto untolled roads inside the cordon",
                 fontsize=11, x=0.01, ha="left")
    return _save(fig, "fig4_excluded_share.png")


def main() -> None:
    """Regenerate every figure from cached data."""
    from datetime import date

    from ..analysis import bunching_rd as brd
    from ..analysis.diversion import excluded_share
    from ..analysis.exposure_did import event_study
    from ..data import crz_entries
    from ..models import mfd
    from ..panel.build import bus_segment_panel, reservoir_panel

    # Via the data layer rather than a glob on data/raw, so `MTWIN_USE_SAMPLE=1`
    # regenerates the same figures from the committed sample tables.
    crz = crz_entries.load()
    cars = brd.block_profile(crz, vehicle_class="1 - Cars, Pickups and Vans")
    taxi = brd.block_profile(crz, vehicle_class="TLC Taxi/FHV")
    print(fig_bunching(cars, taxi, brd.estimate(cars), brd.estimate(taxi)))

    # The three-threshold panel is the sign test, and the README leans on it, so
    # it is regenerated here rather than left as a one-off from a notebook.
    profiles = brd.threshold_profiles(crz)
    results = {n: brd.estimate(p, threshold=brd.THRESHOLDS[n]["minute"]) for n, p in profiles.items()}
    print(fig_thresholds(profiles, results))

    panel = reservoir_panel()
    pre = pl.col("service_date") < date(2024, 6, 1)
    post = pl.col("service_date") >= date(2025, 1, 5)
    print(fig_mfd(mfd.build_curves(panel, pre, post, per_period_scale=True)))

    bp = bus_segment_panel(
        boroughs=("Manhattan", "Brooklyn", "Queens", "Bronx")
    ).filter(pl.col("grp").is_in(["crz", "outer"]))
    es = event_study(bp, weight_col="n_trips", omit=-1)
    pre_coefs = [c for n, c in zip(es.names, es.coef) if int(n[4:-1]) < 0]
    print(fig_event_study(es, null_band=float(np.std(pre_coefs))))

    print(fig_excluded_share(excluded_share()))



def fig_thresholds(profiles: dict, results: dict) -> Path:
    """All three toll thresholds, showing the response flip with price direction."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (name, prof) in zip(axes, profiles.items()):
        r = results[name]
        thr = r.threshold_min / 60.0
        m = prof["minute_of_day"].to_numpy() / 60.0
        v = prof["entries"].to_numpy()
        w = (m >= thr - 3) & (m <= thr + 3)
        rising = "rise" in name
        colour = ORANGE if rising else BLUE
        ax.plot(m[w], v[w], color=colour, lw=2, marker="o", ms=3)
        ax.axvline(thr, color=INK, lw=1.2, ls="--")
        arrow = "toll rises\n$2.25 → $9" if rising else "toll falls\n$9 → $2.25"
        ax.set_title(f"{name}\njump {r.jump_pct:+.1%}", fontsize=9, loc="left", color=INK)
        # Park the note in the emptier half of the panel: entries fall away from
        # the threshold at a price cut and rise into it at a price rise, so the
        # free corner is on opposite sides in the two cases.
        lo, hi = v[w].min(), v[w].max()
        y_text = lo + 0.08 * (hi - lo) if not rising else hi * 0.98
        x_text = thr + 0.4 if not rising else thr - 2.9
        ax.annotate(arrow.replace("$", r"\$"), xy=(thr, hi),
                    xytext=(x_text, y_text), color=MUTED, fontsize=8)
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("entries per 10-min block")
    fig.suptitle(
        "The response flips sign with the direction of the price change",
        fontsize=11, x=0.01, ha="left",
    )
    return _save(fig, "fig5_thresholds.png")


if __name__ == "__main__":
    main()
