"""Reservoir partition of Manhattan and the accumulation scale.

The model works at reservoir resolution rather than link resolution because
that is what the data can identify. Link-level LWR or a cell-transmission model
would need within-link spatial coordinates, link densities and per-cell
capacities, none of which exist here. A macroscopic fundamental diagram over a
handful of reservoirs needs only accumulation and speed, and conservation holds
by construction on a reservoir graph because there are no junctions to
reconcile.

Seven reservoirs: six geographic bands plus one pseudo-reservoir standing for
the excluded roadways (FDR Drive, West Side Highway). The excluded roadways get
their own state because they are the untolled route through the middle of the
cordon -- the diversion channel is then something the model represents rather
than something it pushes into the residual.
"""

from __future__ import annotations

import polars as pl

from ..network.zones import zone_centroids

# (name, min_lat, max_lat, inside_cordon)
BANDS = [
    ("lower_manhattan", 0.0, 40.7220, True),
    ("village_chelsea", 40.7220, 40.7450, True),
    ("midtown_south", 40.7450, 40.7580, True),
    ("midtown_north", 40.7580, 40.7648, True),
    ("uptown_60_96", 40.7648, 40.7900, False),
    ("above_96", 40.7900, 99.0, False),
]
EXCLUDED_ROADWAY = "excluded_roadway"
RESERVOIRS = [b[0] for b in BANDS] + [EXCLUDED_ROADWAY]

# Jam density per lane-km, mid-point of the range reported in the traffic-flow
# literature. The sweep in the sensitivity analysis spans 90-150.
K_JAM_DEFAULT = 120.0

# Manhattan street-network density, lane-km per square kilometre. Used because
# no public lane-km layer was reachable; the accumulation scale is unidentified
# from speed data regardless (see docstring in twin.py), so this enters as an
# external pin subject to the same sweep.
LANE_KM_PER_KM2 = 28.0


def zone_to_reservoir() -> pl.DataFrame:
    """Assign each Manhattan taxi zone to a reservoir by centroid latitude."""
    z = zone_centroids().filter(pl.col("borough") == "Manhattan")
    expr = pl.when(pl.lit(False)).then(pl.lit(""))
    for name, lo, hi, _ in BANDS:
        expr = expr.when((pl.col("lat") >= lo) & (pl.col("lat") < hi)).then(pl.lit(name))
    return z.with_columns(expr.otherwise(pl.lit("above_96")).alias("reservoir"))


def reservoir_capacity(area_km2: dict[str, float] | None = None,
                       k_jam: float = K_JAM_DEFAULT) -> dict[str, float]:
    """Jam accumulation n_jam per reservoir, in vehicles.

    n_jam = k_jam * lane-km, with lane-km approximated from reservoir land area.
    """
    if area_km2 is None:
        area_km2 = _reservoir_area_km2()
    return {r: k_jam * LANE_KM_PER_KM2 * a for r, a in area_km2.items()}


def _reservoir_area_km2() -> dict[str, float]:
    """Approximate reservoir land area from taxi-zone geometry."""
    import duckdb

    from ..network.zones import SHAPEFILE

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    # Shapefile is EPSG:2263 (feet); area in square feet converts to km^2.
    areas = con.execute(
        f"""
        SELECT CAST(LocationID AS INTEGER) AS zone_id,
               ST_Area(geom) / 1e6 * 0.09290304 AS km2
        FROM ST_Read('{SHAPEFILE}')
        """
    ).pl()
    con.close()
    z = zone_to_reservoir().join(areas, on="zone_id", how="left")
    agg = z.group_by("reservoir").agg(pl.col("km2").sum()).to_dicts()
    out = {r["reservoir"]: float(r["km2"]) for r in agg}
    # The excluded roadways are linear features, not an area; their capacity is
    # set from corridor length x lanes rather than from a polygon.
    out.setdefault(EXCLUDED_ROADWAY, 0.0)
    # FDR + West Side Hwy: roughly 2 x 21 km of 3-lane road -> ~126 lane-km,
    # expressed here as the equivalent area under LANE_KM_PER_KM2.
    out[EXCLUDED_ROADWAY] = 126.0 / LANE_KM_PER_KM2
    return out
