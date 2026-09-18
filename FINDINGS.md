# Interim findings

Status as of the first build session. Data ingest is complete for the primary
sources; two Phase-3 gates have been run.

## Data landed

| Source | Coverage | Volume |
|---|---|---|
| MTA bus route segment speeds (Manhattan) | 2023-01 → 2026-07, 43 months | 2.91 M rows, 634 segments, 43 routes, no nulls |
| TLC yellow taxi OD cells | 2023-01 → 2026-06 | ~900 k cells/month |
| MTA CRZ entries | 2025-01 → ongoing | ~310 k rows/month, 12 gantry groups |
| MTA B&T hourly crossings | pulling | Carey + Queens-Midtown only |

Cordon classification: 40 Manhattan taxi zones inside, with zones
`{48, 50, 140, 141, 163, 202}` flagged as straddling the 60th St line and held
out as a robustness set. Roosevelt Island (202) is excluded on access grounds
despite its latitude.

## Gate 1 — toll-threshold bunching: PASSES decisively

The toll steps down $6.75 (peak → overnight) at 21:00 on every day of the week
(verified against the data, not assumed). Local-linear RD on 10-minute entry
blocks, Jan–Jun 2025:

| Bandwidth | Jump per block | Percent | Placebo p |
|---|---|---|---|
| 60 min | +503 | +35.4% | 0.064 |
| 90 min | +487 | +34.9% | 0.076 |
| 120 min | +528 | +38.4% | 0.039 |

Stable across bandwidths and economically large: roughly 3,000 vehicles/day
shifted into the first hour of the overnight window.

**The falsification test is what makes this credible.** Entries by vehicle class
at the same threshold:

| Class | Jump | Placebo p |
|---|---|---|
| Cars, pickups, vans | **+33.8%** | 0.076 |
| Single-unit trucks | +12.5% | 0.104 |
| Multi-unit trucks | −8.1% | 0.293 |
| Buses | +3.7% | 0.745 |
| Motorcycles | −3.2% | 0.670 |
| **TLC taxi / FHV** | **+4.4%** | 0.293 |

Cars face a time-varying toll and bunch hard. Taxis and FHVs pay a *flat
per-trip* fee with no time variation and show essentially nothing. A recording
artefact at the hour boundary would have hit every class equally, so the
car/taxi contrast rules that out. Drivers demonstrably retime in response to
this price.

## Gate 2 — speed validation: DOES NOT PASS as specified

Neither observable reproduces the published network-wide speed effect under a
simple within-Manhattan difference-in-differences (in-cordon vs above 60th St,
weekday 07:00–19:00, balanced panel, aggregate speed = total distance / total
time).

Calendar-matched shift in the in-cordon minus above-60th gap, relative to the
same month of 2024:

| Observable | 2023 (placebo) | 2025 (policy) | 2026 |
|---|---|---|---|
| Bus segment speeds | **+0.127 mph** | +0.100 mph | −0.002 mph |
| Taxi implied speeds | **+0.143 mph** (sd 0.216) | +0.179 mph (sd 0.136) | +0.093 mph (sd 0.191) |

**The 2023 placebo year moves as much as the policy year.** On buses it moves
*more*. The policy-year signal sits inside the null band that ordinary
year-to-year variation generates.

### What this does and does not mean

It is not yet a verdict on the policy. It is a verdict on this specification,
and the plan predicted both failure modes involved:

1. **The control group is contaminated by construction.** Above-60th Manhattan
   receives diverted traffic, so a within-Manhattan DiD differences away part of
   the very effect it is trying to measure. The planned estimator is continuous
   exposure with outer-borough never-takers, which has not been run yet.
2. **Bus speeds are structurally attenuated.** Segment time blends running time
   with dwell time, and dwell does not respond to traffic. In-cordon bus speed
   is ~5.3 mph against ~6.5 mph for taxis on the same streets, so a given
   running-speed improvement is diluted before it reaches the measure.
3. A strong secular decline runs through 2026 in *both* groups, which any
   pre/post comparison must absorb.

### Next step

Run the exposure design with outer-borough never-takers before drawing any
conclusion about effect size. If the effect remains inside the null band under
the correct estimator, the honest paper is the measurement one: *what it takes
to detect a known policy effect, and why the obvious design fails.*
