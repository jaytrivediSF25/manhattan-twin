"""Verified Socrata dataset endpoints.

Every id here was confirmed live against the API during planning. Two sources
are split across a 2024/2025 dataset boundary and must be stitched; the seams
are contiguous and are noted inline.
"""

from __future__ import annotations

from datetime import date

from .socrata import Dataset

NYC = "data.cityofnewyork.us"
NYS = "data.ny.gov"

# --- Primary observable: link-level speeds on the tolled surface grid ---------
# Stop-to-stop segments with lat/lon at both ends. The probe (a bus) is exempt
# from the congestion toll and runs a fixed route, so unlike taxi probes it is
# untreated and carries no route-selection bias.
# Seam: 58t6-89vi ends 2024-12-01, kufs-yh3x starts 2025-01-01. Contiguous.
BUS_SEG_2324 = Dataset(NYS, "58t6-89vi", "bus_segment_speeds_2023_2024", "timestamp")
BUS_SEG_2025 = Dataset(NYS, "kufs-yh3x", "bus_segment_speeds_2025_plus", "timestamp")

# Pre-aggregated CBD cut, used for validation rather than modelling.
BUS_CBD = Dataset(NYS, "r6db-kkzj", "cbd_bus_speeds", "month")

# --- Diversion channel: untolled highways inside the cordon ------------------
# Only ~25 Manhattan links, all limited-access (FDR, West Side Hwy, tunnels,
# bridge approaches). FDR and WSH are *excluded roadways* under the toll, which
# is exactly why they measure diversion.
DOT_SPEEDS = Dataset(NYC, "i4gi-tjb9", "dot_link_speeds", "data_as_of")

# --- Treatment intensity -----------------------------------------------------
# 10-minute resolution, which is what makes the toll-threshold bunching design
# possible. Begins at launch, so it has no pre-period on its own.
CRZ_ENTRIES = Dataset(NYS, "t6yz-b64h", "crz_entries", "toll_date")

# Supplies the pre-period the CRZ dataset lacks, back to 2019, for tunnels and
# tolled bridges. The free East River bridges are absent -- a stated limitation.
BT_CROSSINGS = Dataset(NYS, "ebfx-2m7v", "bt_hourly_crossings", "transit_timestamp")

# --- Mode shift --------------------------------------------------------------
# Seam: wujg-7c2s covers 2020-2024, 5wq4-mkjj covers 2025+.
SUBWAY_2024 = Dataset(NYS, "wujg-7c2s", "subway_hourly_2020_2024", "transit_timestamp")
SUBWAY_2025 = Dataset(NYS, "5wq4-mkjj", "subway_hourly_2025_plus", "transit_timestamp")

# --- Study period ------------------------------------------------------------
# The pre-period deliberately stops at 2024-06-01: Hochul paused congestion
# pricing indefinitely in June 2024 and revived it at $9 that November, so both
# months carry anticipatory behavioural shocks. Training on them would
# contaminate the "pre-policy" baseline.
CLEAN_PRE_START = date(2023, 1, 1)
CLEAN_PRE_END = date(2024, 6, 1)

PAUSE_ANNOUNCED = date(2024, 6, 5)
REVIVAL_ANNOUNCED = date(2024, 11, 14)
POLICY_START = date(2025, 1, 5)

# Full span to ingest; bus segment speeds currently run through 2026-07.
DATA_START = date(2023, 1, 1)
DATA_END = date(2026, 8, 1)
