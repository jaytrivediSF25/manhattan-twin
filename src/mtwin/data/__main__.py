"""Run a single source's ETL: `python -m src.mtwin.data <source>`."""

import logging
import sys

SOURCES = {
    "bus": "bus_speeds",
    "crz": "crz_entries",
    "bt": "bt_crossings",
    "dot": "dot_speeds",
    "subway": "subway",
    "weather": "weather",
    "closures": "closures",
    "citibike": "citibike",
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if len(sys.argv) < 2 or sys.argv[1] not in SOURCES:
        sys.exit(f"usage: python -m src.mtwin.data [{'|'.join(SOURCES)}]")
    mod = __import__(f"src.mtwin.data.{SOURCES[sys.argv[1]]}", fromlist=["pull"])
    mod.pull()
