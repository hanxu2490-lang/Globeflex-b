"""Project paths (used by both questions) and the Question 2 settings (crisis regression and event study).

Question 1 settings live at the top of each module (q1_data.py, q1_module_*.py), as in the teammates' notebooks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
TABLE_DIR = OUT_DIR / "q2"               # Question 2 tables and figures
FIG_DIR = OUT_DIR / "q2"

# ---------------------------------------------------------------- sample
SAMPLE_START = "2001-01"        # first monthly return (Jan 2001)
SAMPLE_END = "2026-08"          # last monthly return (Aug 2026); daily data also end 31 Aug 2026

# ---------------------------------------------------------------- crisis flags (Step 3)
FIN_DRAWDOWN = -0.10            # Fin_t = 1 when MSCI World is more than 10% below its running peak
GEO_QUANTILE = 0.90             # Geo_t = 1 when GPR or TPU is in its top decile (decile over the study sample)
CURRENT_START = "2018-01"       # Current_t = 1 from 2018
CURRENT_START_ROBUST = "2022-01"  # robustness: current period from 2022

# ---------------------------------------------------------------- daily event study (Step 3)
EVENT_PRE = 5                   # trading days before day 0
EVENT_POST = 20                 # trading days after day 0

# Crisis vs non-crisis placebo (Module E placebo test)
PLACEBO_DRAWS = 5000            # placebo sets
PLACEBO_SEED = 20261103

# Named crises. Day 0 = first trading day on or after the date.
# type: Fin = financial, Geo = war / geopolitical, Trade = trade conflict. period: Past = before 2018.
CRISES = [
    ("Lehman bankruptcy",               "2008-09-15", "Fin",   "Past"),
    ("US downgrade / euro crisis",      "2011-08-08", "Fin",   "Past"),
    ("China devaluation",               "2015-08-11", "Fin",   "Past"),
    ("9/11 attacks",                    "2001-09-11", "Geo",   "Past"),
    ("Iraq invasion",                   "2003-03-20", "Geo",   "Past"),
    ("Crimea annexation",               "2014-03-03", "Geo",   "Past"),
    ("COVID crash",                     "2020-02-24", "Fin",   "Current"),
    ("SVB collapse",                    "2023-03-10", "Fin",   "Current"),
    ("Russia invades Ukraine",          "2022-02-24", "Geo",   "Current"),
    ("Hamas attack on Israel",          "2023-10-09", "Geo",   "Current"),
    ("Israel-Iran war",                 "2025-06-13", "Geo",   "Current"),
    ("US Section 301 China tariffs",    "2018-03-22", "Trade", "Current"),
    ("US-China tariff escalation",      "2019-05-06", "Trade", "Current"),
    ("Yuan breaks 7 / new tariffs",     "2019-08-05", "Trade", "Current"),
    ("'Liberation Day' tariffs",        "2025-04-03", "Trade", "Current"),
]
