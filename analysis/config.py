"""Settings for the GlobeFlex B analysis.

Every choice the methods page leaves open is fixed here, so it can be changed in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

# ---------------------------------------------------------------- sample
SAMPLE_START = "2001-01"        # first monthly return (Jan 2001)
SAMPLE_END = "2026-09"          # last monthly return (Sep 2026; MSCI data end 25 Sep 2026)
POST2010_START = "2010-06"      # first monthly return of EM Small Cap net (net series starts May 2010)

# ---------------------------------------------------------------- crisis flags (Step 3)
FIN_DRAWDOWN = -0.10            # Fin_t = 1 when MSCI World is more than 10% below its running peak
GEO_QUANTILE = 0.90             # Geo_t = 1 when GPR or TPU is in its top decile (decile over the study sample)
CURRENT_START = "2018-01"       # Current_t = 1 from 2018
CURRENT_START_ROBUST = "2022-01"  # robustness: current period from 2022

# ---------------------------------------------------------------- Bai-Perron (Step 2)
BP_TRIM = 0.15                  # minimum regime length = 15% of the sample
BP_MAX_BREAKS = 5               # largest number of breaks considered for the BIC
BP_SUPF_CV5 = 11.47             # 5% critical value of supF(1), q = 2, trim 0.15 (Bai & Perron 2003, Table 1)

# ---------------------------------------------------------------- portfolio (Step 4)
OOS_START = "2006-01"           # first out-of-sample month
ROLL_WINDOW = 60                # months used to estimate each month's minimum-variance weights
LONG_ONLY = True                # EM weight kept between 0% and 100%

# ---------------------------------------------------------------- daily event study (Step 3)
EVENT_PRE = 5                   # trading days before day 0
EVENT_POST = 20                 # trading days after day 0

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
