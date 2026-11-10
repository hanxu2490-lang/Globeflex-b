"""Question 1 data, built the way Modules A, A3, B and C build it.

Monthly MSCI net total return indices in USD, 2001-01 to 2026-08. Series are built in simple
returns (last observation of each month), EM without a part is backed out with last month's
weights, and log returns ln(1 + r) are used for correlations, betas, regressions and tests.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

START, END = "2001-01", "2026-08"
COUNTRIES = ["Taiwan", "Korea", "China", "India", "Brazil", "South Africa", "Mexico", "Saudi Arabia"]
BLOCKS = {"2001–05": ("2001-01", "2005-12"), "2006–10": ("2006-01", "2010-12"), "2011–15": ("2011-01", "2015-12"),
          "2016–20": ("2016-01", "2020-12"), "2021–26": ("2021-01", END)}
FULL = {"Full sample": (START, END)}
NAVY, GOLD, GREY, BLUE, RED = "#1B2A41", "#D98E04", "#8A94A6", "#4F7CAC", "#C0392B"   # the team's palette

# the teammates' file names -> the files in data/
FILES = {
    "MSCI_EM_Returns_daily.xlsx": "MSCI EM Returns daily.xlsx",
    "MSCI_EM_IT_market_cap___returns.xlsx": "MSCI EM IT market cap & returns.xlsx",
    "MSCI_USA__net_total_return.xlsx": "MSCI USA  net total return.xlsx",
    "MSCI_USA_IT.xlsx": "MSCI USA IT.xlsx",
    "MSCI_EM_INDX_MARKET_CAP_MONTHLY.xlsx": "MSCI EM INDX MARKET_CAP_MONTHLY.xlsx",
    "UST3M_daily.xlsx": "UST3M daily.xlsx",
    "US_index.xlsx": "SP500 and US Agg.xlsx",
    "index.xlsx": "SP500 and US Agg.xlsx",
}


def load(file, sheet=0):
    """Read a 2-column Bloomberg export (date, value); text rows dropped; Excel serial dates handled."""
    d = pd.read_excel(C.DATA_DIR / FILES.get(file, file), sheet_name=sheet, header=None).iloc[:, :2]
    d.columns = ["date", "value"]
    num = pd.to_numeric(d["date"], errors="coerce").astype(float)
    with np.errstate(over="ignore"):
        serial = pd.Timestamp("1899-12-30") + pd.to_timedelta(num.where(num.between(20000, 60000)), unit="D")
    d["date"] = serial.fillna(pd.to_datetime(d["date"].where(serial.isna()), errors="coerce", format="mixed"))
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().drop_duplicates("date").set_index("date")["value"].sort_index()


def to_month(df):
    """Last observation of each month, indexed at month-end."""
    df = df.groupby(df.index.to_period("M")).last()
    df.index = df.index.to_timestamp("M")
    return df


@lru_cache(maxsize=1)
def panel():
    """Module A's monthly panel plus the extra series A3, B and C read. Returns a dict."""
    levels = pd.DataFrame({
        "EM": load("MSCI_EM_Returns_daily.xlsx"),                                  # NDUEEGF
        "EM_IT": load("MSCI_EM_IT_market_cap___returns.xlsx", " Returns daily"),   # M1EF0IT
        "USA": load("MSCI_USA__net_total_return.xlsx"),                            # NDDUUS
        "USA_IT": load("MSCI_USA_IT.xlsx"),                                        # M1US0ITE
        **{c: load("Country_returns_daily.xlsx", c) for c in COUNTRIES}})
    ret = to_month(levels).pct_change(fill_method=None).loc[START:END]
    mcap = to_month(pd.DataFrame({"EM": load("MSCI_EM_INDX_MARKET_CAP_MONTHLY.xlsx"),
                                  "EM_IT": load("MSCI_EM_IT_market_cap___returns.xlsx", "Market Cap monthly"),
                                  **{c: load("Country_mktcap_monthly.xlsx", c) for c in COUNTRIES}}))
    TS_ALL = (mcap["EM_IT"] / mcap["EM"]).rename("EM_IT")         # IT share of EM, full history
    W_ALL = mcap[COUNTRIES].div(mcap["EM"], axis=0)                # country weights in EM
    rf = (to_month(load("UST3M_daily.xlsx").to_frame())["value"] / 100 / 12).shift(1).reindex(ret.index)

    def ex(parts, weights):
        """EM without `parts`: (r_EM - sum w*r) / (1 - sum w), last month's weights; empty if any input missing."""
        w = weights[parts].shift(1).reindex(ret.index); x = ret[parts]
        out = (ret["EM"] - (w * x).sum(axis=1)) / (1 - w.sum(axis=1))
        return out.where(w.notna().all(axis=1) & x.notna().all(axis=1))

    ret["EM_exIT"] = ex(["EM_IT"], TS_ALL.to_frame())
    ret["EM_exTWKR"] = ex(["Taiwan", "Korea"], W_ALL)

    # U.S. stock / bond benchmark (A3, B): S&P 500 total return (SPXT) and Bloomberg U.S. Aggregate (LBUSTRUU)
    us = to_month(pd.DataFrame({"S&P 500": load("US_index.xlsx", "S&P500 daily"),
                                "US Agg": load("US_index.xlsx", "LBUSTRUU Index daily")})).pct_change(fill_method=None)
    # extra raw series (B, C)
    lv = to_month(pd.DataFrame({"EMSC_P": load("EMSC.xlsx", "EMSC_Price_MXEFSC"),     # EM Small Cap price
                                "EMSC_N": load("EMSC.xlsx", "EMSC_Net_M1EFSC"),       # EM Small Cap net TR (from 2010-05)
                                "EM_P": load("MSCI EM price index.xlsx"),             # EM price (pairs with EMSC_P)
                                "WORLD": load("MSCI WORLD net total return.xlsx")}))
    extra = lv[["EMSC_P", "EMSC_N", "EM_P"]].pct_change(fill_method=None).reindex(ret.index)
    dd = lv["WORLD"] / lv["WORLD"].cummax() - 1                  # running peak from 1990
    crisis = (dd <= C.FIN_DRAWDOWN).reindex(ret.index)

    return {"ret": ret, "lret": np.log1p(ret), "mcap": mcap, "TS_ALL": TS_ALL, "W_ALL": W_ALL,
            "tech_share": TS_ALL.loc[START:END], "W": W_ALL.loc[START:END], "rf": rf, "ex": ex,
            "us": us.reindex(ret.index), "extra": extra, "crisis": crisis}


def weekly_log():
    """Weekly (Friday) log returns for Module B, weeks ending 2001-01 to 2026-08 (5 Jan 2001 to 28 Aug 2026)."""
    lv = pd.concat([load("MSCI_EM_Returns_daily.xlsx").rename("EM"), load("MSCI_USA__net_total_return.xlsx").rename("USA"),
                    load("MSCI_USA_IT.xlsx").rename("USA_IT"),
                    load("EMSC.xlsx", "EMSC_Price_MXEFSC").rename("EM Small Cap Price")], axis=1).sort_index()
    w = lv.resample("W-FRI").last().pct_change(fill_method=None)
    return np.log1p(w).replace([np.inf, -np.inf], np.nan).loc[START:END]
