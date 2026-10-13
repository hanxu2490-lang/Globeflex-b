"""Load the raw files and build the monthly, weekly and daily datasets.

All series are aligned by calendar date (month, week or day), never by row position.
Returns are log returns in percent: r_t = 100 * ln(P_t / P_{t-1}).
"""
import numpy as np
import pandas as pd

from . import config as C

# file name, sheet, rows to skip before the data
PRICE_FILES = {
    "EM":       ("MSCI EM Returns daily.xlsx", 0, 0),                      # MSCI EM, USD net total return
    "EM_PX":    ("MSCI EM price index.xlsx", 0, 0),                        # MSCI EM, USD price (same basis as EMSC)
    "EM_LC":    ("MSCI Emerging Net Total Return Local Index.xlsx", 0, 0), # MSCI EM, local-currency net total return
    "EMSC":     ("EMSC.xlsx", "EMSC_Price_MXEFSC", 0),                     # MSCI EM Small Cap, USD price
    "EMSC_NET": ("EMSC.xlsx", "EMSC_Net_M1EFSC", 0),                       # MSCI EM Small Cap, USD net (from 2010)
    "EM_IT":    ("MSCI EM IT market cap & returns.xlsx", " Returns daily", 0),
    "US":       ("MSCI USA  net total return.xlsx", 0, 0),
    "US_IT":    ("MSCI USA IT.xlsx", 0, 0),
    "WORLD":    ("MSCI WORLD net total return.xlsx", 0, 0),
}
MARKET_FILES = {
    "VIX":   ("VIX daily.xlsx", 0, 0),
    "DXY":   ("DXY daily.xlsx", 0, 0),
    "UST3M": ("UST3M daily.xlsx", 0, 0),
}
MCAP_FILES = {
    "MCAP_EM_IT": ("MSCI EM IT market cap & returns.xlsx", "Market Cap monthly", 0),
    "MCAP_EM":    ("MSCI EM INDX MARKET_CAP_MONTHLY.xlsx", 0, 6),
}

LABELS = {
    "EM": "EM", "EM_PX": "EM (price)", "EM_LC": "EM (local ccy)", "EMSC": "EM Small Cap (price)",
    "EMSC_NET": "EM Small Cap (net)", "EM_EXIT": "EM ex-IT", "EM_IT": "EM IT",
    "US": "USA", "US_IT": "USA IT", "WORLD": "World",
}


def _read_two_cols(fname, sheet, skip):
    df = pd.read_excel(C.DATA_DIR / fname, sheet_name=sheet, header=None, skiprows=skip).iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"].sort_index()
    assert not s.index.duplicated().any(), f"duplicate dates in {fname}"
    return s


def load_raw():
    """Return a dict of raw level series, all cut at the last date common to the MSCI files."""
    raw = {k: _read_two_cols(*v) for k, v in {**PRICE_FILES, **MARKET_FILES, **MCAP_FILES}.items()}
    end = min(raw[k].index.max() for k in PRICE_FILES)            # 25 Sep 2026
    raw = {k: s.loc[:end] for k, s in raw.items()}

    gpr = pd.read_excel(C.DATA_DIR / "GPR monthly.xls")
    gpr = gpr.set_index(pd.to_datetime(gpr["month"]))["GPR"].dropna()
    tpu = pd.read_excel(C.DATA_DIR / "TPU monthly.xlsx", sheet_name="TPU_MONTHLY")
    tpu = tpu.set_index(pd.to_datetime(tpu["DATE"]))["TPU"].dropna()
    raw["GPR"] = gpr.set_axis(gpr.index.to_period("M"))
    raw["TPU"] = tpu.set_axis(tpu.index.to_period("M"))
    raw["END_DATE"] = end
    return raw


def _month_end(s):
    """Last available observation in each calendar month, indexed by month."""
    return s.resample("ME").last().to_period("M")


def _log_ret(levels):
    return 100 * np.log(levels).diff()


def build_monthly(raw):
    """Monthly panel: log returns (%), EM ex-IT, IT weight, crisis flags, controls and risk-free rate."""
    lv = pd.DataFrame({k: _month_end(raw[k]) for k in PRICE_FILES})
    m = _log_ret(lv)

    # IT weight in MSCI EM at the end of month t-1: market value of EM IT / market value of EM
    w_it = (_month_end(raw["MCAP_EM_IT"]) / _month_end(raw["MCAP_EM"])).reindex(m.index)
    m["W_IT"] = w_it
    m["W_IT_LAG"] = w_it.shift(1)

    # EM ex-IT backed out with the IT weight: R_EM = w R_IT + (1 - w) R_exIT (simple returns)
    R_em, R_it = np.exp(m["EM"] / 100) - 1, np.exp(m["EM_IT"] / 100) - 1
    w = m["W_IT_LAG"]
    m["EM_EXIT"] = 100 * np.log(1 + (R_em - w * R_it) / (1 - w))

    # Fin_t: MSCI World more than 10% below its running peak (month-end levels, history from 1990)
    world = lv["WORLD"]
    m["WORLD_DD"] = world / world.cummax() - 1
    m["Fin"] = (m["WORLD_DD"] < C.FIN_DRAWDOWN).astype(int)

    # controls: change in VIX (points) and in the dollar index (log %), month end to month end
    m["dVIX"] = _month_end(raw["VIX"]).diff()
    m["dDXY"] = _log_ret(_month_end(raw["DXY"]))
    # risk-free: 3-month T-bill yield at the end of month t-1, as a simple monthly return (decimal)
    m["RF"] = _month_end(raw["UST3M"]).shift(1) / 1200

    m = m.loc[C.SAMPLE_START:C.SAMPLE_END].copy()

    # Geo_t: GPR or TPU in its top decile, decile computed over the study sample
    gpr, tpu = raw["GPR"].reindex(m.index), raw["TPU"].reindex(m.index)
    m["GPR"], m["TPU"] = gpr, tpu
    m["Geo"] = ((gpr >= gpr.quantile(C.GEO_QUANTILE)) | (tpu >= tpu.quantile(C.GEO_QUANTILE))).astype(int)
    m["Crisis"] = ((m["Fin"] == 1) | (m["Geo"] == 1)).astype(int)
    m["Current"] = (m.index >= pd.Period(C.CURRENT_START, "M")).astype(int)

    need = ["EM", "EM_PX", "EM_LC", "EMSC", "EM_IT", "US", "US_IT", "WORLD", "W_IT_LAG", "EM_EXIT",
            "dVIX", "dDXY", "RF", "GPR", "TPU"]
    missing = m[need].isna().sum()
    assert missing.sum() == 0, f"missing monthly values:\n{missing[missing > 0]}"
    return m


def build_weekly(raw, cols=("EM", "EMSC", "US", "US_IT", "EMSC_NET")):
    """Weekly (Friday) log returns (%)."""
    lv = pd.DataFrame({k: raw[k].resample("W-FRI").last() for k in cols})
    w = _log_ret(lv)
    start = pd.Period(C.SAMPLE_START, "M").start_time
    return w.loc[w.index >= start]


def build_daily(raw, cols=("EM", "WORLD")):
    """Daily log returns (%) on the dates where all requested series are quoted."""
    lv = pd.concat({k: raw[k] for k in cols}, axis=1, join="inner")
    d = _log_ret(lv).dropna()
    return d.loc[d.index >= pd.Period(C.SAMPLE_START, "M").start_time]
