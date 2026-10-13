"""Robustness (methods page).

1. Small Cap uses the price index (the net-return version starts in 2010), so post-2010 results are
   re-checked with net returns (Steps 1, 2 and 4).
2. Local-currency EM returns in the crisis regression (Step 3): how much of EM's crisis performance
   comes from currency moves.
3. 2022 as the start of the current period (Step 3).
"""
import pandas as pd

from . import config as C
from . import utils as U
from . import step1_dcc, step2_beta_tech, step3_crisis, step4_portfolio


def emsc_net_post2010(m, weekly):
    # Step 1: DCC over the post-2010 weeks, Small Cap price vs net
    wk = weekly.dropna(subset=["EMSC_NET"])
    rows = {}
    for col, lab in [("EMSC", "EM Small Cap (price)"), ("EMSC_NET", "EM Small Cap (net)")]:
        r = wk[["EM", col, "US", "US_IT"]].rename(columns={col: "EMSC"})
        corr = step1_dcc.fit_dcc(r)["corr"]
        rows[lab] = {"weeks": len(corr), "mean corr vs USA": corr["EMSC~US"].mean(),
                     "mean corr vs USA IT": corr["EMSC~US_IT"].mean()}
    dcc = pd.DataFrame(rows).T
    U.save_table(dcc, "robust1_post2010_dcc_emsc_price_vs_net")

    # Step 2: regressions on the post-2010 months
    mm = m.loc[C.POST2010_START:]
    samples = {"All months": mm.index, "Non-crisis months": mm.index[mm["Crisis"] == 0]}
    reg = step2_beta_tech.regression_table(mm, ["EMSC_NET", "EMSC", "EM", "EM_PX"], samples)
    U.save_table(reg, "robust1_post2010_regressions", index=False)

    # Step 4: out of sample once 60 months of net data exist
    oos = (pd.Period(C.POST2010_START, "M") + C.ROLL_WINDOW)
    perf, _, _ = step4_portfolio.evaluate(m, ["EMSC_NET", "EMSC", "EM", "EM_PX"], start=str(oos))
    U.save_table(perf, "robust1_post2010_portfolio")
    return {"dcc": dcc, "regressions": reg, "portfolio": perf, "oos_start": str(oos)}


def local_currency(m):
    cols, pvals = {}, {}
    for dep in ("EM", "EM_LC"):
        for controls in (False, True):
            name = f"{'EM (USD)' if dep == 'EM' else 'EM (local ccy)'} - World, {'with' if controls else 'no'} X"
            cols[name], res = step3_crisis.crisis_column(m, dep, controls)
            pvals[name] = res.pvalues.rename(step3_crisis.COEF_NAMES)
    tab = pd.DataFrame(cols)
    U.save_table(tab, "robust2_local_currency_crisis")
    U.save_table(pd.DataFrame(pvals), "robust2_local_currency_crisis_pvalues")
    return tab


def current_from_2022(m):
    tab, pvals = step3_crisis.regression_table(m, current_start=C.CURRENT_START_ROBUST)
    U.save_table(tab, "robust3_current_from_2022_crisis")
    U.save_table(pvals, "robust3_current_from_2022_crisis_pvalues")
    return tab


def run(m, weekly):
    return {"emsc_net": emsc_net_post2010(m, weekly), "local_ccy": local_currency(m),
            "current_2022": current_from_2022(m)}
