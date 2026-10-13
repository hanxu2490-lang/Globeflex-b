"""Step 2 - Is EM more like the US, and is technology the cause? (Question 1).

  r_EM,t = alpha + beta * r_US,t + delta * (Post_t x r_US,t) + e_t                  (1)

Post_t = 1 after the Bai-Perron structural break; delta > 0 means EM has become more sensitive
to US returns. Post_t is then replaced by w_IT,t-1 (IT weight in MSCI EM). Re-estimated on
EM ex-IT and EM Small Cap, and on non-crisis months only. OLS with Newey-West errors.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import utils as U
from .data_prep import LABELS

# dependent variables: EM, EM ex-IT, EM Small Cap (price) and EM (price) as its same-basis comparator
DEPENDENTS = ["EM", "EM_EXIT", "EMSC", "EM_PX"]


# ------------------------------------------------------------------ Bai-Perron
def _segment_ssr(y, X, h):
    """SSR of the OLS fit on every segment [i, j] (inclusive) at least h long."""
    T = len(y)
    ssr = np.full((T, T), np.inf)
    for i in range(T):
        for j in range(i + h - 1, T):
            Xi, yi = X[i:j + 1], y[i:j + 1]
            b, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            e = yi - Xi @ b
            ssr[i, j] = e @ e
    return ssr


def bai_perron(y, X, trim=C.BP_TRIM, max_breaks=C.BP_MAX_BREAKS):
    """Global SSR-minimizing break dates for m = 0..max_breaks (Bai & Perron 1998, 2003).

    All coefficients may change at each break. Returns, for each m, the SSR, the BIC and the
    break positions (index of the last observation of each regime), plus supF(1).
    """
    y, X = np.asarray(y, float), np.asarray(X, float)
    T, q = X.shape
    h = int(np.floor(trim * T))
    ssr = _segment_ssr(y, X, h)
    # opt[m][j]: minimal SSR of y[0..j] with m breaks; arg[m][j]: end of the previous regime
    opt = [ssr[0].copy()]
    arg = [np.full(T, -1)]
    for m in range(1, max_breaks + 1):
        o, a = np.full(T, np.inf), np.full(T, -1)
        for j in range((m + 1) * h - 1, T):
            ks = np.arange(m * h - 1, j - h + 1)
            vals = opt[m - 1][ks] + ssr[ks + 1, j]
            if len(vals):
                o[j], a[j] = vals.min(), ks[vals.argmin()]
        opt.append(o)
        arg.append(a)
    rows = []
    for m in range(max_breaks + 1):
        if not np.isfinite(opt[m][T - 1]):
            break
        brk, j = [], T - 1
        for mm in range(m, 0, -1):
            j = arg[mm][j]
            brk.append(int(j))
        brk = sorted(brk)
        S = opt[m][T - 1]
        p = (m + 1) * q + m
        rows.append({"breaks": m, "ssr": S, "bic": np.log(S / T) + p * np.log(T) / T, "break_pos": brk})
    tab = pd.DataFrame(rows)
    ssr0, ssr1 = tab.loc[0, "ssr"], tab.loc[1, "ssr"]
    supf1 = (T - 2 * q) / q * (ssr0 - ssr1) / ssr1
    return tab, supf1


# ------------------------------------------------------------------ regressions
def beta_regression(m, dep, inter):
    """Equation (1) with interaction variable `inter` (Post or W_IT_LAG)."""
    X = pd.DataFrame({"r_US": m["US"], "inter_x_r_US": m[inter] * m["US"]})
    return U.ols_nw(m[dep], X)


def regression_table(m, dependents=DEPENDENTS, samples=None):
    samples = samples or {"All months": m.index, "Non-crisis months": m.index[m["Crisis"] == 0]}
    rows = []
    for dep in dependents:
        for inter, inter_lab in [("Post", "Post"), ("W_IT_LAG", "w_IT,t-1")]:
            for s_lab, idx in samples.items():
                d = m.loc[idx]
                if d[inter].nunique() < 2:          # interaction collinear with r_US
                    continue
                res = beta_regression(d, dep, inter)
                rows.append({
                    "dependent": LABELS[dep], "interaction": inter_lab, "sample": s_lab, "N": int(res.nobs),
                    "alpha": res.params["const"], "beta": res.params["r_US"], "t_beta": res.tvalues["r_US"],
                    "delta": res.params["inter_x_r_US"], "t_delta": res.tvalues["inter_x_r_US"],
                    "p_delta": res.pvalues["inter_x_r_US"], "R2": res.rsquared,
                })
    return pd.DataFrame(rows)


def run(m):
    bp_tab, supf1 = bai_perron(m["EM"].values, np.column_stack([np.ones(len(m)), m["US"].values]))
    idx = m.index
    bp_tab["break_months"] = bp_tab["break_pos"].apply(lambda b: ", ".join(str(idx[i]) for i in b))
    bp_tab = bp_tab.drop(columns="break_pos")
    bp_tab["bic_choice"] = np.where(bp_tab["bic"] == bp_tab["bic"].min(), "<- min BIC", "")
    U.save_table(bp_tab, "step2_bai_perron", index=False)

    # Post_t is defined by the single-break model (m = 1)
    one_break = bp_tab.loc[bp_tab["breaks"] == 1, "break_months"].iloc[0]
    break_month = pd.Period(one_break, "M")
    m = m.copy()
    m["Post"] = (m.index > break_month).astype(int)
    test = pd.DataFrame({"supF(1)": [supf1], "5% critical value": [C.BP_SUPF_CV5],
                         "break month (last month of regime 1)": [str(break_month)],
                         "first Post month": [str(break_month + 1)]})
    U.save_table(test, "step2_bai_perron_supF", index=False)

    reg = regression_table(m)
    U.save_table(reg, "step2_regressions", index=False)
    return {"panel": m, "break_month": break_month, "bp": bp_tab, "supF": supf1, "regressions": reg}
