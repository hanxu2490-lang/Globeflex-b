"""Module A tests: are the main findings of Module A and A3 beyond sampling noise? (Question 1)

| Part | Tests | Module tested |
|---|---|---|
| 1 | T1-T8: Taiwan & Korea, China, non-tech EM and tech risk in the latest five-year block | Module A |
| 2 | O1: is 10% EM ex-IT worse than 10% MSCI EM? O2: how stable is the optimizer's country choice? | Module A3 |

Moving block bootstrap: 12-month blocks, the same months drawn for every series, resampling within each block or
window (blocks are never mixed), 2,000 draws (1,000 for O2), fixed seed. Percentile 95% CIs. One-sided
p = (draws on the wrong side of zero + 1) / (B + 1). Holm adjustment over T1-T8. Directions were set after seeing
Module A's descriptive results, so this checks those results rather than testing independent hypotheses.
The significance of A3's Sharpe gains (GRS test and bootstrap) is in Module C, section 4c.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from .q1_data import BLOCKS, COUNTRIES, END, panel

LAST, PREV = "2021–26", "2016–20"
BLOCK, B, B_OPT, SEED, ALPHA = 12, 2000, 1000, 0, 0.05
OUT = C.OUT_DIR / "q1" / "module_A_tests"


def block_idx(n, rng):
    st = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
    return (st[:, None] + np.arange(BLOCK)).ravel()[:n]


def boot(df, stat, draws, seed):
    rng = np.random.default_rng(seed)
    return np.array([stat(df.iloc[block_idx(len(df), rng)]) for _ in range(draws)])


def pval(draws, sign):
    return ((draws <= 0).sum() + 1) / (len(draws) + 1) if sign > 0 else ((draws >= 0).sum() + 1) / (len(draws) + 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def corr(d, s):
    return d[s].corr(d["USA_IT"])


def loo(col):
    return lambda d: corr(d, "EM") - corr(d, col)


def run():
    p = panel()
    ret = pd.concat([p["ret"][["EM", "EM_IT", "USA_IT", "EM_exIT", *COUNTRIES]], p["us"]], axis=1)
    W_ALL, ex = p["W_ALL"], p["ex"]
    rf = p["rf"]
    results = {}
    ret["w_IT"] = p["TS_ALL"].shift(1).reindex(ret.index)
    # log-return panel for correlations (series built in simple returns, then converted)
    L = np.log1p(pd.DataFrame({"EM": ret["EM"], "USA_IT": ret["USA_IT"], "exIT": ret["EM_exIT"],
                               "exTWKR": ex(["Taiwan", "Korea"], W_ALL), "exTW": ex(["Taiwan"], W_ALL),
                               "exKR": ex(["Korea"], W_ALL), "exCN": ex(["China"], W_ALL)})).dropna()
    S = ret[["EM", "EM_IT", "w_IT"]].dropna()                         # simple-return panel for the risk share

    # ---------------------------------------------------------------- part 1: Module A, latest five-year block
    def risk_gap(d):
        w = d["w_IT"].mean()
        return w * d["EM_IT"].cov(d["EM"]) / d["EM"].var() - w

    a1, z1 = BLOCKS[LAST]; a0, z0 = BLOCKS[PREV]
    Ld, Lp, Sd = L.loc[a1:z1], L.loc[a0:z0], S.loc[a1:z1]
    single = {"T1": (Ld, loo("exTWKR"), 1), "T2": (Ld, loo("exTW"), 1), "T3": (Ld, loo("exKR"), 1), "T4": (Ld, loo("exCN"), -1),
              "T7": (Ld, loo("exIT"), 1), "T8": (Sd, risk_gap, 1)}
    rows = {}
    for k, (d, f, sgn) in single.items():
        dr = boot(d, f, B, SEED)
        rows[k] = {"estimate": f(d), "2.5%": np.quantile(dr, .025), "97.5%": np.quantile(dr, .975),
                   "expected": "> 0" if sgn > 0 else "< 0", "p one-sided": pval(dr, sgn)}
    for k, f, sgn in [("T5", loo("exTWKR"), 1), ("T6", lambda d: corr(d, "exIT"), -1)]:
        dr = boot(Ld, f, B, SEED) - boot(Lp, f, B, SEED + 1)
        rows[k] = {"estimate": f(Ld) - f(Lp), "2.5%": np.quantile(dr, .025), "97.5%": np.quantile(dr, .975),
                   "expected": "> 0" if sgn > 0 else "< 0", "p one-sided": pval(dr, sgn)}
    labels = {"T1": "remove Taiwan + Korea, 2021–26", "T2": "remove Taiwan, 2021–26", "T3": "remove Korea, 2021–26",
              "T4": "remove China, 2021–26", "T5": "T1 change vs 2016–20", "T6": "Corr(EM ex-IT, US tech) change vs 2016–20",
              "T7": "gap EM − EM ex-IT, 2021–26", "T8": "IT risk share − IT weight, 2021–26"}
    p1 = pd.DataFrame(rows).T.loc[[f"T{i}" for i in range(1, 9)]]
    p1.insert(0, "statistic", [labels[i] for i in p1.index])
    p1["p Holm"] = holm(p1["p one-sided"].astype(float)); p1["significant 5% (Holm)"] = p1["p Holm"] < ALPHA
    for c in ["estimate", "2.5%", "97.5%", "p one-sided", "p Holm"]:
        p1[c] = p1[c].astype(float).round(4)
    results["P1_main_tests"] = p1

    # supplementary (CIs only, no Holm): removing Taiwan + Korea together in every block
    sup = {}
    for k, (a, z) in BLOCKS.items():
        d = L.loc[a:z]; dr = boot(d, loo("exTWKR"), B, SEED)
        sup[k] = {"estimate": loo("exTWKR")(d), "2.5%": np.quantile(dr, .025), "97.5%": np.quantile(dr, .975), "months": len(d)}
    results["P1b_twkr_every_block"] = pd.DataFrame(sup).T.round(4)

    # ---------------------------------------------------------------- part 2: Module A3
    def fixed10_sharpe(d, v, x=0.10, grid=451):
        r = d[["S&P 500", "US Agg", v]]; mu = r.mean().values * 12; Sg = r.cov().values * 12; r_f = d["rf"].mean() * 12
        s = np.linspace(0, 1 - x, grid); Wm = np.column_stack([s, 1 - x - s, np.full_like(s, x)])
        return np.max((Wm @ mu - r_f) / np.sqrt(np.einsum("ij,jk,ik->i", Wm, Sg, Wm)))

    P = ret[["S&P 500", "US Agg", "EM", "EM_exIT"]].rename(columns={"EM": "MSCI EM"}).join(rf.rename("rf")).dropna()
    O1_WIN = {"5-year 2021-09…2026-08": ("2021-09", END), "10-year 2016-09…2026-08": ("2016-09", END)}
    o1 = {}
    for k, (a, z) in O1_WIN.items():
        d = P.loc[a:z]
        f = lambda dd: fixed10_sharpe(dd, "MSCI EM") - fixed10_sharpe(dd, "EM_exIT")
        dr = boot(d, f, B, SEED)
        o1[k] = {"Sharpe 10% MSCI EM": fixed10_sharpe(d, "MSCI EM"), "Sharpe 10% EM ex-IT": fixed10_sharpe(d, "EM_exIT"),
                 "difference": f(d), "2.5%": np.quantile(dr, .025), "97.5%": np.quantile(dr, .975), "p one-sided (> 0)": pval(dr, 1)}
    results["P2_O1_exIT_vs_EM"] = pd.DataFrame(o1).T.round(4)

    ASSETS = ["S&P 500", "US Agg"] + COUNTRIES
    Q = ret[ASSETS].join(rf.rename("rf"))

    def max_sharpe_w(d):
        r = d[ASSETS]; mu = r.mean().values * 12; Sg = r.cov().values * 12; r_f = d["rf"].mean() * 12; n = len(mu)
        res = minimize(lambda w: -(w @ mu - r_f) / np.sqrt(w @ Sg @ w), np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-10, "maxiter": 500})
        return res.x

    def min_var_w(d):
        Sg = d[ASSETS].cov().values * 12; n = Sg.shape[0]
        res = minimize(lambda w: w @ Sg @ w, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-12, "maxiter": 500})
        return res.x

    O2_WIN = {"5-year 2021-09…2026-08": ("2021-09", END), "full 2015-04…2026-08": ("2015-04", END)}
    o2 = {}
    for k, (a, z) in O2_WIN.items():
        d = Q.loc[a:z].dropna(); rng = np.random.default_rng(SEED)
        for kind, fn in [("max Sharpe", max_sharpe_w), ("min variance", min_var_w)]:
            point = pd.Series(fn(d), ASSETS) * 100
            draws = np.array([fn(d.iloc[block_idx(len(d), rng)]) for _ in range(B_OPT)]) * 100
            o2[(k, kind)] = pd.DataFrame({"weight in sample %": point, "chosen in % of draws (>1%)": (draws > 1).mean(axis=0) * 100,
                                          "mean weight %": draws.mean(axis=0), "5%": np.quantile(draws, .05, axis=0),
                                          "95%": np.quantile(draws, .95, axis=0)}, index=ASSETS)
    results["P2_O2_country_stability"] = pd.concat(o2).round(1)

    # ---------------------------------------------------------------- save
    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "module_A_tests.xlsx") as xl:
        pd.DataFrame({"setting": {"method": f"moving block bootstrap, {BLOCK}-month blocks, within block/window, B={B} (O2: {B_OPT}), seed={SEED}",
                                  "blocks": ", ".join(BLOCKS), "p-value": "one-sided, (wrong-side draws + 1)/(B + 1); Holm over T1–T8",
                                  "returns": "log returns for correlations; simple returns for risk share and optimizer",
                                  "not repeated": "Sharpe-gain significance of A3 is in Module C section 4c"}}).to_excel(xl, sheet_name="README")
        for name, df in results.items():
            df.to_excel(xl, sheet_name=name[:31])
    return results
