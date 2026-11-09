"""Module A: how EM's composition changed and what it means for EM's link to the U.S. (Question 1).

| Section | Question | Output |
|---|---|---|
| 0 | Data | T0 |
| 1 | How has EM's composition changed? | Fig 1-2, T1 |
| 2 | Do EM's risk and return come more and more from tech? | Fig 3, T2-T3 |
| 3 | Has EM's link to the U.S. changed, and is tech the channel? | Fig 4-5, T4-T5 |
| 4 | Which countries carry EM's link to U.S. tech, and how has that changed? | Fig 6-7, T6-T7 |
| 5 | Save | Excel |

Simple returns to build series and for risk and return; log returns ln(1 + r) for correlations, betas and the
trend regression. Weights are last month-end values; a month is left empty if any input is missing.
Time is shown as continuous series, 60-month rolling windows, and five non-overlapping blocks.
No turning point is imposed here; formal tests of change are in Module C.
"""
from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .q1_data import BLOCKS, BLUE, COUNTRIES, END, FULL, GOLD, GREY, NAVY, RED, panel  # noqa: E402

ROLL = 60
OUT = C.OUT_DIR / "q1" / "module_A"


def _save(name):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.close("all")


def corr_beta(df, s, b):
    d = df[[s, b]].dropna()
    return pd.Series({"corr": d[s].corr(d[b]), "beta": d[s].cov(d[b]) / d[b].var(), "months": len(d)})


def perf(r, rf_):
    return pd.DataFrame({"ann. return %": ((1 + r).prod() ** (12 / r.count()) - 1) * 100, "ann. vol %": r.std() * np.sqrt(12) * 100,
                         "Sharpe": r.sub(rf_, axis=0).mean() / r.std() * np.sqrt(12)})


def run():
    plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False})
    p = panel()
    ret, lret, W_ALL, TS_ALL, rf, ex = p["ret"], p["lret"], p["W_ALL"], p["TS_ALL"], p["rf"], p["ex"]
    tech_share, W = p["tech_share"], p["W"]
    results = {}

    # ---------------------------------------------------------------- 0. data
    t0 = pd.DataFrame({"returns from": ret.apply(lambda s: s.first_valid_index().strftime("%Y-%m")), "months": ret.count()})
    results["T0_coverage"] = t0

    # ---------------------------------------------------------------- 1. composition
    # Tech share = MSCI EM IT market cap / MSCI EM market cap. In Dec 2018 GICS moved internet firms out of IT;
    # the adjusted line multiplies pre-Dec-2018 values by the Dec/Nov-2018 ratio (approximation).
    factor = tech_share["2018-12"].iloc[0] / tech_share["2018-11"].iloc[0]
    tech_adj = tech_share.where(tech_share.index >= "2018-12-31", tech_share * factor)
    ax = (tech_share * 100).plot(color=NAVY, lw=2, label="raw")
    (tech_adj * 100).plot(ax=ax, color=GREY, ls="--", lw=1.5, label="adjusted to post-2018 GICS")
    ax.axvline(pd.Timestamp("2018-12-31"), color=GREY, ls=":", lw=1)
    ax.set(title="Fig 1. IT share of MSCI EM (%)", xlabel=""); ax.legend(frameon=False)
    _save("fig1_tech_share.png")

    ax = None
    for c, col in zip(["Taiwan", "Korea", "China", "India"], [GOLD, NAVY, RED, BLUE]):
        ax = (W[c] * 100).plot(ax=ax, color=col, lw=2, label=c)
    (W.drop(columns=["Taiwan", "Korea", "China", "India"]) * 100).plot(ax=ax, color=GREY, lw=1, legend=False)
    ax.set(title="Fig 2. Country weights in MSCI EM (%), grey = Brazil, South Africa, Mexico, Saudi Arabia", xlabel="")
    ax.legend(frameon=False)
    _save("fig2_country_weights.png")

    t1 = pd.concat([tech_share.rename("IT share"), tech_adj.rename("IT share (adj.)"), W, W.sum(axis=1).rename("8 countries")], axis=1)
    t1 = (t1.resample("YE").last() * 100).round(1); t1.index = t1.index.year
    results["T1_composition_yearend"] = t1

    # ---------------------------------------------------------------- 2. risk and return
    # IT share of EM risk (60-month rolling) = IT weight x Cov(r_IT, r_EM) / Var(r_EM), simple returns.
    w_it = TS_ALL.shift(1).reindex(ret.index)
    risk = pd.DataFrame({"IT weight %": w_it,
                         "IT risk share %": w_it * ret["EM_IT"].rolling(ROLL).cov(ret["EM"]) / ret["EM"].rolling(ROLL).var()}) * 100
    ax = risk["IT weight %"].plot(color=NAVY, lw=2, label="IT weight in EM")
    risk["IT risk share %"].plot(ax=ax, color=GOLD, lw=2, label=f"IT share of EM risk ({ROLL}m)")
    ax.set(title="Fig 3. IT weight vs. IT share of EM risk (%)", xlabel=""); ax.legend(frameon=False)
    _save("fig3_risk_share.png")

    def block_risk(a, z):
        r = ret.loc[a:z]; w = w_it.loc[a:z].mean()
        return pd.Series({"IT weight %": w * 100, "IT risk share %": w * r["EM_IT"].cov(r["EM"]) / r["EM"].var() * 100})
    t2 = pd.DataFrame({k: block_risk(a, z) for k, (a, z) in {**BLOCKS, **FULL}.items()}).T
    t2["risk − weight (pp)"] = t2["IT risk share %"] - t2["IT weight %"]; t2 = t2.round(1)

    COLS = ["EM", "EM_IT", "EM_exIT", "EM_exTWKR", "USA"]
    t3 = pd.concat({k: perf(ret.loc[a:z, COLS], rf.loc[a:z]) for k, (a, z) in {**BLOCKS, **FULL}.items()}).round(2)
    results["T2_risk_share_by_block"] = t2; results["T3_performance_by_block"] = t3

    # ---------------------------------------------------------------- 3. link to the U.S.
    def roll_cb(s, b):
        return lret[s].rolling(ROLL).corr(lret[b]), lret[s].rolling(ROLL).cov(lret[b]) / lret[b].rolling(ROLL).var()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for b, col in [("USA", NAVY), ("USA_IT", GOLD)]:
        c, be = roll_cb("EM", b)
        c.plot(ax=axes[0], color=col, lw=2, label=f"EM vs {b}"); be.plot(ax=axes[1], color=col, lw=2, label=f"EM vs {b}")
    axes[0].set(title=f"Fig 4a. {ROLL}-month correlation", xlabel=""); axes[1].set(title=f"Fig 4b. {ROLL}-month beta", xlabel="")
    axes[0].legend(frameon=False)
    _save("fig4_em_us_link.png")

    ax = None
    for s, col in [("EM_IT", GOLD), ("EM_exIT", GREY), ("EM_exTWKR", NAVY)]:
        ax = roll_cb(s, "USA_IT")[0].plot(ax=ax, color=col, lw=2, label=s)
    ax.set(title=f"Fig 5. {ROLL}-month correlation with U.S. tech: EM tech vs. EM without tech", xlabel=""); ax.legend(frameon=False)
    _save("fig5_tech_vs_nontech.png")

    SERIES = ["EM", "EM_IT", "EM_exIT", "EM_exTWKR"]
    t4 = pd.concat({(b, k): pd.DataFrame({s: corr_beta(lret.loc[a:z], s, b) for s in SERIES}).T[["corr", "beta"]]
                    for b in ["USA", "USA_IT"] for k, (a, z) in {**BLOCKS, **FULL}.items()}, axis=1).round(2)

    def trend(s, b, lags=12):
        """r = a + (b0 + b1 t) r_bench, t in decades since 2001-01, Newey-West (12 lags)."""
        d = pd.DataFrame({"y": lret[s], "x": lret[b]}).dropna()
        t = (np.arange(len(lret)) / 120)[lret.index.isin(d.index)]
        X = np.column_stack([np.ones(len(d)), d["x"], d["x"] * t]); y = d["y"].values
        XtX_inv = np.linalg.inv(X.T @ X); bhat = XtX_inv @ X.T @ y; u = y - X @ bhat
        S = (X * u[:, None]).T @ (X * u[:, None])
        for L in range(1, lags + 1):
            g = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None]); S += (1 - L / (lags + 1)) * (g + g.T)
        se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv)); tval = bhat[2] / se[2]
        pv = 2 * (1 - 0.5 * (1 + erf(abs(tval) / sqrt(2))))
        return pd.Series({"beta at 2001": bhat[1], "change per decade": bhat[2], "t-stat": tval, "p-value": pv,
                          "implied beta 2026-08": bhat[1] + bhat[2] * t[-1], "months": len(d)})
    t5 = pd.DataFrame({f"{s} vs {b}": trend(s, b) for b in ["USA", "USA_IT"] for s in SERIES}).T.round(3)
    results["T4_link_by_block"] = t4; results["T5_trend_regression"] = t5

    # ---------------------------------------------------------------- 4. countries
    heat = pd.DataFrame({c: lret[c].rolling(ROLL).corr(lret["USA_IT"]) for c in COUNTRIES + ["EM"]}).resample("YE").last()
    heat.index = heat.index.year; heat = heat.dropna(how="all")
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(heat.T.values, aspect="auto", cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_yticks(range(heat.shape[1])); ax.set_yticklabels(heat.columns)
    ax.set_xticks(range(len(heat))); ax.set_xticklabels(heat.index, rotation=90, fontsize=8)
    for i in range(heat.shape[1]):
        for j in range(len(heat)):
            if pd.notna(heat.iloc[j, i]):
                ax.text(j, i, f"{heat.iloc[j, i]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"Fig 6. {ROLL}-month correlation with MSCI USA IT, year-end (blank = not enough data)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    _save("fig6_country_heatmap.png")

    def loo(a, z, parts):
        e = np.log1p(ex(parts, W_ALL)).loc[a:z]; r = lret.loc[a:z]; m = e.notna() & r["EM"].notna()
        if m.sum() < 24:
            return np.nan
        return r.loc[m, "EM"].corr(r.loc[m, "USA_IT"]) - e[m].corr(r.loc[m, "USA_IT"])

    t6 = pd.DataFrame({k: {c: loo(a, z, [c]) for c in COUNTRIES} for k, (a, z) in {**BLOCKS, **FULL}.items()}).round(3)
    ax = t6[list(BLOCKS)].T.plot.bar(color=[GOLD, NAVY, RED, BLUE, "#27AE60", "#8E44AD", "#16A085", "#7F8C8D"], width=0.8, figsize=(12, 4))
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.set(title="Fig 7. Drop in EM’s correlation with U.S. tech when one country is removed, by block", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    _save("fig7_leave_one_out.png")

    t7 = pd.DataFrame({k: {"Taiwan only": loo(a, z, ["Taiwan"]), "Korea only": loo(a, z, ["Korea"]),
                           "both together": loo(a, z, ["Taiwan", "Korea"])} for k, (a, z) in {**BLOCKS, **FULL}.items()})
    t7.loc["sum of the two"] = t7.loc["Taiwan only"] + t7.loc["Korea only"]; t7 = t7.round(3)
    results["T6_leave_one_out_by_block"] = t6; results["T7_taiwan_korea_by_block"] = t7

    # ---------------------------------------------------------------- 5. save
    results = {"monthly_data": ret.join(tech_share.rename("tech_share")).join(W.add_prefix("w_")).join(rf.rename("rf")),
               "monthly_data_log": lret, **results}
    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "module_A_results.xlsx") as xl:
        for name, df in results.items():
            df.to_excel(xl, sheet_name=name[:31])
    handoff = pd.concat([tech_share.rename("tech_share"), ret[["EM_exIT", "EM_exTWKR"]],
                         lret[["EM_exIT", "EM_exTWKR"]].add_suffix("_log")], axis=1)
    readme = pd.DataFrame({"definition": {
        "tech_share": "MSCI EM IT index market cap / MSCI EM index market cap, month end (raw; GICS change Dec 2018)",
        "EM_exIT": "(r_EM - w*r_EM_IT) / (1 - w), w = last month tech_share; simple returns",
        "EM_exTWKR": "(r_EM - w_TW*r_TW - w_KR*r_KR) / (1 - w_TW - w_KR), last month country weights; simple returns",
        "*_log": "ln(1 + r); use for correlations, betas, regressions and tests",
        "missing": "a month is empty if any input is missing (EM_exTWKR starts 2001-02)",
        "version": f"v4, data to {END}"}})
    with pd.ExcelWriter(OUT / "module_A_handoff.xlsx") as xl:
        handoff.to_excel(xl, sheet_name="data"); readme.to_excel(xl, sheet_name="README")
    return results
