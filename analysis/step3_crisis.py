"""Step 3 - Crisis behaviour today versus the past (Question 2).

  r_EM,t - r_DM,t = alpha + g1 Fin_t + g2 Geo_t + g3 (Fin_t x Current_t) + g4 (Geo_t x Current_t)
                    + theta Post_t + phi' X_t + e_t                                          (2)

Fin_t = 1 when MSCI World is more than 10% below its peak; Geo_t = 1 when GPR or TPU is in its
top decile; Current_t = 1 from 2018. X_t = changes in the VIX and the dollar index (reported with
and without). Repeated for EM ex-IT. Daily cumulative EM-minus-DM returns from 5 days before to
20 days after each named crisis.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import utils as U
from .data_prep import LABELS

COEF_NAMES = {"Fin": "g1 Fin", "Geo": "g2 Geo", "Fin_x_Current": "g3 Fin x Current",
              "Geo_x_Current": "g4 Geo x Current", "Post": "theta Post", "dVIX": "phi dVIX",
              "dDXY": "phi dDXY", "const": "alpha"}


def crisis_regression(m, dep, controls, current_start=C.CURRENT_START, dm="WORLD"):
    cur = (m.index >= pd.Period(current_start, "M")).astype(int)
    X = pd.DataFrame({"Fin": m["Fin"], "Geo": m["Geo"], "Fin_x_Current": m["Fin"] * cur,
                      "Geo_x_Current": m["Geo"] * cur, "Post": m["Post"]}, index=m.index)
    if controls:
        X[["dVIX", "dDXY"]] = m[["dVIX", "dDXY"]]
    return U.ols_nw(m[dep] - m[dm], X), cur


def crisis_column(m, dep, controls, current_start=C.CURRENT_START):
    """One results column: coefficients (t-stats), current-crisis totals and month counts."""
    res, cur = crisis_regression(m, dep, controls, current_start)
    col = {}
    for k, lab in COEF_NAMES.items():
        if k in res.params:
            col[lab] = res.params[k]
            col[f"t: {lab}"] = res.tvalues[k]
    for lab, names in [("g1+g3 (current Fin)", ["Fin", "Fin_x_Current"]),
                       ("g2+g4 (current Geo)", ["Geo", "Geo_x_Current"])]:
        est, t, p = U.lincom(res, names)
        col[lab], col[f"t: {lab}"] = est, t
    col["R2"], col["N"] = res.rsquared, res.nobs
    col["Fin months, past"] = int(((m["Fin"] == 1) & (cur == 0)).sum())
    col["Fin months, current"] = int(((m["Fin"] == 1) & (cur == 1)).sum())
    col["Geo months, past"] = int(((m["Geo"] == 1) & (cur == 0)).sum())
    col["Geo months, current"] = int(((m["Geo"] == 1) & (cur == 1)).sum())
    return col, res


def regression_table(m, deps=("EM", "EM_EXIT"), current_start=C.CURRENT_START):
    cols, pvals = {}, {}
    for dep in deps:
        for controls in (False, True):
            name = f"{LABELS[dep]} - World, {'with' if controls else 'no'} X"
            cols[name], res = crisis_column(m, dep, controls, current_start)
            pvals[name] = res.pvalues.rename(COEF_NAMES)
    return pd.DataFrame(cols), pd.DataFrame(pvals)


# ------------------------------------------------------------------ daily event study
def event_study(daily, crises=C.CRISES, pre=C.EVENT_PRE, post=C.EVENT_POST):
    """Cumulative EM-minus-World log return (%) from day -pre to each day up to +post."""
    rel = daily["EM"] - daily["WORLD"]
    dates = rel.index
    paths, info = {}, []
    for name, date, kind, period in crises:
        i0 = dates.searchsorted(pd.Timestamp(date))      # first trading day on/after the date
        win = rel.iloc[i0 - pre:i0 + post + 1]
        paths[name] = pd.Series(win.cumsum().values, index=range(-pre, post + 1))
        info.append({"crisis": name, "type": kind, "period": period, "day0": dates[i0].date()})
    paths = pd.DataFrame(paths)
    info = pd.DataFrame(info).set_index("crisis")
    horizons = [-1, 0, 5, 10, 20]
    tab = info.join(paths.loc[horizons].T.rename(columns=lambda h: f"CAR[-{pre},{h:+d}]"))
    groups = info["period"] + " " + info["type"]
    group_paths = paths.T.groupby(groups).mean().T
    group_tab = tab.groupby(groups)[[c for c in tab.columns if c.startswith("CAR")]].mean()
    group_tab.insert(0, "n events", groups.value_counts())
    return paths, tab, group_paths, group_tab


def plot_events(group_paths, name):
    panels = [("Fin", "Financial crises"), ("Geo", "War / geopolitical crises"), ("Trade", "Trade conflicts")]
    fig, axes = U.new_fig(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (kind, title) in zip(axes, panels):
        for k, period in enumerate(["Past", "Current"]):
            col = f"{period} {kind}"
            if col in group_paths:
                ax.plot(group_paths.index, group_paths[col], color=U.SERIES[k], lw=1.8,
                        label=f"{period} ({'before' if period == 'Past' else 'from'} 2018)")
        ax.axhline(0, color=U.AXIS, lw=1)
        ax.axvline(0, color=U.MUTED, lw=1)
        ax.set_title(title, fontsize=10, color=U.INK, loc="left")
        ax.set_xlabel("Trading days from day 0", fontsize=9, color=U.INK_2)
    axes[0].set_ylabel("Cumulative EM minus World (log %)", fontsize=9, color=U.INK_2)
    fig.suptitle("EM relative to World around named crises, average by group", fontsize=11,
                 color=U.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    U.fig_legend(fig, axes[0], ncol=2)
    U.save_fig(fig, name)


def run(m, daily):
    coefs, pvals = regression_table(m)
    U.save_table(coefs, "step3_crisis_regressions")
    U.save_table(pvals, "step3_crisis_regressions_pvalues")
    paths, tab, group_paths, group_tab = event_study(daily)
    U.save_table(paths, "step3_event_paths_daily")
    U.save_table(tab, "step3_event_study")
    U.save_table(group_tab, "step3_event_study_by_group")
    plot_events(group_paths, "step3_event_study")
    return {"coefs": coefs, "pvals": pvals, "events": tab, "event_groups": group_tab}
