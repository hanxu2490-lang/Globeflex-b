"""Question 2 report outputs (Step 3 results gathered for the write-up).

Table 1: equation (2) in every version (EM, EM ex-IT, EM local currency; with / without X;
         Current from 2018 and from 2022), coefficients with Newey-West t-stats.
Table 2: event study, one row per crisis, with the currency and tech parts of EM minus World.
Figure:  average cumulative EM minus World from day -5 to +20, past vs current,
         financial crises and war / trade crises in separate panels.
Drivers: average monthly values in each crisis cell (dollar, VIX, currency part, tech part).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data_prep
from . import utils as U
from .step3_crisis import crisis_regression

ROWS = [("Fin", "γ1 Fin"), ("Geo", "γ2 Geo"), ("Fin_x_Current", "γ3 Fin × Current"),
        ("Geo_x_Current", "γ4 Geo × Current"), ("Post", "θ Post"), ("dVIX", "φ ΔVIX"),
        ("dDXY", "φ ΔDXY"), ("const", "α")]


def table1(m):
    versions = [
        ("A", "EM", "EM", False, C.CURRENT_START), ("A", "EM", "EM", True, C.CURRENT_START),
        ("A", "EM ex-IT", "EM_EXIT", False, C.CURRENT_START), ("A", "EM ex-IT", "EM_EXIT", True, C.CURRENT_START),
        ("A", "EM local ccy", "EM_LC", False, C.CURRENT_START), ("A", "EM local ccy", "EM_LC", True, C.CURRENT_START),
        ("B", "EM", "EM", False, C.CURRENT_START_ROBUST), ("B", "EM", "EM", True, C.CURRENT_START_ROBUST),
        ("B", "EM ex-IT", "EM_EXIT", False, C.CURRENT_START_ROBUST),
        ("B", "EM ex-IT", "EM_EXIT", True, C.CURRENT_START_ROBUST),
    ]
    num, txt = {}, {}
    for k, (panel, lab, dep, ctrl, cur) in enumerate(versions, start=1):
        res, _ = crisis_regression(m, dep, ctrl, cur)
        col = f"({k}) {lab}{' + X' if ctrl else ''}"
        n, t = {"panel": panel, "current from": cur[:4]}, {"panel": panel}
        for key, rl in ROWS:
            if key in res.params:
                b, tv, p = res.params[key], res.tvalues[key], res.pvalues[key]
                n[rl], n[f"t {rl}"] = b, tv
                t[rl] = f"{b:.2f}{U.stars(p)}<br><span class='t'>({tv:.2f})</span>"
            else:
                t[rl] = ""
        for lab2, names in [("γ1+γ3 current Fin", ["Fin", "Fin_x_Current"]),
                            ("γ2+γ4 current Geo", ["Geo", "Geo_x_Current"])]:
            b, tv, p = U.lincom(res, names)
            n[lab2], n[f"t {lab2}"] = b, tv
            t[lab2] = f"{b:.2f}{U.stars(p)}<br><span class='t'>({tv:.2f})</span>"
        n["R2"], n["months"] = res.rsquared, int(res.nobs)
        t["R²"], t["Months"] = f"{res.rsquared:.3f}", f"{int(res.nobs)}"
        num[col], txt[col] = n, t
    return pd.DataFrame(num), pd.DataFrame(txt)


def daily_components(raw, m):
    """Daily EM minus World, its currency part and EM ex-IT minus World (log %)."""
    d = data_prep.build_daily(raw, cols=("EM", "WORLD", "EM_LC", "EM_IT"))
    w = m["W_IT"].shift(1).reindex(d.index.to_period("M")).values     # weight at previous month end
    R_em, R_it = np.exp(d["EM"] / 100) - 1, np.exp(d["EM_IT"] / 100) - 1
    exit_ = 100 * np.log(1 + (R_em - w * R_it) / (1 - w))
    return pd.DataFrame({"rel": d["EM"] - d["WORLD"], "ccy": d["EM"] - d["EM_LC"],
                         "rel_exit": exit_ - d["WORLD"]}, index=d.index)


def table2(comp, pre=C.EVENT_PRE, post=C.EVENT_POST):
    rows, paths = [], {}
    for name, date, kind, period in C.CRISES:
        i0 = comp.index.searchsorted(pd.Timestamp(date))
        win = comp.iloc[i0 - pre:i0 + post + 1]
        cum = win.cumsum()
        paths[name] = pd.Series(cum["rel"].values, index=range(-pre, post + 1))
        rows.append({"crisis": name, "type": kind, "period": period, "day 0": comp.index[i0].date(),
                     "CAR[-5,0]": cum["rel"].iloc[pre], "CAR[-5,+5]": cum["rel"].iloc[pre + 5],
                     "CAR[-5,+20]": cum["rel"].iloc[-1],
                     "currency part": cum["ccy"].iloc[-1],
                     "local-ccy CAR[-5,+20]": cum["rel"].iloc[-1] - cum["ccy"].iloc[-1],
                     "EM ex-IT CAR[-5,+20]": cum["rel_exit"].iloc[-1],
                     "tech part": cum["rel"].iloc[-1] - cum["rel_exit"].iloc[-1]})
    tab = pd.DataFrame(rows)
    order = {"Past Fin": 0, "Current Fin": 1, "Past Geo": 2, "Current Geo": 3, "Current Trade": 4}
    tab["_o"] = (tab["period"] + " " + tab["type"]).map(order)
    tab = tab.sort_values(["_o", "day 0"]).drop(columns="_o").reset_index(drop=True)
    num_cols = [c for c in tab.columns if "CAR" in c or "part" in c]
    grp = tab.assign(group=tab["period"] + " " + tab["type"]).groupby("group", sort=False)
    avg = grp[num_cols].mean()
    avg.insert(0, "events", grp.size())
    # war and trade together (= Geo in the regression)
    geo_cur = tab[(tab["period"] == "Current") & tab["type"].isin(["Geo", "Trade"])]
    avg.loc["Current Geo + Trade"] = [len(geo_cur), *geo_cur[num_cols].mean().values]
    return tab, avg, pd.DataFrame(paths)


def drivers(m):
    """Average monthly values per crisis cell (cells can overlap: a month may be Fin and Geo)."""
    cur = m["Current"] == 1
    cells = {"Non-crisis": m["Crisis"] == 0,
             "Fin, past": (m["Fin"] == 1) & ~cur, "Fin, current": (m["Fin"] == 1) & cur,
             "Geo, past": (m["Geo"] == 1) & ~cur, "Geo, current": (m["Geo"] == 1) & cur}
    v = pd.DataFrame({"EM − World": m["EM"] - m["WORLD"], "currency part (USD − local)": m["EM"] - m["EM_LC"],
                      "tech part (EM − EM ex-IT)": m["EM"] - m["EM_EXIT"],
                      "ΔDXY (%)": m["dDXY"], "ΔVIX (points)": m["dVIX"], "IT weight (%)": 100 * m["W_IT_LAG"]})
    out = pd.DataFrame({k: v[s].mean() for k, s in cells.items()}).T
    out.insert(0, "months", [int(s.sum()) for s in cells.values()])
    return out


def plot(tab, paths, name="q2_event_paths"):
    key = tab.set_index("crisis")
    panels = [("Financial crises", {"Past": ["Fin"], "Current": ["Fin"]}),
              ("War and trade crises", {"Past": ["Geo"], "Current": ["Geo", "Trade"]})]
    fig, axes = U.new_fig(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (title, sel) in zip(axes, panels):
        counts = []
        for k, period in enumerate(["Past", "Current"]):
            names = key[(key["period"] == period) & key["type"].isin(sel[period])].index
            counts.append(len(names))
            for nm in names:
                ax.plot(paths.index, paths[nm], color=U.SERIES[k], lw=0.8, alpha=0.25)
            ax.plot(paths.index, paths[names].mean(axis=1), color=U.SERIES[k], lw=2.2,
                    label=f"{period} ({'before' if period == 'Past' else 'from'} 2018), average")
        ax.axhline(0, color=U.AXIS, lw=1)
        ax.axvline(0, color=U.MUTED, lw=1)
        ax.set_title(f"{title}: {counts[0]} past, {counts[1]} current", fontsize=10, color=U.INK, loc="left")
        ax.set_xlabel("Trading days from day 0", fontsize=9, color=U.INK_2)
        ax.set_xticks([-5, 0, 5, 10, 15, 20])
    axes[0].set_ylabel("Cumulative EM minus World (log %)", fontsize=9, color=U.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    U.fig_legend(fig, axes[0], ncol=2)
    U.save_fig(fig, name)


def run(raw, m):
    t1_num, t1_txt = table1(m)
    U.save_table(t1_num, "q2_table1_crisis_regressions")
    U.save_table(t1_txt, "q2_table1_formatted")
    comp = daily_components(raw, m)
    t2, t2_avg, paths = table2(comp)
    U.save_table(t2, "q2_table2_event_study", index=False)
    U.save_table(t2_avg, "q2_table2_event_study_by_group")
    drv = drivers(m)
    U.save_table(drv, "q2_drivers_by_crisis_cell")
    plot(t2, paths)
    return {"t1_num": t1_num, "t1_txt": t1_txt, "t2": t2, "t2_avg": t2_avg, "drivers": drv}
