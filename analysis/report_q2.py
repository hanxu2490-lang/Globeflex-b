# -*- coding: utf-8 -*-
"""Question 2 outputs prepared for the report page (make_report.py builds the page itself).

Table 1: equation (2) in every version (EM, EM ex-IT, EM local currency; with / without X;
         Current from 2018 and from 2022), coefficients with Newey-West t-stats.
Figure:  average cumulative EM minus World from day -5 to +20, past vs current,
         financial crises and war / trade crises in separate panels (event windows from Step 3).
"""
import pandas as pd

from . import config as C
from . import utils as U
from .q2_step3_crisis import crisis_regression

ROWS = [("Fin", "γ1 Fin"), ("Geo", "γ2 Geo"), ("Fin_x_Current", "γ3 Fin × Current"),
        ("Geo_x_Current", "γ4 Geo × Current"), ("Post", "θ Post"), ("dVIX", "φ ΔVIX"),
        ("dDXY", "φ ΔDXY"), ("const", "α")]


def table1(m):
    S18, S22 = C.CURRENT_START, C.CURRENT_START_ROBUST
    # (panel, label, dependent, controls X, current from, Post in the equation)
    versions = [
        ("A", "EM", "EM", False, S18, True), ("A", "EM", "EM", True, S18, True),
        ("A", "EM ex-IT", "EM_EXIT", False, S18, True), ("A", "EM ex-IT", "EM_EXIT", True, S18, True),
        ("B", "EM", "EM", False, S18, False), ("B", "EM", "EM", True, S18, False),
        ("B", "EM ex-IT", "EM_EXIT", False, S18, False), ("B", "EM ex-IT", "EM_EXIT", True, S18, False),
        ("C", "EM local ccy", "EM_LC", False, S18, True), ("C", "EM local ccy", "EM_LC", True, S18, True),
        ("C", "EM, from 2022", "EM", False, S22, True), ("C", "EM, from 2022", "EM", True, S22, True),
        ("C", "EM ex-IT, from 2022", "EM_EXIT", False, S22, True), ("C", "EM ex-IT, from 2022", "EM_EXIT", True, S22, True),
    ]
    num, txt = {}, {}
    for k, (panel, lab, dep, ctrl, cur, post) in enumerate(versions, start=1):
        res, _ = crisis_regression(m, dep, ctrl, cur, post=post)
        col = f"({k}) {lab}{' + X' if ctrl else ''}"
        n, t = {"panel": panel, "current from": cur[:4], "Post": "yes" if post else "no"}, {"panel": panel}
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


def plot(tab, paths, name="q2_event_paths"):
    key = tab
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


def run(m, r3):
    t1_num, t1_txt = table1(m)
    U.save_table(t1_num, "q2_table1_crisis_regressions")
    U.save_table(t1_txt, "q2_table1_formatted")
    plot(r3["events"], r3["paths"])
    return {"t1_num": t1_num, "t1_txt": t1_txt}
