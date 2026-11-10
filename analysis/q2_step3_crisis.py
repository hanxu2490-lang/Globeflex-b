"""Step 3 - Crisis behaviour today versus the past (Question 2).

  r_EM,t - r_DM,t = alpha + g1 Fin_t + g2 Geo_t + g3 (Fin_t x Current_t) + g4 (Geo_t x Current_t)
                    + theta Post_t + phi' X_t + e_t                                          (2)

Equation (2) is estimated with and without the Post_t term.

Fin_t = 1 when MSCI World is more than 10% below its peak; Geo_t = 1 when GPR or TPU is in its
top decile; Current_t = 1 from 2018. X_t = changes in the VIX and the dollar index (reported with
and without). Repeated for EM ex-IT.

Event study (Module E): each named crisis is checked against the crisis rule in its event month
(crisis selection); daily EM, World and EM-minus-World log returns from 5 days before to 20 days after
each crisis (event windows); the crises' mean cumulative EM minus World is compared with non-crisis
placebo dates (placebo test).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import utils as U
from .q2_data_prep import LABELS

COEF_NAMES = {"Fin": "g1 Fin", "Geo": "g2 Geo", "Fin_x_Current": "g3 Fin x Current",
              "Geo_x_Current": "g4 Geo x Current", "Post": "theta Post", "dVIX": "phi dVIX",
              "dDXY": "phi dDXY", "const": "alpha"}


def crisis_regression(m, dep, controls, current_start=C.CURRENT_START, dm="WORLD", post=True):
    cur = (m.index >= pd.Period(current_start, "M")).astype(int)
    X = pd.DataFrame({"Fin": m["Fin"], "Geo": m["Geo"], "Fin_x_Current": m["Fin"] * cur,
                      "Geo_x_Current": m["Geo"] * cur}, index=m.index)
    if post:
        X["Post"] = m["Post"]
    if controls:
        X[["dVIX", "dDXY"]] = m[["dVIX", "dDXY"]]
    return U.ols_nw(m[dep] - m[dm], X), cur


def crisis_column(m, dep, controls, current_start=C.CURRENT_START, post=True):
    """One results column: coefficients (t-stats), current-crisis totals and month counts."""
    res, cur = crisis_regression(m, dep, controls, current_start, post=post)
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


def regression_table(m, deps=("EM", "EM_EXIT"), current_start=C.CURRENT_START, posts=(True,)):
    """Columns are named "<index> - World, no X" (with Post) and "..., no X, no Post" (without Post)."""
    cols, pvals = {}, {}
    for post in posts:
        for dep in deps:
            for controls in (False, True):
                name = f"{LABELS[dep]} - World, {'with' if controls else 'no'} X{'' if post else ', no Post'}"
                cols[name], res = crisis_column(m, dep, controls, current_start, post=post)
                pvals[name] = res.pvalues.rename(COEF_NAMES)
    return pd.DataFrame(cols), pd.DataFrame(pvals)


# ------------------------------------------------------------------ crisis selection (Module E)
SELECTION_STATUS = {"Met": "Core event (rule met)", "Met only after event": "Retained - shock-triggered",
                    "Not met": "Additional case - rule not met"}


def crisis_selection(raw, m, crises=C.CRISES, post=C.EVENT_POST):
    """Check each named crisis against the crisis rule in its event month (Module E crisis selection).

    Fin:   MSCI World month-end drawdown from its running peak in the event month at or below -10% -> Met;
           otherwise, the daily drawdown reaches -10% between Day 0 and Day +post -> Met only after event.
    Geo:   GPR in the event month in its top decile over the sample -> Met.
    Trade: TPU in the event month in its top decile over the sample -> Met.
    Status: Met = core event; Met only after event = retained, shock-triggered; Not met = additional case.
    """
    world = pd.concat({"EM": raw["EM"], "WORLD": raw["WORLD"]}, axis=1, join="inner")["WORLD"]
    dd_daily = world / world.cummax() - 1            # daily drawdown on the event-study calendar
    thr = {"Fin": C.FIN_DRAWDOWN, "Geo": m["GPR"].quantile(C.GEO_QUANTILE), "Trade": m["TPU"].quantile(C.GEO_QUANTILE)}
    test = {"Fin": "World month-end drawdown", "Geo": "GPR", "Trade": "TPU"}
    rows = []
    for name, date, kind, period in crises:
        d = pd.Timestamp(date)
        i0 = world.index.searchsorted(d)
        if kind == "Fin":
            val = m.loc[d.to_period("M"), "WORLD_DD"]
            res = ("Met" if val <= thr[kind] else
                   "Met only after event" if (dd_daily.iloc[i0:i0 + post + 1] <= thr[kind]).any() else "Not met")
        else:
            val = m.loc[d.to_period("M"), "GPR" if kind == "Geo" else "TPU"]
            res = "Met" if val >= thr[kind] else "Not met"
        rows.append({"event": name, "event date": d.date(), "type": kind, "period": period, "day 0": world.index[i0].date(),
                     "test": test[kind], "test value (event month)": val, "threshold": thr[kind],
                     "daily drawdown at day 0": dd_daily.iloc[i0], "result": res, "status": SELECTION_STATUS[res]})
    return pd.DataFrame(rows).set_index("event")


# ------------------------------------------------------------------ daily event windows (Module E)
def event_windows(daily, sel, pre=C.EVENT_PRE, post=C.EVENT_POST):
    """Day -pre to +post around each crisis: daily and cumulative EM, World and EM - World log returns (%)."""
    out = []
    for name, r in sel.iterrows():
        i0 = daily.index.searchsorted(pd.Timestamp(r["event date"]))     # first trading day on/after the date
        assert i0 - pre >= 0 and i0 + post < len(daily), f"window outside the data: {name}"
        w = daily.iloc[i0 - pre:i0 + post + 1]
        out.append(pd.DataFrame({
            "event": name, "event date": r["event date"], "day 0": daily.index[i0].date(), "type": r["type"],
            "period": r["period"], "status": r["status"], "relative day": range(-pre, post + 1), "trading date": w.index.date,
            "EM daily": w["EM"].values, "World daily": w["WORLD"].values, "EM - World daily": (w["EM"] - w["WORLD"]).values,
            "EM cumulative": w["EM"].cumsum().values, "World cumulative": w["WORLD"].cumsum().values,
            "EM - World cumulative": (w["EM"] - w["WORLD"]).cumsum().values}))
    return pd.concat(out, ignore_index=True)


def event_study(windows, sel, pre=C.EVENT_PRE, post=C.EVENT_POST):
    """Per-crisis summary, cumulative EM - World paths and group averages (Past / Current x type)."""
    paths = windows.pivot(index="relative day", columns="event", values="EM - World cumulative")[list(sel.index)]
    last = windows[windows["relative day"] == post].set_index("event")
    first = windows[windows["relative day"] == -pre].set_index("event")
    tab = sel[["type", "period", "status", "day 0"]].copy()
    tab["window start"], tab["window end"] = first["trading date"], last["trading date"]
    tab[f"EM cumulative [-{pre},+{post}]"] = last["EM cumulative"]
    tab[f"World cumulative [-{pre},+{post}]"] = last["World cumulative"]
    horizons = [-1, 0, 5, 10, 20]
    tab = tab.join(paths.loc[horizons].T.rename(columns=lambda h: f"CAR[-{pre},{h:+d}]"))
    tab["observations"] = windows.groupby("event").size()
    groups = tab["period"] + " " + tab["type"]
    group_paths = paths.T.groupby(groups).mean().T
    car_cols = [c for c in tab.columns if c.startswith(("CAR", "EM cum", "World cum"))]
    group_tab = tab.groupby(groups)[car_cols].mean()
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


# ------------------------------------------------------------------ crisis vs non-crisis placebo (Module E)
def placebo_test(daily, m, sel, pre=C.EVENT_PRE, post=C.EVENT_POST, B=C.PLACEBO_DRAWS, seed=C.PLACEBO_SEED):
    """Mean CAR[-5,+20] of the named crises against placebo sets drawn from non-crisis dates.

    Control dates: from 2001, in months with no Fin, GPR or TPU flag (Crisis_t = 0), and outside
    +/-(pre + post) trading days of every named event. Each placebo set draws, without replacement,
    as many Past and as many Current dates as there are events. Two-sided empirical p-value centred
    on the placebo mean. Samples: all 15, the core events (rule met), without Crimea, past, current.
    Descriptive benchmark, not a causal test.
    """
    dates = daily.index
    arr = (daily["EM"] - daily["WORLD"]).to_numpy()

    def car(i):                                   # sum of EM - World from day -pre through day +post
        return float(arr[i - pre:i + post + 1].sum())

    ev = sel.reset_index()[["event", "event date", "type", "period", "status"]]
    idx = np.array([dates.searchsorted(pd.Timestamp(d)) for d in ev["event date"]])
    assert all(i >= pre and i + post < len(dates) for i in idx)
    ev["day0"] = dates[idx]
    ev["car_log_pp"] = [car(i) for i in idx]

    flagged = set(m.index[m["Crisis"] == 1].strftime("%Y-%m"))
    exclude = np.zeros(len(dates), bool)
    for i in idx:
        exclude[max(0, i - (pre + post)):min(len(dates), i + (pre + post) + 1)] = True
    current_year = pd.Period(C.CURRENT_START, "M").year
    cand = [(i, "Past" if dates[i].year < current_year else "Current") for i in range(pre, len(dates) - post)
            if dates[i].year >= 2001 and not exclude[i] and dates[i].strftime("%Y-%m") not in flagged]
    controls = pd.DataFrame(cand, columns=["i", "period"])
    controls["date"] = dates[controls["i"]]
    controls["car_log_pp"] = [car(i) for i in controls["i"]]

    rng = np.random.default_rng(seed)
    pool = {p: controls.loc[controls["period"] == p, "car_log_pp"].to_numpy() for p in ["Past", "Current"]}
    assert all(len(v) > 0 for v in pool.values())

    def test(df):
        n = df["period"].value_counts().to_dict()
        obs, draws = df["car_log_pp"].mean(), np.zeros(B)
        for p, k in n.items():
            draws += np.array([rng.choice(pool[p], size=k, replace=False).sum() for _ in range(B)])
        draws /= len(df)
        mu = draws.mean()
        pval = (1 + np.sum(abs(draws - mu) >= abs(obs - mu))) / (B + 1)
        return {"n events": len(df), "observed mean": obs, "placebo mean": mu, "difference": obs - mu,
                "p two-sided": pval, "placebo 2.5%": np.quantile(draws, .025),
                "placebo 97.5%": np.quantile(draws, .975)}, draws

    core = ev[ev["status"].str.startswith("Core")]
    no_crimea = ev[~ev["event"].str.contains("Crimea")]
    samples = [("All 15", ev), (f"Core {len(core)}", core), (f"Excluding Crimea ({len(no_crimea)})", no_crimea),
               ("Past", ev[ev["period"] == "Past"]), ("Current", ev[ev["period"] == "Current"])]
    rows, dists = {}, {}
    for name, df in samples:
        rows[name], dists[name] = test(df)
    summary = pd.DataFrame(rows).T
    summary.index.name = "sample"
    summary["n events"] = summary["n events"].astype(int)
    return summary, ev, controls, dists


def plot_placebo(ev, dists, name):
    fig, ax = U.new_fig(figsize=(8.4, 4.6))
    a = dists["All 15"]
    ax.hist(a, bins=45, color=U.SERIES[0], alpha=0.75, edgecolor="white")
    ax.axvline(ev["car_log_pp"].mean(), color=U.SERIES[1], lw=2.5, label="Observed 15-crisis mean")
    ax.axvline(a.mean(), color=U.INK, lw=1.2, ls="--", label="Non-crisis placebo mean")
    ax.set_xlabel("Mean cumulative EM - World log return (percentage points)", fontsize=9, color=U.INK_2)
    ax.set_ylabel("Placebo repetitions", fontsize=9, color=U.INK_2)
    ax.set_title("Crisis vs non-crisis placebo (matched by period)", fontsize=11, color=U.INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    U.save_fig(fig, name)


def run(raw, m, daily):
    coefs, pvals = regression_table(m, posts=(True, False))
    U.save_table(coefs, "step3_crisis_regressions")
    U.save_table(pvals, "step3_crisis_regressions_pvalues")

    sel = crisis_selection(raw, m)
    U.save_table(sel, "step3_crisis_selection")
    windows = event_windows(daily, sel)
    U.save_table(windows, "step3_event_windows", index=False)
    paths, tab, group_paths, group_tab = event_study(windows, sel)
    U.save_table(paths, "step3_event_paths_daily")
    U.save_table(tab, "step3_event_study")
    U.save_table(group_tab, "step3_event_study_by_group")
    plot_events(group_paths, "step3_event_study")

    placebo, ev, controls, dists = placebo_test(daily, m, sel)
    U.save_table(placebo, "step3_placebo_summary")
    U.save_table(controls.set_index("date"), "step3_placebo_control_pool")
    U.save_table(pd.DataFrame({k: dists[k] for k in list(dists)[:3]},
                              index=pd.RangeIndex(1, C.PLACEBO_DRAWS + 1, name="replicate")), "step3_placebo_draws")
    plot_placebo(ev, dists, "step3_placebo")
    return {"coefs": coefs, "pvals": pvals, "selection": sel, "events": tab, "paths": paths,
            "event_groups": group_tab, "placebo": placebo, "placebo_controls": controls["period"].value_counts()}
