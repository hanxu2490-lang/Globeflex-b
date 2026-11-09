"""Step 3 - Crisis behaviour today versus the past (Question 2).

  r_EM,t - r_DM,t = alpha + g1 Fin_t + g2 Geo_t + g3 (Fin_t x Current_t) + g4 (Geo_t x Current_t)
                    + theta Post_t + phi' X_t + e_t                                          (2)

Equation (2) is estimated with and without the Post_t term.

Fin_t = 1 when MSCI World is more than 10% below its peak; Geo_t = 1 when GPR or TPU is in its
top decile; Current_t = 1 from 2018. X_t = changes in the VIX and the dollar index (reported with
and without). Repeated for EM ex-IT. Daily cumulative EM-minus-DM returns from 5 days before to
20 days after each named crisis, compared with non-crisis placebo dates (Module E placebo test).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import utils as U
from .data_prep import LABELS

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


# ------------------------------------------------------------------ crisis vs non-crisis placebo (Module E)
def placebo_test(daily, m, crises=C.CRISES, pre=C.EVENT_PRE, post=C.EVENT_POST, B=C.PLACEBO_DRAWS,
                 seed=C.PLACEBO_SEED):
    """Mean CAR[-5,+20] of the named crises against placebo sets drawn from non-crisis dates.

    Control dates: from 2001, in months with no Fin, GPR or TPU flag (Crisis_t = 0), and outside
    +/-(pre + post) trading days of every named event. Each placebo set draws, without replacement,
    as many Past and as many Current dates as there are events. Two-sided empirical p-value centred
    on the placebo mean. Descriptive benchmark, not a causal test.
    """
    dates = daily.index
    arr = (daily["EM"] - daily["WORLD"]).to_numpy()

    def car(i):                                   # sum of EM - World from day -pre through day +post
        return float(arr[i - pre:i + post + 1].sum())

    ev = pd.DataFrame(crises, columns=["event", "event_date", "type", "period"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev["status"] = np.where(ev["event"].isin(C.NON_CORE), "Robustness", "Core")
    idx = np.array([dates.searchsorted(d) for d in ev["event_date"]])
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

    samples = [("All 15", ev), ("Core 12", ev[ev["status"] == "Core"]),
               ("Excluding Crimea (14)", ev[~ev["event"].str.contains("Crimea")]),
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


def run(m, daily):
    coefs, pvals = regression_table(m, posts=(True, False))
    U.save_table(coefs, "step3_crisis_regressions")
    U.save_table(pvals, "step3_crisis_regressions_pvalues")
    paths, tab, group_paths, group_tab = event_study(daily)
    U.save_table(paths, "step3_event_paths_daily")
    U.save_table(tab, "step3_event_study")
    U.save_table(group_tab, "step3_event_study_by_group")
    plot_events(group_paths, "step3_event_study")
    placebo, ev, controls, dists = placebo_test(daily, m)
    U.save_table(placebo, "step3_placebo_summary")
    U.save_table(controls.set_index("date"), "step3_placebo_control_pool")
    U.save_table(pd.DataFrame({k: dists[k] for k in ["All 15", "Core 12", "Excluding Crimea (14)"]},
                              index=pd.RangeIndex(1, C.PLACEBO_DRAWS + 1, name="replicate")), "step3_placebo_draws")
    plot_placebo(ev, dists, "step3_placebo")
    return {"coefs": coefs, "pvals": pvals, "events": tab, "event_groups": group_tab, "placebo": placebo,
            "placebo_controls": controls["period"].value_counts()}
