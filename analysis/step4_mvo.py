"""Step 4 - Mean-variance optimizer (Questions 1 and 2).

Following the professor's suggestion (Markowitz mean-variance):
  1. Efficient frontier of conventional assets: S&P 500 (total return) + Bloomberg US Aggregate bonds.
  2. Add EM, then EM ex-IT, and let the optimizer decide how much to hold, in big windows (5 and 10 years).
  3. Force the optimizer to own 10% of the EM asset (with and without tech).
  4. Load the individual EM country indices and let the optimizer build the EM part itself
     (free, and with the countries together held at 10%).
All portfolios are long-only and fully invested. Expected returns and covariances are the sample
values of monthly simple returns within each window, annualized; the risk-free rate is the
window's average 3-month T-bill. The analysis is in-sample: it shows what the optimizer would have
chosen with hindsight in each window, not an investable strategy.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from . import config as C
from . import utils as U

BASE = ["S&P 500", "US Agg"]
COLORS = {"S&P 500 + US Agg": U.INK_2, "+ EM": U.SERIES[0], "+ EM ex-IT": U.SERIES[1], "+ EM countries": U.SERIES[2]}


# ------------------------------------------------------------------ data
def _read(fname, sheet):
    df = pd.read_excel(C.DATA_DIR / fname, sheet_name=sheet, header=None).iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().drop_duplicates("date").set_index("date")["value"].sort_index()


def monthly_returns(raw, m):
    """Monthly simple returns (decimal) of all assets, aligned on calendar months."""
    end = raw["END_DATE"]
    me = lambda s: s.loc[:end].resample("ME").last().to_period("M")
    lv = pd.DataFrame({"S&P 500": me(_read("SP500 and US Agg.xlsx", "S&P500 daily")),
                       "US Agg": me(_read("SP500 and US Agg.xlsx", "LBUSTRUU Index daily")),
                       **{c: me(_read("Country_returns_daily.xlsx", c)) for c in C.EM_COUNTRIES}})
    R = lv.pct_change()
    R["EM"] = np.exp(m["EM"] / 100) - 1
    R["EM ex-IT"] = np.exp(m["EM_EXIT"] / 100) - 1
    R = R.loc[C.SAMPLE_START:C.SAMPLE_END]
    return R, m["RF"].reindex(R.index)


# ------------------------------------------------------------------ optimizer
def _constraints(n, group, fixed):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if fixed is not None:
        cons.append({"type": "eq", "fun": lambda w, g=group: w[g].sum() - fixed})
    return cons


def _solve(S, cons, n, x0=None):
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def frontier(mu, S, rf, group=(), fixed=None, points=40):
    """Long-only efficient frontier and tangency (max-Sharpe) portfolio. group = indices of EM assets."""
    n, group = len(mu), list(group)
    cons = _constraints(n, group, fixed)
    w_min = _solve(S, cons, n)
    # highest achievable return under the same constraints
    A_eq, b_eq = [np.ones(n)], [1.0]
    if fixed is not None:
        a = np.zeros(n); a[group] = 1; A_eq.append(a); b_eq.append(fixed)
    w_max = linprog(-mu, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, 1)] * n).x
    targets = np.linspace(w_min @ mu, w_max @ mu, points)
    W = [w_min]
    for t in targets[1:]:
        W.append(_solve(S, cons + [{"type": "eq", "fun": lambda w, t=t: w @ mu - t}], n, W[-1]))
    W = np.array(W)
    vol, ret = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W)), W @ mu
    sharpe = (ret - rf) / vol
    # refine the tangency portfolio from the best frontier point
    best = W[np.argmax(sharpe)]
    res = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ S @ w), best, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    w_tan = res.x if (-res.fun) >= sharpe.max() else best
    return {"vol": vol, "ret": ret, "w_min": w_min, "w_tan": w_tan}


def best_at_risk(mu, S, sigma, group=(), fixed=None, x0=None):
    """Highest-return long-only portfolio with volatility no higher than sigma."""
    n, group = len(mu), list(group)
    cons = _constraints(n, group, fixed) + [{"type": "ineq", "fun": lambda w: sigma ** 2 - w @ S @ w}]
    starts = [np.full(n, 1 / n) if x0 is None else x0] + [np.eye(n)[i] for i in range(n)]
    best = None
    for w0 in starts:                         # several starts: the risk constraint makes SLSQP start-sensitive
        r = minimize(lambda w: -(w @ mu), w0, method="SLSQP", bounds=[(0, 1)] * n, constraints=cons,
                     options={"ftol": 1e-12, "maxiter": 500})
        ok = r.success and r.x @ S @ r.x <= sigma ** 2 * (1 + 1e-6) and abs(r.x.sum() - 1) < 1e-6
        if ok and (best is None or r.x @ mu > best @ mu):
            best = r.x
    return best


def variants(window_assets):
    """The asset sets compared in each window: (name, assets, EM assets, fixed EM weight)."""
    cty = [c for c in C.EM_COUNTRIES if c in window_assets]
    return [("S&P 500 + US Agg", BASE, [], None),
            ("+ EM", BASE + ["EM"], ["EM"], None),
            ("+ EM ex-IT", BASE + ["EM ex-IT"], ["EM ex-IT"], None),
            ("+ EM countries", BASE + cty, cty, None),
            ("+ EM, fixed 10%", BASE + ["EM"], ["EM"], C.MVO_FIXED_EM),
            ("+ EM ex-IT, fixed 10%", BASE + ["EM ex-IT"], ["EM ex-IT"], C.MVO_FIXED_EM),
            ("+ EM countries, together 10%", BASE + cty, cty, C.MVO_FIXED_EM)]


def analyse_window(R, rf, start, end):
    r = R.loc[start:end]
    full = [c for c in r.columns if r[c].notna().all()]          # countries need data for the whole window
    r = r[full]
    rf_ann = 12 * rf.loc[start:end].mean()
    label = f"{start[:4]}–{end[:4]}" if end[:4] != start[:4] else start
    rows, weights, curves = [], {}, {}
    base_sharpe = base_ret = None
    S0 = 12 * r[BASE].cov().values
    w6040 = np.array([0.6, 0.4])
    sigma6040 = np.sqrt(w6040 @ S0 @ w6040)                  # risk of a 60/40 S&P 500 / US Agg portfolio
    for name, assets, em, fixed in variants(full):
        mu, S = 12 * r[assets].mean().values, 12 * r[assets].cov().values
        f = frontier(mu, S, rf_ann, [assets.index(a) for a in em], fixed)
        w = f["w_tan"]
        ret, vol = w @ mu, np.sqrt(w @ S @ w)
        sharpe = (ret - rf_ann) / vol
        base_sharpe = sharpe if base_sharpe is None else base_sharpe
        g = [assets.index(a) for a in em]
        w6 = best_at_risk(mu, S, sigma6040, g, fixed)
        ret6 = w6 @ mu
        base_ret = ret6 if base_ret is None else base_ret
        rows.append({"window": label, "months": len(r), "portfolio": name,
                     "max Sharpe": sharpe, "change vs S&P+Agg": sharpe - base_sharpe,
                     "EM weight in max-Sharpe %": 100 * sum(w[assets.index(a)] for a in em),
                     "return %": 100 * ret, "volatility %": 100 * vol,
                     "min-variance volatility %": 100 * np.sqrt(f["w_min"] @ S @ f["w_min"]),
                     "60/40 volatility %": 100 * sigma6040,
                     "best return at 60/40 risk %": 100 * ret6, "return gain at 60/40 risk (pp)": 100 * (ret6 - base_ret),
                     "EM weight at 60/40 risk %": 100 * w6[g].sum() if g else 0.0,
                     "countries available": ", ".join(em) if name.startswith("+ EM countries") else ""})
        weights[(name, "max Sharpe")] = pd.Series(100 * w, index=assets)
        weights[(name, "at 60/40 risk")] = pd.Series(100 * w6, index=assets)
        curves[name] = (100 * f["vol"], 100 * f["ret"], 100 * vol, 100 * ret)
    wt = pd.DataFrame(weights).T
    wt.index.names = ["portfolio", "point on frontier"]
    wt.insert(0, "window", label)
    return pd.DataFrame(rows), wt, curves


# ------------------------------------------------------------------ figures
def plot_frontiers(all_curves, name="step4_mvo_frontiers"):
    fig, axes = U.new_fig(2, 3, figsize=(13, 7.4))
    axes = axes.ravel()
    for ax, (label, curves) in zip(axes, all_curves.items()):
        for v in ["S&P 500 + US Agg", "+ EM", "+ EM ex-IT", "+ EM countries"]:
            x, y, tx, ty = curves[v]
            ax.plot(x, y, color=COLORS[v], lw=2 if v != "S&P 500 + US Agg" else 2.4, label=v,
                    ls="-" if v != "S&P 500 + US Agg" else "--")
            ax.plot(tx, ty, "o", color=COLORS[v], ms=6, mec=U.SURFACE, mew=1.5)
        ax.set_title(label, fontsize=10, color=U.INK, loc="left")
        ax.set_xlabel("Volatility (% a year)", fontsize=9, color=U.INK_2)
        ax.set_ylabel("Return (% a year)", fontsize=9, color=U.INK_2)
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.55, "Long-only efficient frontiers.\nDots = max-Sharpe portfolio.\n"
                  "Dashed = S&P 500 + US Agg only.\nThe optimizer chooses how much EM\nto hold (0–100%).",
                  fontsize=10, color=U.INK_2, va="center", transform=axes[-1].transAxes)
    fig.suptitle("Efficient frontiers by 5-year window: does adding EM move the frontier?", fontsize=12,
                 color=U.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    U.fig_legend(fig, axes[0], ncol=4)
    U.save_fig(fig, name)


def plot_gains(table, name="step4_mvo_gains"):
    fig, axes = U.new_fig(2, 2, figsize=(12, 7.2), sharex=True)
    windows = list(dict.fromkeys(table["window"]))
    x = np.arange(len(windows))
    cols = [("Optimizer free to choose EM (0–100%)", ["+ EM", "+ EM ex-IT", "+ EM countries"]),
            ("Forced to hold 10% EM", ["+ EM, fixed 10%", "+ EM ex-IT, fixed 10%", "+ EM countries, together 10%"])]
    rows = [("change vs S&P+Agg", "Change in max Sharpe ratio"),
            ("return gain at 60/40 risk (pp)", "Change in return at 60/40 risk (pp a year)")]
    labels = ["EM", "EM ex-IT", "EM countries (optimizer picks)"]
    for i, (metric, ylab) in enumerate(rows):
        for j, (title, names) in enumerate(cols):
            ax = axes[i, j]
            for k, (v, lab) in enumerate(zip(names, labels)):
                vals = [table[(table["window"] == w) & (table["portfolio"] == v)][metric].iloc[0] for w in windows]
                ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=U.SERIES[k], label=lab)
            ax.axhline(0, color=U.AXIS, lw=1)
            ax.set_xticks(x, windows, fontsize=9)
            if i == 0:
                ax.set_title(title, fontsize=10, color=U.INK, loc="left")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9, color=U.INK_2)
    for i in range(2):                         # same scale within a row, so free and forced compare directly
        lo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0]); hi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for j in range(2):
            axes[i, j].set_ylim(lo, hi)
    fig.suptitle("What EM adds to an S&P 500 + US Agg portfolio, by 5-year window", fontsize=12, color=U.INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    U.fig_legend(fig, axes[0, 0], ncol=3)
    U.save_fig(fig, name)


# ------------------------------------------------------------------ run
def run(raw, m):
    R, rf = monthly_returns(raw, m)
    tabs, wts, curves = [], [], {}
    for start, end in C.MVO_WINDOWS:
        t, w, c = analyse_window(R, rf, start, end)
        tabs.append(t); wts.append(w); curves[t["window"].iloc[0]] = c
    table, weights = pd.concat(tabs, ignore_index=True), pd.concat(wts)
    tabs10 = [analyse_window(R, rf, s, e)[0] for s, e in C.MVO_WINDOWS_10Y]
    table10 = pd.concat(tabs10, ignore_index=True)
    corr = R.loc["2021-01":, ["S&P 500", "US Agg", "EM", "EM ex-IT"] + C.EM_COUNTRIES].corr()

    U.save_table(table, "step4_mvo_5y", index=False)
    U.save_table(table10, "step4_mvo_10y", index=False)
    U.save_table(weights.round(2), "step4_mvo_weights_5y")
    U.save_table(corr.round(2), "step4_mvo_correlations_2021_2026")
    plot_frontiers(curves)
    plot_gains(table)
    return {"table": table, "table10": table10, "weights": weights}
