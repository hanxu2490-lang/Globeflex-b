"""Step 4 - Portfolio evaluation (Questions 1 and 2).

A US investor holding MSCI USA adds EM, EM Small Cap or EM ex-IT, with minimum-variance weights
re-estimated monthly from the previous 60 months. Out of sample from 2006 we compare volatility,
Sharpe ratio (3-month T-bill as risk-free rate), maximum drawdown and crisis-month returns.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import utils as U
from .data_prep import LABELS

ADDED = ["EM", "EMSC", "EM_EXIT", "EM_PX"]   # EM (price) is the same-basis comparator for EM Small Cap


def simple(r_log_pct):
    return np.exp(r_log_pct / 100) - 1


def min_var_weight(R_us, R_x, long_only=C.LONG_ONLY):
    """Weight on asset x in the two-asset minimum-variance portfolio."""
    S = np.cov(np.column_stack([R_us, R_x]).T)
    w = (S[0, 0] - S[0, 1]) / (S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return float(np.clip(w, 0, 1)) if long_only else float(w)


def backtest(m, add, start=C.OOS_START, window=C.ROLL_WINDOW):
    """Monthly-rebalanced US + `add` portfolio; weights from the previous `window` months."""
    R = simple(m[["US", add]].dropna())
    months = R.index[R.index >= pd.Period(start, "M")]
    w, rp = {}, {}
    for t in months:
        hist = R.loc[:t - 1].iloc[-window:]
        if len(hist) < window:
            continue
        w[t] = min_var_weight(hist["US"].values, hist[add].values)
        rp[t] = (1 - w[t]) * R.at[t, "US"] + w[t] * R.at[t, add]
    return pd.Series(w, name=add), pd.Series(rp, name=add)


def max_drawdown(r):
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def performance(r, m, weights=None):
    """Annualized return, volatility, Sharpe, max drawdown and crisis-month returns."""
    rf = m.loc[r.index, "RF"]
    ex = r - rf
    cur = m.loc[r.index, "Current"] == 1
    fin, geo = m.loc[r.index, "Fin"] == 1, m.loc[r.index, "Geo"] == 1
    out = {
        "months": len(r),
        "ann. return %": 100 * ((1 + r).prod() ** (12 / len(r)) - 1),
        "ann. volatility %": 100 * r.std() * np.sqrt(12),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(12),
        "max drawdown %": 100 * max_drawdown(r),
        "avg return in crisis months %": 100 * r[fin | geo].mean(),
        "Fin months, past %": 100 * r[fin & ~cur].mean(),
        "Fin months, current %": 100 * r[fin & cur].mean(),
        "Geo months, past %": 100 * r[geo & ~cur].mean(),
        "Geo months, current %": 100 * r[geo & cur].mean(),
    }
    if weights is not None:
        out["avg EM weight %"] = 100 * weights.mean()
        out["last EM weight %"] = 100 * weights.iloc[-1]
    return out


def evaluate(m, added=ADDED, start=C.OOS_START):
    weights, rets = {}, {}
    for a in added:
        weights[a], rets[a] = backtest(m, a, start)
    idx = rets[added[0]].index
    for a in added:
        assert rets[a].index.equals(idx), "portfolios must share the same out-of-sample months"
    us_only = simple(m.loc[idx, "US"])
    rows = {"USA only": performance(us_only, m)}
    for a in added:
        rows[f"USA + {LABELS[a]}"] = performance(rets[a], m, weights[a])
    perf = pd.DataFrame(rows).T
    return perf, pd.DataFrame(weights), pd.DataFrame(rets).assign(US_only=us_only)


def plot_portfolios(rets, weights, name):
    fig, axes = U.new_fig(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    x = rets.index.to_timestamp()
    series = [("US_only", "USA only"), ("EM", "USA + EM"), ("EMSC", "USA + EM Small Cap (price)"),
              ("EM_EXIT", "USA + EM ex-IT")]
    for k, (c, lab) in enumerate(series):
        axes[0].plot(x, (1 + rets[c]).cumprod(), color=U.SERIES[k], lw=1.6, label=lab)
    axes[0].set_ylabel("Growth of $1", fontsize=9, color=U.INK_2)
    axes[0].set_title("Out-of-sample minimum-variance portfolios, monthly rebalanced, 2006-2026",
                      fontsize=11, color=U.INK, loc="left")
    U.legend(axes[0], loc="upper left")
    for k, (c, lab) in enumerate(series[1:], start=1):
        axes[1].plot(x, 100 * weights[c], color=U.SERIES[k], lw=1.6, label=lab.replace("USA + ", ""))
    axes[1].set_ylabel("Weight in EM index (%)", fontsize=9, color=U.INK_2)
    U.legend(axes[1], loc="upper left")
    fig.tight_layout()
    U.save_fig(fig, name)


def run(m):
    perf, weights, rets = evaluate(m)
    U.save_table(perf, "step4_portfolio_performance")
    U.save_table(weights.rename(columns=LABELS), "step4_weights_monthly")
    U.save_table(rets.rename(columns={**LABELS, "US_only": "USA only"}), "step4_portfolio_returns_monthly")
    plot_portfolios(rets, weights, "step4_portfolios")
    return {"performance": perf, "weights": weights}
