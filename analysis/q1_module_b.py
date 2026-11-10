"""Module B: out-of-sample portfolio validation and DCC-GARCH (Question 1).

Module A3 reports the in-sample mean-variance results. B uses the same A3 optimization convention only for its
own out-of-sample tests; it does not repeat A3's full-sample or rolling in-sample tables and figures.

Portfolio tests
- Monthly simple returns; arithmetic mean and covariance annualized by x12.
- Long-only, fully invested portfolios; same A3 SLSQP objective and constraints.
- MSCI EM and EM ex-IT: 60- and 120-month training windows; free weights and fixed 10%/20% exposures;
  next-month returns reported by five-year block.
- Country test: jointly optimize S&P 500, U.S. Aggregate and all eight EM countries using the previous
  60 months; first OOS month April 2020.

Dynamic-correlation tests
- Weekly and monthly univariate GARCH diagnostics (`arch` package) followed by pairwise DCC-GARCH.
- DCC persistence-cap sensitivity and 36-/60-month rolling correlation gaps.
"""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from arch import arch_model  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402

from . import config as C  # noqa: E402
from .q1_data import COUNTRIES, panel, weekly_log  # noqa: E402

OUT_DIR = C.OUT_DIR / "q1" / "module_B"
START, END = "2001-01-31", "2026-08-31"
COUNTRY_START = "2015-04-30"
BASE = ["S&P 500", "US Agg"]
COUNTRY_ASSETS = BASE + COUNTRIES
EXPOSURES = {"MSCI EM": "EM", "EM ex-IT": "EM_exIT"}
FIXED_OOS_SHARES = [0.10, 0.20]
GRID_POINTS = 2001
BLOCKS = {
    "2006-2010": ("2006-01-31", "2010-12-31"),
    "2011-2015": ("2011-01-31", "2015-12-31"),
    "2016-2020": ("2016-01-31", "2020-12-31"),
    "2021-2026-08": ("2021-01-31", END),
}


# ---------------------------------------------------------------- 2. A3 optimizer (same annualization and constraints)
def annualized_inputs(r, rf_window):
    """A3 convention: monthly arithmetic mean and covariance annualized by 12."""
    d = pd.concat([r, rf_window.rename("rf")], axis=1).dropna()
    if len(d) < 12:
        raise ValueError(f"Only {len(d)} complete months.")
    x = d[r.columns]
    return x.mean().to_numpy() * 12, x.cov().to_numpy() * 12, float(d["rf"].mean() * 12), x


def solve_a3(mu, cov, rf_ann, objective="sharpe", retry_start=None):
    """A3 SLSQP optimizer. Same objective/bounds/fully-invested constraint; retry from last successful weights if needed."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    cov = (cov + cov.T) / 2
    if objective == "sharpe":
        def loss(w):
            variance = float(w @ cov @ w)
            return 1e12 if variance <= 1e-14 else -float((w @ mu - rf_ann) / np.sqrt(variance))
    elif objective == "minvar":
        def loss(w):
            return float(w @ cov @ w)
    else:
        raise ValueError("objective must be 'sharpe' or 'minvar'")

    n = len(mu)
    equal = np.full(n, 1 / n)
    starts = [equal]
    if retry_start is not None:
        warm = np.asarray(retry_start, float)
        if warm.shape == (n,) and np.isfinite(warm).all():
            warm = np.clip(warm, 0, 1)
            if warm.sum() > 0:
                warm = warm / warm.sum()
                if not np.allclose(warm, equal):
                    starts.append(warm)

    feasible_results, messages = [], []
    for x0 in starts:
        res = minimize(
            loss, x0, method="SLSQP", bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 1500}
        )
        w = np.asarray(res.x, float)
        feasible = (np.isfinite(res.fun) and np.isfinite(w).all()
                    and (w >= -1e-8).all() and (w <= 1 + 1e-8).all()
                    and np.isclose(w.sum(), 1, atol=1e-7))
        messages.append(str(res.message))
        if feasible:
            feasible_results.append((bool(res.success), float(res.fun), w, str(res.message)))
        if res.success and feasible:
            break

    converged = [x for x in feasible_results if x[0]]
    if converged:
        _, _, w, message = min(converged, key=lambda x: x[1])
        return w, True, message
    if feasible_results:
        _, _, w, _ = min(feasible_results, key=lambda x: x[1])
        return w, False, "; retry did not converge: " + " | ".join(messages)
    return None, False, "No feasible SLSQP solution: " + " | ".join(messages)


def fixed_em_a3(mu, cov, rf_ann, share, objective="minvar"):
    """A3's 2,001-point grid with EM fixed and the U.S. stock/bond split optimized."""
    if len(mu) != 3:
        raise ValueError("Expected assets [S&P 500, US Agg, EM]")
    split = np.linspace(0, 1 - share, GRID_POINTS)
    weights = np.column_stack([split, 1 - share - split, np.full(GRID_POINTS, share)])
    returns = weights @ mu
    variances = np.maximum(np.einsum("ij,jk,ik->i", weights, cov, weights), 0)
    if objective == "sharpe":
        scores = np.divide(
            returns - rf_ann, np.sqrt(variances),
            out=np.full_like(returns, -np.inf), where=variances > 1e-14
        )
        chosen = int(np.nanargmax(scores))
    elif objective == "minvar":
        chosen = int(np.nanargmin(variances))
    else:
        raise ValueError("objective must be 'sharpe' or 'minvar'")
    w = weights[chosen]
    assert np.isclose(w.sum(), 1, atol=1e-10) and np.isclose(w[2], share, atol=1e-10)
    return w


def realized_metrics(r, rf_series):
    d = pd.concat([r.rename("r"), rf_series.rename("rf")], axis=1).dropna()
    if d.empty:
        return {"N months": 0, "Annual Return %": np.nan, "Annual Volatility %": np.nan,
                "Sharpe": np.nan, "Max Drawdown %": np.nan}
    wealth = (1 + d.r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    volatility = d.r.std(ddof=1) * np.sqrt(12) if len(d) > 1 else np.nan
    return {
        "N months": len(d),
        "Annual Return %": d.r.mean() * 1200,
        "Annual Volatility %": volatility * 100,
        "Sharpe": ((d.r.mean() - d.rf.mean()) * 12 / volatility if volatility and volatility > 0 else np.nan),
        "Max Drawdown %": drawdown.min() * 100,
    }


def _feasible(w):
    return (w is not None and np.isfinite(w).all()
            and (w >= -1e-8).all() and (w <= 1 + 1e-8).all()
            and np.isclose(w.sum(), 1, atol=1e-7))


# ---------------------------------------------------------------- 3. B out-of-sample test: MSCI EM and EM ex-IT
def rolling_oos(data, rf, asset_cols, lookback, strategy, objective, exposure=None, fixed_share=None):
    """Each month, weights from the previous `lookback` complete months, applied only to the next month."""
    d = data[asset_cols].dropna().sort_index()
    rows, weights, diagnostics = [], [], []
    previous_weights = None
    for j in range(lookback, len(d)):
        train, date = d.iloc[j-lookback:j], d.index[j]
        rf_train = rf.reindex(train.index)
        if rf_train.isna().any() or pd.isna(rf.get(date, np.nan)):
            continue
        mu, cov, rfa, _ = annualized_inputs(train, rf_train)
        if fixed_share is None:
            w, converged, status = solve_a3(mu, cov, rfa, objective, retry_start=previous_weights)
        else:
            w = fixed_em_a3(mu, cov, rfa, float(fixed_share), objective)
            converged, status = True, f"A3 fixed-share grid ({GRID_POINTS} points)"

        feasible = _feasible(w)
        diagnostics.append({"Date": date, "Lookback": lookback, "Strategy": strategy,
                            "Objective": objective, "Converged": bool(converged),
                            "Feasible": bool(feasible), "Status": status,
                            "Weight Sum": np.nan if w is None else float(np.sum(w)),
                            "Min Weight": np.nan if w is None else float(np.min(w))})
        if not converged or not feasible:
            continue

        previous_weights = w.copy()
        value = float(d.loc[date].to_numpy() @ w)
        rows.append({"Date": date, "Lookback": lookback, "Strategy": strategy,
                     "Objective": objective, "Exposure": exposure or "Benchmark",
                     "Fixed EM %": np.nan if fixed_share is None else fixed_share*100,
                     "Return": value, "RF": float(rf.loc[date]),
                     "Converged": True, "Optimization Status": status})
        weights += [{"Date": date, "Lookback": lookback, "Strategy": strategy,
                     "Objective": objective, "Asset": asset, "Weight": float(wi),
                     "Converged": True, "Optimization Status": status}
                    for asset, wi in zip(asset_cols, w)]
    return rows, weights, diagnostics


# ---------------------------------------------------------------- 5. GARCH diagnostics using `arch`
def fit_garch_one(s, asset, frequency):
    x = pd.Series(s).dropna().astype(float) * 100
    base_specs = [
        ("AR(2)-GARCH-t", "GARCH", 1, 0, 1, 2, "t"),
        ("AR(2)-GJR-GARCH-t", "GARCH", 1, 1, 1, 2, "t"),
    ]
    # Targeted alternatives for the only weekly series whose squared-residual and ARCH-LM tests failed.
    if frequency == "Weekly" and asset == "USA_IT":
        specs = base_specs + [
            ("AR(2)-EGARCH-t", "EGARCH", 1, 1, 1, 2, "t"),
            ("AR(1)-GARCH-skewt", "GARCH", 1, 0, 1, 1, "skewt"),
            ("AR(1)-GJR-GARCH-skewt", "GARCH", 1, 1, 1, 1, "skewt"),
            ("AR(2)-GJR-GARCH-skewt", "GARCH", 1, 1, 1, 2, "skewt"),
            ("AR(1)-ARCH(10)-t", "ARCH", 10, 0, 0, 1, "t"),
            ("AR(2)-ARCH(10)-t", "ARCH", 10, 0, 0, 2, "t"),
            ("AR(1)-ARCH(10)-skewt", "ARCH", 10, 0, 0, 1, "skewt"),
        ]
    else:
        specs = base_specs

    candidates = []
    for name, vol, p, o, q, mean_lags, dist in specs:
        try:
            fit = arch_model(x, mean="AR", lags=mean_lags, vol=vol,
                             p=p, o=o, q=q, dist=dist, rescale=False).fit(
                                 disp="off", show_warning=False, options={"maxiter": 3000})
            z = fit.std_resid.dropna()
            lb = float(acorr_ljungbox(z, lags=[10], model_df=mean_lags, return_df=True).lb_pvalue.iloc[0])
            lb2 = float(acorr_ljungbox(z**2, lags=[10], return_df=True).lb_pvalue.iloc[0])
            arch_p = float(het_arch(z, nlags=10, ddof=1 + mean_lags)[1])
            passed = [lb >= 0.05, lb2 >= 0.05, arch_p >= 0.05]
            row = {"Asset": asset, "Candidate": name, "Distribution": dist,
                   "Mean Lags": mean_lags, "N": len(z), "AIC": fit.aic, "BIC": fit.bic,
                   "Convergence": fit.convergence_flag == 0,
                   "LB Residual p(10)": lb, "LB Squared p(10)": lb2,
                   "ARCH LM p(10)": arch_p, "Diagnostics Passed": sum(passed),
                   "All Diagnostics Pass": all(passed)}
            candidates.append((row, z, fit))
        except Exception as exc:
            candidates.append(({"Asset": asset, "Candidate": name, "Distribution": dist,
                                "Mean Lags": mean_lags, "Convergence": False,
                                "Diagnostics Passed": 0, "All Diagnostics Pass": False,
                                "Error": str(exc)}, None, None))

    valid = [v for v in candidates if v[2] is not None and v[0]["Convergence"]]
    if not valid:
        raise RuntimeError(f"No converged GARCH candidate for {frequency} {asset}")
    passing = [v for v in valid if v[0]["All Diagnostics Pass"]]
    pool = passing if passing else valid
    chosen = min(pool, key=lambda v: (v[0]["BIC"] if passing else -v[0]["Diagnostics Passed"], v[0]["BIC"]))
    row = dict(chosen[0])
    row["Selected"] = True
    row["Selection Note"] = ("Lowest BIC among converged models passing all three diagnostics"
                             if passing else "No candidate passed all diagnostics; selected the converged model passing the most tests, then lowest BIC")
    return chosen[1], row, [v[0] for v in candidates], chosen[2]


def fit_garch_panel(panel_, frequency):
    std = pd.DataFrame(index=panel_.index)
    selected, candidates, fits = [], [], {}
    for asset in panel_:
        z, row, rows, fit = fit_garch_one(panel_[asset], asset, frequency)
        selected.append(row); candidates.extend(rows); fits[asset] = fit
        std.loc[z.index, asset] = z
    return std, pd.DataFrame(selected).set_index("Asset"), pd.DataFrame(candidates), fits


# ---------------------------------------------------------------- 6. DCC-GARCH(1,1), cap sensitivity, rolling correlations
def dcc_path(a, b, z, qbar):
    q = qbar.copy(); rho = np.empty(len(z)); c = 1 - a - b
    for t in range(len(z)):
        if t:
            q = c*qbar + a*np.outer(z[t-1], z[t-1]) + b*q
        rho[t] = q[0, 1] / np.sqrt(max(q[0, 0]*q[1, 1], 1e-14))
    return np.clip(rho, -0.999999, 0.999999)


def fit_dcc_pair(zdf, label, cap=0.998):
    d = zdf.dropna(); z = d.to_numpy(float)
    if len(z) < 80:
        raise ValueError(f"{label}: insufficient paired residuals ({len(z)})")
    qbar = np.cov(z, rowvar=False); qbar = (qbar + qbar.T)/2 + np.eye(2)*1e-8

    # Smooth transformation guarantees a>0, b>0 and a+b<cap at every optimizer step.
    def decode(theta):
        a = cap * expit(theta[0])
        b = (cap - a) * expit(theta[1])
        return float(a), float(b)

    def encode(a0, b0):
        aa = np.clip(a0/cap, 1e-6, 1-1e-6)
        bb = np.clip(b0/max(cap-a0, 1e-8), 1e-6, 1-1e-6)
        return np.array([logit(aa), logit(bb)], dtype=float)

    def objective(theta):
        a, b = decode(theta)
        rho = dcc_path(a, b, z, qbar)
        det = np.maximum(1-rho*rho, 1e-12)
        quad = (z[:, 0]**2 + z[:, 1]**2 - 2*rho*z[:, 0]*z[:, 1]) / det
        return float(0.5*np.sum(np.log(det) + quad - z[:, 0]**2 - z[:, 1]**2))

    starts = [(0.03, 0.94), (0.05, 0.88), (0.10, 0.75)]
    fits = []
    for a0, b0 in starts:
        if a0 + b0 >= cap:
            continue
        res = minimize(objective, encode(a0, b0), method="L-BFGS-B",
                       bounds=[(-12, 12), (-12, 12)],
                       options={"maxiter": 3000, "ftol": 1e-12, "maxls": 50})
        a, b = decode(res.x)
        if np.isfinite(res.fun):
            fits.append((bool(res.success), float(res.fun), a, b, str(res.message),
                         float(np.linalg.norm(getattr(res, "jac", np.array([np.nan]))))))

    converged = [x for x in fits if x[0]]
    pool = converged if converged else fits
    if not pool:
        raise RuntimeError(f"DCC produced no finite solution for {label} (cap={cap})")
    ok, nll, a, b, message, grad_norm = min(pool, key=lambda x: x[1])
    distance = cap - (a+b)
    row = {"Pair": label, "Cap": cap, "a": a, "b": b, "a+b": a+b,
           "Distance to Cap": distance, "NLL": nll, "N": len(d),
           "Convergence": ok, "Near Cap": bool(distance <= 0.002),
           "Gradient Norm": grad_norm, "Status": message}
    return row, pd.Series(dcc_path(a, b, z, qbar), index=d.index, name=label)


def dcc_panel(std, pairs, cap=0.998):
    rows, paths = [], {}
    for label, cols in pairs.items():
        row, path = fit_dcc_pair(std[cols], label, cap)
        rows.append(row); paths[label] = path
    return pd.DataFrame(rows).set_index("Pair"), pd.DataFrame(paths)


def optimizer_summary(d, group_cols):
    return (d.groupby(group_cols).Converged.agg(Attempts="count", Converged_Count="sum", Success_Rate="mean")
            .reset_index().assign(Failed=lambda x: x.Attempts-x.Converged_Count))


# ---------------------------------------------------------------- 1. data (Module A's monthly panel, q1_data)
def monthly_data():
    """B's monthly simple returns: S&P 500 and U.S. Aggregate, A's EM, EM ex-IT and countries, 3M T-bill (RF);
    and A's monthly log returns for the monthly GARCH / DCC and rolling correlations."""
    p = panel()
    monthly = pd.concat([p["us"], p["ret"][["EM", "EM_exIT", *COUNTRIES]], p["rf"].rename("RF")],
                        axis=1).loc[START:END].sort_index()
    monthly_log = p["lret"].loc[START:END].copy()
    return monthly, monthly_log


def _savefig(fig, name):
    fig.tight_layout(); fig.savefig(OUT_DIR / name, dpi=220); plt.close(fig)


def run():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)                       # no stale files from earlier B versions
    OUT_DIR.mkdir(parents=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.18, "figure.figsize": (10, 5)})
    monthly, monthly_log = monthly_data()
    rf = monthly["RF"].rename("rf")
    coverage = pd.DataFrame({c: {"First": s.first_valid_index(), "Last": s.last_valid_index(), "N": int(s.notna().sum())}
                             for c, s in monthly.items()}).T

    # ---------------------------------------------------------------- 3. OOS: MSCI EM and EM ex-IT
    OOS_ROWS, OOS_WEIGHTS, OOS_DIAGNOSTICS = [], [], []
    for lookback in [60, 120]:
        for obj in ["sharpe", "minvar"]:
            r, w, diag = rolling_oos(monthly, rf, BASE, lookback, f"Benchmark | {obj}", obj)
            OOS_ROWS += r; OOS_WEIGHTS += w; OOS_DIAGNOSTICS += diag
        for exposure, col in EXPOSURES.items():
            assets = BASE + [col]
            for obj in ["sharpe", "minvar"]:
                r, w, diag = rolling_oos(monthly, rf, assets, lookback, f"{exposure} | Free | {obj}", obj, exposure)
                OOS_ROWS += r; OOS_WEIGHTS += w; OOS_DIAGNOSTICS += diag
            for share in FIXED_OOS_SHARES:
                for obj in ["sharpe", "minvar"]:
                    label = f"{exposure} | Fixed {int(share*100)}% | {obj}"
                    r, w, diag = rolling_oos(monthly, rf, assets, lookback, label, obj, exposure, share)
                    OOS_ROWS += r; OOS_WEIGHTS += w; OOS_DIAGNOSTICS += diag

    oos_returns = pd.DataFrame(OOS_ROWS).sort_values(["Lookback", "Strategy", "Date"]).reset_index(drop=True)
    oos_weights = pd.DataFrame(OOS_WEIGHTS)
    oos_optimizer_diagnostics = pd.DataFrame(OOS_DIAGNOSTICS)
    if oos_returns.empty:
        raise RuntimeError("No converged EM OOS results were produced.")

    # 3.1 OOS performance by the same five-year reporting blocks
    oos_rows = []
    for (lookback, strategy), g in oos_returns.groupby(["Lookback", "Strategy"]):
        for block, (a, z) in BLOCKS.items():
            h = g.set_index("Date").loc[a:z]
            m = realized_metrics(h["Return"], h["RF"])
            oos_rows.append({"Lookback": lookback, "Strategy": strategy, "Block": block,
                             "First month": h.index.min() if len(h) else pd.NaT,
                             "Last month": h.index.max() if len(h) else pd.NaT, **m})
    oos_block_summary = pd.DataFrame(oos_rows)
    oos_overall = []
    for (lookback, strategy), g in oos_returns.groupby(["Lookback", "Strategy"]):
        oos_overall.append({"Lookback": lookback, "Strategy": strategy, "Block": "All available OOS",
                            "First month": g.Date.min(), "Last month": g.Date.max(),
                            **realized_metrics(g.set_index("Date").Return, g.set_index("Date").RF)})
    oos_summary = pd.concat([pd.DataFrame(oos_overall), oos_block_summary], ignore_index=True)

    # OOS wealth chart for free-weight Minimum Variance, with 60- and 120-month training windows.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, lb in zip(axes, [60, 120]):
        g = oos_returns[(oos_returns.Lookback == lb) & (oos_returns.Strategy.str.contains("minvar"))]
        for name, h in g.groupby("Strategy"):
            h = h.sort_values("Date"); wealth = (1 + h.Return).cumprod()
            ax.plot(h.Date, wealth, label=name, lw=1.5)
        ax.set(title=f"Out-of-sample cumulative wealth — {lb}-month estimation window", ylabel="$1 growth")
        ax.legend(frameon=False, ncol=3, fontsize=8)
    _savefig(fig, "B_04_OOS_Wealth_60_120M.png")

    # Fixed 5-year block comparison: annualized Sharpe by lookback and exposure.
    plot_data = oos_block_summary[oos_block_summary.Strategy.str.contains("Fixed")].copy()
    for metric in ["Sharpe", "Annual Volatility %"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), sharey=True)
        for ax, (lb, part) in zip(axes, plot_data.groupby("Lookback")):
            pv = part.pivot_table(index="Block", columns="Strategy", values=metric)
            pv.plot(kind="bar", ax=ax, width=.8)
            ax.set(title=f"{metric} by five-year block — {lb}-month lookback", xlabel="OOS block")
            ax.tick_params(axis="x", rotation=0); ax.legend(frameon=False, fontsize=7)
        _savefig(fig, f"B_05_Fixed10_20_OOS_{metric.replace(' ', '_')}.png")

    # ---------------------------------------------------------------- 4. country OOS: joint optimization of all 10 assets
    country_data = monthly[COUNTRY_ASSETS].loc[COUNTRY_START:END].dropna().sort_index()
    country_oos_rows, country_oos_w, country_diag_rows = [], [], []
    previous_country_weights = {"sharpe": None, "minvar": None}
    for j in range(60, len(country_data)):
        train, date = country_data.iloc[j-60:j], country_data.index[j]
        rf_train = rf.reindex(train.index)
        if rf_train.isna().any() or pd.isna(rf.get(date, np.nan)):
            continue
        mu, cov, rfa, _ = annualized_inputs(train, rf_train)
        for obj in ["sharpe", "minvar"]:
            w, converged, status = solve_a3(mu, cov, rfa, obj, retry_start=previous_country_weights[obj])
            feasible = _feasible(w)
            country_diag_rows.append({"Date": date, "Objective": obj, "Converged": bool(converged),
                                      "Feasible": bool(feasible), "Status": status,
                                      "Weight Sum": np.nan if w is None else float(np.sum(w)),
                                      "Min Weight": np.nan if w is None else float(np.min(w))})
            if not converged or not feasible:
                continue
            previous_country_weights[obj] = w.copy()
            country_oos_rows.append({"Date": date, "Objective": obj,
                                     "Return": float(country_data.loc[date].to_numpy() @ w),
                                     "RF": float(rf.loc[date]), "Converged": True, "Status": status})
            country_oos_w += [{"Date": date, "Objective": obj, "Asset": asset,
                               "Weight": float(wi), "Converged": True, "Status": status}
                              for asset, wi in zip(COUNTRY_ASSETS, w)]

    country_oos = pd.DataFrame(country_oos_rows)
    country_oos_weights = pd.DataFrame(country_oos_w)
    country_optimizer_diagnostics = pd.DataFrame(country_diag_rows)
    if country_oos.empty:
        raise RuntimeError("No converged country OOS observations produced.")
    country_oos_summary = pd.DataFrame([
        {"Objective": obj, "First month": g.Date.min(), "Last month": g.Date.max(),
         **realized_metrics(g.set_index("Date").Return, g.set_index("Date").RF)}
        for obj, g in country_oos.groupby("Objective")])
    country_oos_blocks = pd.DataFrame([
        {"Objective": obj, "Block": block, "First month": h.Date.min() if len(h) else pd.NaT,
         "Last month": h.Date.max() if len(h) else pd.NaT,
         **realized_metrics(h.set_index("Date").Return, h.set_index("Date").RF)}
        for obj, g in country_oos.groupby("Objective") for block, (a, z) in BLOCKS.items()
        for h in [g.set_index("Date").loc[a:z].reset_index()]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, obj in zip(axes, ["sharpe", "minvar"]):
        pv = country_oos_weights[country_oos_weights.Objective == obj].pivot(index="Date", columns="Asset", values="Weight")
        pv.plot.area(ax=ax, linewidth=0)
        ax.set(title=f"Joint 10-asset OOS weights — {obj}", ylabel="Portfolio weight", xlabel="Date", ylim=(0, 1))
        ax.legend(frameon=False, fontsize=7, ncol=2)
    _savefig(fig, "B_06_Country_Joint_OOS_Weights.png")

    # ---------------------------------------------------------------- 5. GARCH: weekly (daily levels -> Friday, weeks 2001-01 to 2026-08) and monthly (A's log returns)
    weekly_log_ = weekly_log()
    monthly_garch_log = monthly_log[[c for c in ["EM", "EM_exIT", "EM_exTWKR", "USA", "USA_IT"] if c in monthly_log]].copy()
    weekly_std, weekly_diag, weekly_candidates, weekly_fits = fit_garch_panel(weekly_log_, "Weekly")
    monthly_std, monthly_diag, monthly_candidates, monthly_fits = fit_garch_panel(monthly_garch_log, "Monthly")

    # ---------------------------------------------------------------- 6. DCC
    weekly_pairs = {"EM vs USA": ["EM", "USA"], "EM vs USA IT": ["EM", "USA_IT"],
                    "EM Small Cap vs USA": ["EM Small Cap Price", "USA"],
                    "EM Small Cap vs USA IT": ["EM Small Cap Price", "USA_IT"]}
    monthly_pairs = {"EM vs USA": ["EM", "USA"], "EM ex-IT vs USA": ["EM_exIT", "USA"],
                     "EM ex-TWKR vs USA": ["EM_exTWKR", "USA"], "EM vs USA IT": ["EM", "USA_IT"],
                     "EM ex-IT vs USA IT": ["EM_exIT", "USA_IT"],
                     "EM ex-TWKR vs USA IT": ["EM_exTWKR", "USA_IT"]}
    weekly_dcc_params, weekly_dcc = dcc_panel(weekly_std, weekly_pairs, 0.998)
    monthly_dcc_params, monthly_dcc = dcc_panel(monthly_std, monthly_pairs, 0.998)
    cap_sensitivity = pd.concat([dcc_panel(weekly_std, weekly_pairs, c)[0].reset_index()
                                 for c in [0.995, 0.998, 0.999]], ignore_index=True)

    roll_rows = []
    for win in [36, 60]:
        for benchmark in ["USA", "USA_IT"]:
            broad = monthly_log["EM"].rolling(win).corr(monthly_log[benchmark])
            for alternative in ["EM_exIT", "EM_exTWKR"]:
                alt = monthly_log[alternative].rolling(win).corr(monthly_log[benchmark])
                roll_rows.append(pd.DataFrame({"Window": win, "Benchmark": benchmark,
                                               "Alternative": alternative, "Broad EM Corr": broad,
                                               "Alternative Corr": alt, "Gap": alt-broad}))
    rolling_corr = pd.concat(roll_rows).reset_index(names="Date")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for name, g in rolling_corr[(rolling_corr.Window == 60) & (rolling_corr.Benchmark == "USA")].groupby("Alternative"):
        ax.plot(g.Date, g.Gap, label=f"{name} minus Broad EM")
    ax.axhline(0, color="black", lw=.8)
    ax.set(title="60-month rolling correlation gap vs U.S. market", xlabel="Date", ylabel="Correlation gap")
    ax.legend(frameon=False); _savefig(fig, "B_09_Rolling_Correlation_Gaps.png")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for c in weekly_dcc:
        ax.plot(weekly_dcc.index, weekly_dcc[c], lw=1, label=c)
    ax.set(title="Weekly DCC-GARCH conditional correlations", xlabel="Date", ylabel="Conditional correlation")
    ax.legend(frameon=False, ncol=2, fontsize=8); _savefig(fig, "B_07_Weekly_DCC.png")
    for benchmark, cols in [("USA", ["EM vs USA", "EM ex-IT vs USA", "EM ex-TWKR vs USA"]),
                            ("USA IT", ["EM vs USA IT", "EM ex-IT vs USA IT", "EM ex-TWKR vs USA IT"])]:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for c in cols:
            ax.plot(monthly_dcc.index, monthly_dcc[c], lw=1.1, label=c)
        ax.set(title=f"Monthly DCC-GARCH vs {benchmark}", xlabel="Date", ylabel="Conditional correlation")
        ax.legend(frameon=False, fontsize=8)
        _savefig(fig, f"B_08_Monthly_DCC_{benchmark.replace(' ', '_')}.png")

    # ---------------------------------------------------------------- 7. export results and verify constraints
    oos_dates = oos_returns.groupby("Lookback").Date.agg(["min", "max", "nunique"]).reset_index()
    country_dates = pd.DataFrame([{
        "First OOS month": country_oos.Date.min(), "Last OOS month": country_oos.Date.max(),
        "N OOS months": country_oos.Date.nunique(), "Training lookback": 60,
    }])
    oos_optimizer_summary = optimizer_summary(oos_optimizer_diagnostics, ["Lookback", "Strategy", "Objective"])
    country_optimizer_summary = optimizer_summary(country_optimizer_diagnostics, ["Objective"])

    readme = pd.DataFrame({
        "Item": ["Return convention", "Optimizer convention", "EM / EM ex-IT OOS", "Fixed EM OOS",
                 "Country OOS", "GARCH / DCC", "Optimizer audit", "A3 overlap"],
        "Definition": [
            "Monthly simple returns; arithmetic mean and covariance annualized by 12; monthly 3M T-bill mean annualized by 12",
            "A3 SLSQP objective, long-only bounds and fully-invested constraint; retries only after failed convergence using prior successful weights",
            "MSCI EM and EM ex-IT; 60/120-month estimation windows; weights applied to the following month",
            "Fixed 10% and 20% EM; Maximum Sharpe and Minimum Variance; remaining weight split by A3 grid",
            "S&P 500 + US Agg + eight countries jointly optimized; 60-month training; April 2020–August 2026; 77 expected observations before any failed optimizations",
            "Univariate GARCH diagnostics; weekly/monthly DCC using transformed parameters; persistence-cap sensitivity; 36/60-month rolling correlation gaps",
            "Every SLSQP attempt is logged; non-converged or infeasible results are excluded from OOS realized returns and reported in diagnostics sheets",
            "A3's in-sample tables, figures, and window-by-window optimization outputs are not rerun or exported in B",
        ],
    })
    sheets = {
        "README": readme, "Coverage": coverage.reset_index().rename(columns={"index": "Asset"}),
        "OOS_By_5Y_Block": oos_block_summary, "OOS_All_Periods": oos_summary,
        "OOS_Date_Coverage": oos_dates, "OOS_Optimizer_Audit": oos_optimizer_summary,
        "OOS_Optimizer_Details": oos_optimizer_diagnostics,
        "OOS_Country_Joint": country_oos_summary, "OOS_Country_By_5Y_Block": country_oos_blocks,
        "OOS_Country_Dates": country_dates, "OOS_Country_Weights": country_oos_weights,
        "Country_Optimizer_Audit": country_optimizer_summary,
        "Country_Optimizer_Details": country_optimizer_diagnostics,
        "Weekly_GARCH": weekly_diag.reset_index(), "Weekly_GARCH_Candidates": weekly_candidates,
        "Monthly_GARCH": monthly_diag.reset_index(), "Monthly_GARCH_Candidates": monthly_candidates,
        "Weekly_DCC_Params": weekly_dcc_params.reset_index(), "Weekly_DCC_Cap_Sensitivity": cap_sensitivity,
        "Monthly_DCC_Params": monthly_dcc_params.reset_index(), "Rolling_Correlations": rolling_corr,
    }
    with pd.ExcelWriter(OUT_DIR / "B_Module_Results.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]; ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    csv_outputs = {
        "OOS_Returns": oos_returns, "OOS_Weights": oos_weights,
        "OOS_Optimizer_Diagnostics": oos_optimizer_diagnostics,
        "OOS_Country_Returns": country_oos, "OOS_Country_Weights": country_oos_weights,
        "OOS_Country_Optimizer_Diagnostics": country_optimizer_diagnostics,
        "Weekly_DCC_Full": weekly_dcc, "Monthly_DCC_Full": monthly_dcc,
        "Rolling_Correlations": rolling_corr,
    }
    for name, frame in csv_outputs.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False if "Date" in frame.columns else True)

    # Hard checks: only converged/feasible OOS weights are applied; all portfolios are fully invested.
    assert not oos_weights.empty and np.isfinite(oos_weights.Weight).all()
    assert np.isclose(oos_weights.groupby(["Lookback", "Strategy", "Date"]).Weight.sum().to_numpy(), 1, atol=1e-6).all()
    assert (oos_weights.Weight >= -1e-8).all()
    for _, group in country_oos_weights.groupby(["Date", "Objective"]):
        assert np.isclose(group.Weight.sum(), 1, atol=1e-6) and (group.Weight >= -1e-8).all()
    for _, group in oos_weights[oos_weights.Strategy.str.contains("Fixed")].groupby(["Lookback", "Strategy", "Date"]):
        target = float(group.Strategy.iloc[0].split("Fixed ")[1].split("%")[0]) / 100
        em_weight = group.loc[group.Asset.isin(["EM", "EM_exIT"]), "Weight"].sum()
        assert np.isclose(em_weight, target, atol=1e-8)
    assert country_oos.Date.min() == pd.Timestamp("2020-04-30")
    assert country_oos.Date.max() == pd.Timestamp(END)
    assert oos_returns[oos_returns.Lookback == 60].Date.min() == pd.Timestamp("2006-01-31")
    assert oos_returns[oos_returns.Lookback == 120].Date.min() == pd.Timestamp("2011-01-31")

    return {"coverage": coverage, "oos_block_summary": oos_block_summary, "oos_summary": oos_summary,
            "oos_optimizer_summary": oos_optimizer_summary, "country_oos_summary": country_oos_summary,
            "country_oos_blocks": country_oos_blocks, "country_optimizer_summary": country_optimizer_summary,
            "weekly_garch": weekly_diag, "monthly_garch": monthly_diag,
            "weekly_dcc_params": weekly_dcc_params, "monthly_dcc_params": monthly_dcc_params,
            "cap_sensitivity": cap_sensitivity, "rolling_corr": rolling_corr}
