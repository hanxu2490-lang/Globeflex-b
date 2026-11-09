"""Module A3: mean-variance optimization with EM (Question 1).

Does adding EM improve a U.S. stock-bond portfolio, does that depend on tech, and which EM countries
would an optimizer choose?

| Step | Question | Assets | Sample, windows | Output |
|---|---|---|---|---|
| 1 | What is the baseline? | S&P 500 + U.S. Aggregate | 2001-01...2026-08; rolling 5-year (21) and 10-year (16) | T2 |
| 2 | Does adding MSCI EM help, at different times and amounts? | + MSCI EM fixed at 0%, 10%, ..., 100% | same | Fig 1-2, T3-T4 |
| 3 | Is it about tech? | same with EM ex-IT instead of MSCI EM | same | Fig 1-2, T3-T4 |
| 4 | Which EM countries would the optimizer choose? | S&P 500 + U.S. Aggregate + 8 countries, weights free | 2015-04...2026-08; rolling 5-year (7) | Fig 3-4, T5-T6 |

S&P 500 total return (SPXT), Bloomberg U.S. Aggregate total return (LBUSTRUU), MSCI net USD indices.
Monthly simple returns; mean and covariance annualized (x 12, arithmetic). Risk-free rate = average 3-month
T-bill in the window. Long-only, weights sum to 1. Rolling windows move one year at a time; every window ends
in August. Steps 2-3: EM is fixed at x; the optimizer chooses the split of the remaining 1 - x between the
S&P 500 and U.S. Aggregate on a 2,001-point grid; the max-Sharpe and min-variance portfolios are reported.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from . import config as C  # noqa: E402
from .q1_data import COUNTRIES, END, GOLD, GREY, NAVY, panel  # noqa: E402

START, START_C = "2001-01", "2015-04"          # steps 1-3 / step 4
WINDOWS_Y = [5, 10]                            # rolling window lengths (years), steps 1-3
EM_STEPS = np.round(np.arange(0, 1.01, 0.10), 2)
GRID = 2001                                    # grid points for the S&P 500 / U.S. Agg split
US, VERSIONS = ["S&P 500", "US Agg"], ["MSCI EM", "EM ex-IT"]
PALETTE = dict(zip(US + COUNTRIES, [NAVY, "#B9C2D0", GOLD, "#4F7CAC", "#C0392B", "#E67E22", "#27AE60", "#8E44AD", "#16A085", "#7F8C8D"]))
OUT = C.OUT_DIR / "q1" / "module_A3"


def _save(name, dpi=150):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / name, dpi=dpi, bbox_inches="tight"); plt.close("all")


def data():
    """A3's return panel: S&P 500, U.S. Agg, MSCI EM, EM IT, countries, EM ex-IT; and the risk-free rate."""
    p = panel()
    ret = pd.concat([p["us"], p["ret"][["EM", "EM_IT"]].rename(columns={"EM": "MSCI EM", "EM_IT": "EM IT"}),
                     p["ret"][COUNTRIES]], axis=1).loc[START:END]
    w_it = p["TS_ALL"].shift(1).reindex(ret.index)
    ret["EM ex-IT"] = ((ret["MSCI EM"] - w_it * ret["EM IT"]) / (1 - w_it)).where(w_it.notna())
    return ret, p["rf"]


def windows(start, years):
    first_end = pd.Timestamp(start) + pd.DateOffset(years=years)
    first_end = pd.Timestamp(f"{first_end.year if first_end.month <= 8 else first_end.year + 1}-08-31")
    ends = pd.date_range(first_end, pd.Period(END, "M").end_time.normalize(), freq="12ME")
    return {f"{(e - pd.DateOffset(months=12 * years - 1)):%Y-%m}…{e:%Y-%m}":
            ((e - pd.DateOffset(months=12 * years - 1)).strftime("%Y-%m"), e.strftime("%Y-%m")) for e in ends}


def solve(mu, S, r_f, kind="sharpe"):
    """General optimizer (SLSQP) for step 4: max Sharpe or min variance, long-only, weights sum to 1."""
    n = len(mu)
    f = (lambda w: -(w @ mu - r_f) / np.sqrt(w @ S @ w)) if kind == "sharpe" else (lambda w: w @ S @ w)
    res = minimize(f, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def run():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    ret, rf = data()
    results = {}
    WIN = {y: windows(START, y) for y in WINDOWS_Y}
    WIN_C = windows(START_C, 5)
    results["T0_coverage"] = pd.DataFrame({"first month": ret.apply(lambda s: s.first_valid_index().strftime("%Y-%m")),
                                           "months": ret.count()})

    def inputs(assets, a, z):
        r = ret.loc[a:z, assets].dropna()
        return r.mean().values * 12, r.cov().values * 12, rf.loc[a:z].mean() * 12

    def fixed_em(version, x, a, z):
        """EM fixed at x; the S&P 500 share s runs over [0, 1 - x] on a fine grid (U.S. Agg = 1 - x - s)."""
        mu, S, r_f = inputs(US + [version], a, z)
        s = np.linspace(0, 1 - x, GRID if x < 1 else 1)
        W = np.column_stack([s, 1 - x - s, np.full_like(s, x)])
        pr, pv = W @ mu, np.sqrt(np.einsum("ij,jk,ik->i", W, S, W))
        return pd.DataFrame({"spx": s, "agg": 1 - x - s, "ret": pr, "vol": pv, "sharpe": (pr - r_f) / pv})

    def summarize(df):
        i, j = df["sharpe"].idxmax(), df["vol"].idxmin()
        return {"max-Sharpe Sharpe": df.loc[i, "sharpe"], "max-Sharpe return %": df.loc[i, "ret"] * 100, "max-Sharpe vol %": df.loc[i, "vol"] * 100,
                "max-Sharpe S&P 500 %": df.loc[i, "spx"] * 100, "max-Sharpe US Agg %": df.loc[i, "agg"] * 100,
                "min-var vol %": df.loc[j, "vol"] * 100, "min-var return %": df.loc[j, "ret"] * 100}

    def efficient(df):
        return df[df["ret"] >= df.loc[df["vol"].idxmin(), "ret"]].sort_values("vol")

    # ---------------------------------------------------------------- step 1: inputs and baseline
    def describe(a, z):
        r = ret.loc[a:z, US + VERSIONS]
        return pd.DataFrame({"return %": r.mean() * 1200, "vol %": r.std() * np.sqrt(12) * 100, "corr S&P 500": r.corrwith(r["S&P 500"])})
    t1 = pd.concat({(f"{y}y", w): describe(a, z) for y in WINDOWS_Y for w, (a, z) in WIN[y].items()}).round(2)
    t2 = pd.DataFrame({(f"{y}y", w): summarize(fixed_em("MSCI EM", 0.0, a, z)) for y in WINDOWS_Y for w, (a, z) in WIN[y].items()}).T.round(2)
    results["T1_inputs"] = t1; results["T2_baseline"] = t2

    # ---------------------------------------------------------------- steps 2-3: EM fixed at 0%, 10%, ..., 100%
    rows = []
    for y in WINDOWS_Y:
        for w, (a, z) in WIN[y].items():
            for v in VERSIONS:
                for x in EM_STEPS:
                    rows.append({"window length": f"{y}y", "window": w, "version": v, "EM %": int(round(x * 100)), **summarize(fixed_em(v, x, a, z))})
    t3 = pd.DataFrame(rows).set_index(["window length", "window", "version", "EM %"])
    base = t3.xs(0, level="EM %")
    t3["Sharpe gain"] = t3["max-Sharpe Sharpe"] - base["max-Sharpe Sharpe"].reindex(t3.index.droplevel("EM %")).values
    t3["min-var vol change"] = t3["min-var vol %"] - base["min-var vol %"].reindex(t3.index.droplevel("EM %")).values
    t3 = t3.round(3)

    def best(g):
        g = g.droplevel([0, 1, 2])
        return pd.Series({"best EM % (Sharpe)": g["max-Sharpe Sharpe"].idxmax(), "Sharpe gain at best": g["Sharpe gain"].max(),
                          "Sharpe gain at 10%": g.loc[10, "Sharpe gain"],
                          "best EM % (min var)": g["min-var vol %"].idxmin(), "vol change at best (pp)": g["min-var vol change"].min(),
                          "vol change at 10% (pp)": g.loc[10, "min-var vol change"]})
    t4 = t3.groupby(level=["window length", "window", "version"], sort=False).apply(best).round(3)
    results["T3_fixed_EM_all"] = t3; results["T4_best_share"] = t4

    # Fig 1: efficient frontiers, every window; one curve per EM share (light = 0%, dark = 100%)
    cmap = plt.get_cmap("viridis")
    for y in WINDOWS_Y:
        wins = list(WIN[y]); n = len(wins); cols = 4; rws = int(np.ceil(n / cols))
        for v in VERSIONS:
            fig, axes = plt.subplots(rws, cols, figsize=(16, 3.2 * rws), sharex=True, sharey=True)
            for ax, w in zip(axes.flat, wins):
                a, z = WIN[y][w]
                for k, x in enumerate(EM_STEPS):
                    df = fixed_em(v, x, a, z); e = efficient(df); col = cmap(k / (len(EM_STEPS) - 1))
                    ax.plot(e["vol"] * 100, e["ret"] * 100, color=col, lw=1.2)
                    i = df["sharpe"].idxmax(); ax.scatter(df.loc[i, "vol"] * 100, df.loc[i, "ret"] * 100, color=col, s=10, zorder=3)
                ax.set_title(w, fontsize=9)
            for ax in axes.flat[n:]:
                ax.axis("off")
            fig.suptitle(f"Fig 1. Efficient frontiers with {v} fixed at 0–100% ({y}-year windows)  ·  x = volatility %, y = return %", y=1.0)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)); fig.colorbar(sm, ax=axes, fraction=0.015, label="EM share %")
            _save(f'fig1_frontiers_{v.replace(" ", "_")}_{y}y.png')

    # Fig 2: Sharpe gain by EM share over time
    for y in WINDOWS_Y:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        for ax, v in zip(axes, VERSIONS):
            g = t3.xs((f"{y}y", v), level=["window length", "version"])["Sharpe gain"].unstack("EM %")
            for k, x in enumerate(EM_STEPS[1:]):
                ax.plot(range(len(g)), g[int(round(x * 100))], color=cmap((k + 1) / (len(EM_STEPS) - 1)), lw=1.4, label=f"{int(round(x * 100))}%")
            ax.axhline(0, color="black", lw=0.8); ax.set_xticks(range(len(g))); ax.set_xticklabels([w[-7:] for w in g.index], rotation=90, fontsize=7)
            ax.set_title(f"{v}: Sharpe gain vs. no EM ({y}-year windows, by window end)")
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        _save(f"fig2_sharpe_gain_{y}y.png")

    # ---------------------------------------------------------------- step 4: countries
    ASSETS = US + COUNTRIES
    WIN4 = {**WIN_C, f"{START_C}…{END} (full)": (START_C, END)}
    rows, wts = {}, {}
    for w, (a, z) in WIN4.items():
        mu_b, S_b, r_f = inputs(US, a, z); mu_e, S_e, _ = inputs(ASSETS, a, z)
        wb, we, ve = solve(mu_b, S_b, r_f), solve(mu_e, S_e, r_f), solve(mu_e, S_e, r_f, "minvar")
        vb = solve(mu_b, S_b, r_f, "minvar")
        sh = lambda w_, m, S: (w_ @ m - r_f) / np.sqrt(w_ @ S @ w_)
        rows[w] = {"Sharpe baseline": sh(wb, mu_b, S_b), "Sharpe with countries": sh(we, mu_e, S_e), "Sharpe gain": sh(we, mu_e, S_e) - sh(wb, mu_b, S_b),
                   "EM total, max Sharpe %": we[2:].sum() * 100, "EM total, min var %": ve[2:].sum() * 100,
                   "min-var vol baseline %": np.sqrt(vb @ S_b @ vb) * 100, "min-var vol with countries %": np.sqrt(ve @ S_e @ ve) * 100}
        wts[("max Sharpe", w)] = pd.Series(we, ASSETS) * 100; wts[("min variance", w)] = pd.Series(ve, ASSETS) * 100
    t5 = pd.DataFrame(rows).T.round(3); t5w = pd.DataFrame(wts).T.round(1)
    t6 = pd.DataFrame({w: ret.loc[a:z, ASSETS].corrwith(ret.loc[a:z, "S&P 500"]) for w, (a, z) in WIN4.items()}).T.round(2)
    results["T5_countries_summary"] = t5; results["T5b_countries_weights"] = t5w; results["T6_country_corr_SP500"] = t6

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, k in zip(axes, ["max Sharpe", "min variance"]):
        t5w.loc[k].plot.bar(stacked=True, ax=ax, color=[PALETTE[c] for c in ASSETS], width=0.75, legend=False)
        ax.set(title=f"Fig 3. Weights, {k} (%)", xlabel="", ylim=(0, 100)); ax.tick_params(axis="x", rotation=60)
    axes[1].legend(ASSETS, frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save("fig3_country_weights.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True, sharey=True)
    for ax, (w, (a, z)) in zip(axes.flat, WIN4.items()):
        for assets, col, lab in [(US, GREY, "S&P 500 + US Agg"), (ASSETS, GOLD, "+ 8 EM countries")]:
            mu, S, _ = inputs(assets, a, z); lo, hi = solve(mu, S, 0, "minvar") @ mu, mu.max(); pts = []
            for t in np.linspace(lo, hi, 30):
                res = minimize(lambda w_: w_ @ S @ w_, np.full(len(mu), 1 / len(mu)), method="SLSQP", bounds=[(0, 1)] * len(mu),
                               constraints=[{"type": "eq", "fun": lambda w_: w_.sum() - 1}, {"type": "eq", "fun": lambda w_, t=t: w_ @ mu - t}])
                pts.append((np.sqrt(res.x @ S @ res.x) * 100, t * 100))
            ax.plot(*zip(*pts), color=col, lw=1.6, label=lab)
        ax.set_title(w, fontsize=9)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fig 4. Efficient frontier with and without the 8 EM countries (x = volatility %, y = return %)")
    _save("fig4_country_frontiers.png")

    # ---------------------------------------------------------------- save
    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "module_A3_optimizer_results.xlsx") as xl:
        pd.DataFrame({"setting": {"steps 1-3": f"{START}…{END}; rolling {WINDOWS_Y} years, one window ending each August; EM fixed 0–100% in 10% steps; S&P 500 / U.S. Agg split chosen by grid search",
                                  "step 4": f"{START_C}…{END}; rolling 5 years; S&P 500, U.S. Agg and 8 countries, weights free (SLSQP)",
                                  "returns": "monthly simple returns; mean and covariance annualized x12",
                                  "risk-free": "average 3-month T-bill in window", "constraints": "long-only, weights sum to 1"}}).to_excel(xl, sheet_name="README")
        for name, df in results.items():
            df.to_excel(xl, sheet_name=name[:31])
    return results
