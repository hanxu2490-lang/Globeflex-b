"""Step 1 - Time-varying correlation (Question 1).

DCC-GARCH(1,1) estimated by maximum likelihood (Engle 2002, two-step) on weekly log returns
of MSCI EM, EM Small Cap, USA and USA IT. We track the correlations of EM and EM Small Cap
with USA and USA IT.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from . import utils as U
from .data_prep import LABELS

PAIRS = [("EM", "US"), ("EM", "US_IT"), ("EMSC", "US"), ("EMSC", "US_IT")]


def fit_garch(x):
    """Univariate GARCH(1,1) with constant mean; returns the fit and standardized residuals."""
    res = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    return res, res.resid / res.conditional_volatility


def _dcc_path(params, z, Qbar):
    a, b = params
    T, N = z.shape
    R = np.empty((T, N, N))
    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(z[t - 1], z[t - 1]) + b * Q
        d = 1 / np.sqrt(np.diag(Q))
        R[t] = Q * np.outer(d, d)
    return R


def _dcc_nll(params, z, Qbar):
    """Negative correlation part of the DCC log-likelihood."""
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10
    R = _dcc_path(params, z, Qbar)
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum("ti,tij,tj->t", z, np.linalg.inv(R), z)
    return 0.5 * np.sum(logdet + quad - np.sum(z ** 2, axis=1))


def _num_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_dcc(returns):
    """Two-step DCC(1,1): GARCH(1,1) per series, then (a, b) by ML. Returns dict of results."""
    garch, z = {}, {}
    for c in returns.columns:
        garch[c], z[c] = fit_garch(returns[c])
    Z = pd.DataFrame(z).dropna()
    zv = Z.values
    Qbar = np.cov(zv.T, bias=True)
    opt = minimize(_dcc_nll, x0=[0.03, 0.95], args=(zv, Qbar), method="SLSQP",
                   bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
                   constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}])
    a, b = opt.x
    try:
        se = np.sqrt(np.diag(np.linalg.inv(_num_hessian(lambda p: _dcc_nll(p, zv, Qbar), opt.x))))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    R = _dcc_path(opt.x, zv, Qbar)
    cols = list(Z.columns)
    corr = pd.DataFrame({f"{i}~{j}": R[:, cols.index(i), cols.index(j)] for i, j in PAIRS if i in cols and j in cols},
                        index=Z.index)
    garch_tab = pd.DataFrame({LABELS.get(c, c): {"mu": g.params["mu"], "omega": g.params["omega"],
                                                 "alpha": g.params["alpha[1]"], "beta": g.params["beta[1]"],
                                                 "loglik": g.loglikelihood}
                              for c, g in garch.items()}).T
    dcc_tab = pd.DataFrame({"estimate": [a, b, a + b], "std_err": [se[0], se[1], np.nan]},
                           index=["a", "b", "a+b"])
    dcc_tab.loc["corr_loglik", "estimate"] = -opt.fun
    dcc_tab.loc["weeks", "estimate"] = len(Z)
    return {"corr": corr, "garch": garch_tab, "dcc": dcc_tab, "converged": bool(opt.success)}


def pair_label(col):
    i, j = col.split("~")
    return f"{LABELS[i]} vs {LABELS[j]}"


def plot_corr(corr, name, title):
    fig, axes = U.new_fig(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, base in zip(axes, ["EM", "EMSC"]):
        for k, other in enumerate(["US", "US_IT"]):
            col = f"{base}~{other}"
            if col in corr:
                ax.plot(corr.index, corr[col], color=U.SERIES[k], lw=1.4, label=f"vs {LABELS[other]}")
        ax.set_title(f"{LABELS[base]}", fontsize=10, color=U.INK, loc="left")
        ax.set_ylim(0.2, 0.9)
    axes[0].set_ylabel("Weekly DCC correlation", fontsize=9, color=U.INK_2)
    fig.suptitle(title, fontsize=11, color=U.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    U.fig_legend(fig, axes[0], ncol=2)
    U.save_fig(fig, name)


def run(weekly):
    ret = weekly[["EM", "EMSC", "US", "US_IT"]].dropna()
    out = fit_dcc(ret)
    corr = out["corr"]
    U.save_table(out["garch"], "step1_garch_params")
    U.save_table(out["dcc"], "step1_dcc_params")
    U.save_table(corr.rename(columns=pair_label), "step1_dcc_correlations_weekly")
    annual = corr.groupby(corr.index.year).mean().rename(columns=pair_label)
    annual.index.name = "year"
    U.save_table(annual, "step1_dcc_correlations_annual")
    plot_corr(corr, "step1_dcc_correlations",
              "DCC-GARCH(1,1) correlation with US equities, weekly, 2001-2026")
    return {**out, "annual": annual}
