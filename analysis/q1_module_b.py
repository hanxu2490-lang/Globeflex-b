"""Module B: A3 optimizer extensions, out-of-sample backtests and DCC-GARCH (Question 1).

Portfolio optimization follows the Module A3 optimizer: annualized arithmetic mean and covariance (monthly x12),
long-only SLSQP with weights summing to one for free-weight portfolios, and A3's 2,001-point stock/bond split grid
for fixed EM allocations. The eight-country universe uses the same joint S&P 500 + U.S. Aggregate + eight-country
optimizer as A3 Step 4. B adds rolling out-of-sample application without changing A3's optimizer definition.
DCC-GARCH: univariate mean/volatility models chosen by residual diagnostics and BIC (weekly USA_IT also tests
AR-ARCH(5/10)-t), bivariate DCC(1,1) with a 0.998 persistence cap, re-estimated at 0.995/0.998/0.999 as a check.
No structural break is applied in this module.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, gammaln, logit  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402

from . import config as C  # noqa: E402
from .q1_data import BLUE, COUNTRIES, GOLD, GREY, NAVY, RED, panel, weekly_log  # noqa: E402

OUT_DIR = C.OUT_DIR / "q1" / "module_B"
START = "2001-01-31"
END = "2026-08-31"
WINDOW_YEARS = [5, 10]
EM_STEPS = np.round(np.arange(0.0, 1.01, 0.10), 2)
GRID_POINTS = 2001
LOOKBACK_MONTHS = 60
BASE_ASSETS = ["S&P 500", "US Agg"]
A3_OPTIMIZER_STATUS_LOG = []

def coverage_table(df):
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        rows.append({"Series": c, "First month": s.index.min() if len(s) else pd.NaT,
                     "Last month": s.index.max() if len(s) else pd.NaT, "N months": len(s)})
    return pd.DataFrame(rows).set_index("Series")


def annualized_inputs(returns, rf_monthly):
    """A3 convention: arithmetic mean x12, covariance x12, annualized monthly risk-free x12."""
    aligned = pd.concat([returns, rf_monthly.rename("rf")], axis=1).dropna()
    if len(aligned) < 12:
        raise ValueError(f"Only {len(aligned)} complete monthly observations; at least 12 required.")
    r = aligned[returns.columns]
    mu = r.mean().to_numpy() * 12
    cov = r.cov().to_numpy() * 12
    rf_ann = float(aligned["rf"].mean() * 12)
    return mu, cov, rf_ann, r


def solve_weights(mu, cov, rf_ann, objective="sharpe"):
    """Module A3 Step-4 optimizer: SLSQP, long-only, fully invested, same starts/tolerances."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    n = len(mu)
    if "A3_OPTIMIZER_STATUS_LOG" not in globals():
        globals()["A3_OPTIMIZER_STATUS_LOG"] = []

    if objective == "sharpe":
        def loss(w):
            variance = float(w @ cov @ w)
            if variance <= 1e-14:
                return 1e12
            return -float((w @ mu - float(rf_ann)) / np.sqrt(variance))
    elif objective in ("minvar", "min variance"):
        def loss(w):
            return float(w @ cov @ w)
    else:
        raise ValueError("objective must be 'sharpe' or 'minvar'")

    # This is the optimizer used in Module_A3_Optimizer Step 4.
    res = minimize(
        loss,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = np.asarray(res.x, dtype=float)
    # A3's original helper returns res.x after SLSQP. In a few windows SLSQP can report
    # "positive directional derivative" while the returned vector is finite and satisfies
    # all portfolio constraints. To match A3's implementation, accept such feasible outputs
    # but log the solver status for audit; reject only invalid / non-finite solutions.
    valid = (np.isfinite(res.fun) and np.all(np.isfinite(w)) and
             np.all(w >= -1e-8) and np.all(w <= 1.0 + 1e-8) and
             np.isclose(w.sum(), 1.0, atol=1e-7))
    if not valid:
        raise RuntimeError(f"A3 SLSQP returned invalid weights ({objective}): {res.message}; sum={w.sum()}, min={w.min()}, max={w.max()}")
    if not res.success:
        A3_OPTIMIZER_STATUS_LOG.append({"Objective": objective, "Status": str(res.message),
                                        "Feasible weights accepted": True,
                                        "Weight sum": float(w.sum()), "Objective value": float(res.fun)})
    return w


def a3_fixed_em_weights(mu, cov, rf_ann, fixed_asset_idx, fixed_weight=0.10, objective="minvar", grid_points=2001):
    """Module A3 Steps 2–3: fix EM share and grid-search the remaining S&P 500 / US Agg split."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    n = len(mu)
    if n != 3:
        raise ValueError("A3 fixed-EM grid expects exactly S&P 500, US Agg and one EM exposure")
    j = int(fixed_asset_idx)
    if j not in (0, 1, 2):
        raise ValueError("fixed_asset_idx must be 0, 1 or 2")
    x = float(fixed_weight)
    if not 0 <= x <= 1:
        raise ValueError("fixed_weight must be between 0 and 1")
    other = [k for k in range(n) if k != j]
    split = np.linspace(0.0, 1.0, int(grid_points))
    W = np.zeros((len(split), n), dtype=float)
    W[:, j] = x
    W[:, other[0]] = split * (1.0 - x)
    W[:, other[1]] = (1.0 - split) * (1.0 - x)
    means = W @ mu
    vars_ = np.maximum(np.einsum("ij,jk,ik->i", W, cov, W), 0.0)
    if objective == "sharpe":
        scores = np.divide(means - float(rf_ann), np.sqrt(vars_), out=np.full_like(means, -np.inf), where=vars_ > 1e-14)
        idx = int(np.nanargmax(scores))
    elif objective in ("minvar", "min variance"):
        idx = int(np.nanargmin(vars_))
    else:
        raise ValueError("objective must be 'sharpe' or 'minvar'")
    w = W[idx]
    if not np.isclose(w.sum(), 1.0, atol=1e-12) or not np.isclose(w[j], x, atol=1e-12):
        raise AssertionError("A3 fixed-EM grid violated the fixed-weight or full-investment constraint")
    return w

def portfolio_metrics(mu, cov, rf_ann, w):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf_ann) / vol if vol > 0 else np.nan
    return {"Annual Return %": ret * 100, "Annual Volatility %": vol * 100, "Sharpe": sharpe}


def efficient_frontier(returns, rf_monthly, n_points=50):
    """A3-style long-only efficient frontier, solving each target-return point with SLSQP."""
    mu, cov, rf_ann, r_used = annualized_inputs(returns, rf_monthly)
    n = len(mu)
    w_min = solve_weights(mu, cov, rf_ann, "minvar")
    r_min = float(w_min @ mu)
    targets = np.linspace(r_min, float(np.max(mu)), n_points)
    cov = (cov + cov.T) / 2
    pts = []
    for target in targets:
        def loss(w):
            return float(w @ cov @ w)
        cons = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            {"type": "eq", "fun": lambda w, target=target: float(w @ mu - target)},
        ]
        res = minimize(
            loss, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
            constraints=cons, options={"ftol": 1e-12, "maxiter": 1000}
        )
        if res.success and np.isfinite(res.fun):
            vol = np.sqrt(max(float(res.x @ cov @ res.x), 0.0)) * 100
            pts.append((vol, float(target) * 100))
        elif np.isfinite(res.fun):
            w = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(w)) and np.all(w >= -1e-8) and np.isclose(w.sum(), 1.0, atol=1e-7) and np.isclose(w @ mu, target, atol=1e-6):
                vol = np.sqrt(max(float(w @ cov @ w), 0.0)) * 100
                pts.append((vol, float(target) * 100))
    return pd.DataFrame(pts, columns=["Volatility %", "Return %"]), r_used

def window_summary(returns, rf_monthly):
    mu, cov, rf_ann, used = annualized_inputs(returns, rf_monthly)
    w_sh = solve_weights(mu, cov, rf_ann, "sharpe")
    w_mv = solve_weights(mu, cov, rf_ann, "minvar")
    out = {"N months": len(used)}
    out.update({f"Max-Sharpe {k}": v for k, v in portfolio_metrics(mu, cov, rf_ann, w_sh).items()})
    out.update({f"Min-Variance {k}": v for k, v in portfolio_metrics(mu, cov, rf_ann, w_mv).items()})
    return out, pd.Series(w_sh, index=returns.columns), pd.Series(w_mv, index=returns.columns), used


def _garch_unpack(theta, ar_lags, vol_model, dist):
    """Decode the optimizer vector into interpretable model parameters."""
    k = 0
    mu = float(theta[k]); k += 1
    phi = np.asarray(theta[k:k + ar_lags], dtype=float)
    k += ar_lags

    if vol_model in ("GARCH", "GJR-GARCH"):
        omega = float(np.exp(theta[k])); k += 1
        alpha = float(theta[k]); k += 1
        beta = float(theta[k]); k += 1
        gamma = float(theta[k]) if vol_model == "GJR-GARCH" else 0.0
        if vol_model == "GJR-GARCH":
            k += 1
    elif vol_model == "EGARCH":
        omega = float(theta[k]); k += 1
        alpha = float(theta[k]); k += 1
        gamma = float(theta[k]); k += 1
        beta = float(theta[k]); k += 1
    else:
        raise ValueError(f"Unsupported volatility model: {vol_model}")

    nu = float(2.0 + np.exp(theta[k])) if dist == "t" else np.inf
    return mu, phi, omega, alpha, beta, gamma, nu


def _garch_parameter_layout(y, spec):
    """Create starting values and box bounds for L-BFGS-B."""
    ar_lags = int(spec["ar_lags"])
    vol_model = spec["vol"]
    dist = spec["dist"]
    variance = max(float(np.var(y, ddof=1)), 1e-8)

    mean0 = [float(np.mean(y))] + [0.0] * ar_lags
    mean_bounds = [(float(np.mean(y) - 5*np.std(y) - 1.0),
                    float(np.mean(y) + 5*np.std(y) + 1.0))]
    mean_bounds += [(-0.98, 0.98)] * ar_lags

    if vol_model == "GARCH":
        vol0 = [np.log(max(variance * 0.05, 1e-10)), 0.08, 0.88]
        vol_bounds = [(np.log(max(variance * 1e-8, 1e-12)),
                       np.log(max(variance * 10.0, 1e-8))),
                      (1e-6, 0.60), (1e-6, 0.998)]
    elif vol_model == "GJR-GARCH":
        vol0 = [np.log(max(variance * 0.05, 1e-10)), 0.05, 0.84, 0.08]
        vol_bounds = [(np.log(max(variance * 1e-8, 1e-12)),
                       np.log(max(variance * 10.0, 1e-8))),
                      (1e-6, 0.60), (1e-6, 0.998), (0.0, 1.50)]
    elif vol_model == "EGARCH":
        vol0 = [float(np.log(variance) * 0.10), 0.10, -0.05, 0.90]
        vol_bounds = [(-10.0, 10.0), (1e-6, 3.0), (-3.0, 3.0), (-0.995, 0.995)]
    else:
        raise ValueError(f"Unsupported volatility model: {vol_model}")

    dist0, dist_bounds = ([np.log(6.0)] , [(np.log(0.05), np.log(198.0))]) if dist == "t" else ([], [])
    return np.asarray(mean0 + vol0 + dist0, dtype=float), mean_bounds + vol_bounds + dist_bounds


# Numba acceleration is optional; the exact same likelihood remains available as Python fallback.
try:
    from numba import njit
    NUMBA_GARCH_AVAILABLE = True
except Exception:
    NUMBA_GARCH_AVAILABLE = False


def _garch_nll_core(theta, y, ar_lags, vol_code, dist_code, sample_var):
    """Fast conditional Gaussian / standardised Student-t negative log likelihood.
    vol_code: 0=GARCH, 1=GJR-GARCH, 2=EGARCH; dist_code: 0=normal, 1=Student-t.
    """
    n = len(y)
    mu = theta[0]
    phi = np.empty(ar_lags, dtype=np.float64)
    for j in range(ar_lags):
        phi[j] = theta[1+j]

    k = 1 + ar_lags
    if vol_code == 2:
        omega = theta[k]; alpha = theta[k+1]; gamma = theta[k+2]; beta = theta[k+3]
        k += 4
    else:
        omega = np.exp(theta[k]); alpha = theta[k+1]; beta = theta[k+2]
        gamma = theta[k+3] if vol_code == 1 else 0.0
        k += 4 if vol_code == 1 else 3

    nu = 2.0 + np.exp(theta[k]) if dist_code == 1 else 1.0e9
    if vol_code in (0, 1):
        persistence = alpha + beta + (gamma/2.0 if vol_code == 1 else 0.0)
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or persistence >= 0.999:
            return 1.0e12 + 1.0e9 * max(persistence - 0.999, 0.0)**2
    else:
        if abs(beta) >= 0.999:
            return 1.0e12

    eps = np.empty(n, dtype=np.float64)
    for t in range(n):
        e = y[t] - mu
        for j in range(1, ar_lags+1):
            if t >= j:
                e -= phi[j-1] * y[t-j]
        eps[t] = e

    h = np.empty(n, dtype=np.float64)
    h[0] = sample_var
    if vol_code in (0, 1):
        for t in range(1, n):
            eprev = eps[t-1]
            asym = gamma * eprev * eprev if (vol_code == 1 and eprev < 0.0) else 0.0
            ht = omega + alpha*eprev*eprev + beta*h[t-1] + asym
            if not np.isfinite(ht) or ht <= 1.0e-12:
                return 1.0e12
            h[t] = ht
    else:
        log_h = np.empty(n, dtype=np.float64)
        log_h[0] = np.log(sample_var)
        if dist_code == 1:
            expected_abs = (np.sqrt(nu-2.0) *
                            np.exp(math.lgamma((nu-1.0)/2.0) - math.lgamma(nu/2.0)) /
                            np.sqrt(np.pi))
        else:
            expected_abs = np.sqrt(2.0/np.pi)
        zprev = eps[0] / np.sqrt(sample_var)
        for t in range(1, n):
            log_h[t] = omega + beta*log_h[t-1] + alpha*(abs(zprev)-expected_abs) + gamma*zprev
            if log_h[t] > 30.0:
                log_h[t] = 30.0
            elif log_h[t] < -30.0:
                log_h[t] = -30.0
            h[t] = np.exp(log_h[t])
            zprev = eps[t] / np.sqrt(h[t])

    total = 0.0
    if dist_code == 0:
        c0 = 0.5*np.log(2.0*np.pi)
        for t in range(ar_lags, n):
            ht = h[t]
            if ht <= 0.0 or not np.isfinite(ht):
                return 1.0e12
            total += c0 + 0.5*np.log(ht) + 0.5*eps[t]*eps[t]/ht
    else:
        if nu <= 2.0:
            return 1.0e12
        c = (math.lgamma((nu+1.0)/2.0) - math.lgamma(nu/2.0)
             - 0.5*np.log(np.pi*(nu-2.0)))
        for t in range(ar_lags, n):
            ht = h[t]
            if ht <= 0.0 or not np.isfinite(ht):
                return 1.0e12
            total += -c + 0.5*np.log(ht) + ((nu+1.0)/2.0)*np.log1p(
                eps[t]*eps[t]/(ht*(nu-2.0)))
    return total if np.isfinite(total) else 1.0e12


if NUMBA_GARCH_AVAILABLE:
    _garch_nll_fast = njit(cache=False)(_garch_nll_core)
else:
    _garch_nll_fast = _garch_nll_core


def fit_garch_candidate(series, spec, scale=100.0):
    """Fit one AR-GARCH candidate using bounded L-BFGS-B (no SLSQP clipping)."""
    x = pd.Series(series).dropna().astype(float) * scale
    y = x.to_numpy(dtype=float)
    n = len(y)
    ar_lags = int(spec["ar_lags"])
    vol_model = spec["vol"]
    dist = spec["dist"]
    if n < max(80, ar_lags + 30):
        raise ValueError(f"Not enough observations: {n}")
    if not np.isfinite(y).all():
        raise ValueError("Input contains non-finite values after cleaning.")

    x0, bounds = _garch_parameter_layout(y, spec)
    sample_var = max(float(np.var(y, ddof=1)), 1e-8)

    def residuals_from_mean(mu, phi):
        eps = y - mu
        for lag in range(1, ar_lags + 1):
            eps[lag:] -= phi[lag - 1] * y[:-lag]
        return eps

    vol_code = {"GARCH": 0, "GJR-GARCH": 1, "EGARCH": 2}[vol_model]
    dist_code = 0 if dist == "normal" else 1

    def objective(theta):
        return float(_garch_nll_fast(
            np.asarray(theta, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            ar_lags, vol_code, dist_code, sample_var
        ))

    # Fit the standard feasible start first. Try a second feasible start only if needed.
    # This reduces repeated likelihood evaluations without changing the model or constraints.
    start2 = x0.copy()
    vol_start = 1 + ar_lags
    if vol_model == "GARCH":
        start2[vol_start + 1] = 0.12
        start2[vol_start + 2] = 0.78
    elif vol_model == "GJR-GARCH":
        start2[vol_start + 1] = 0.10
        start2[vol_start + 2] = 0.75
        start2[vol_start + 3] = 0.10
    elif vol_model == "EGARCH":
        start2[vol_start + 1] = 0.18
        start2[vol_start + 2] = -0.10
        start2[vol_start + 3] = 0.82

    options = {"maxiter": 1800, "ftol": 1e-9, "maxls": 35}
    fit = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=options)
    if not (fit.success and np.isfinite(fit.fun)):
        retry = minimize(objective, start2, method="L-BFGS-B", bounds=bounds, options=options)
        if retry.success and np.isfinite(retry.fun):
            fit = retry
        else:
            raise RuntimeError(
                "Neither feasible GARCH starting point converged successfully. "
                f"Start 1: {fit.message}; Start 2: {retry.message}"
            )

    mu, phi, omega, alpha, beta, gamma, nu = _garch_unpack(
        fit.x, ar_lags, vol_model, dist
    )
    if ar_lags == 1 and abs(phi[0]) >= 1.0:
        raise RuntimeError("Estimated AR(1) mean equation is non-stationary.")
    if ar_lags == 2:
        ar_roots = np.roots([-phi[1], -phi[0], 1.0])
        if not np.all(np.abs(ar_roots) > 1.0):
            raise RuntimeError("Estimated AR(2) mean equation is non-stationary.")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise RuntimeError("Estimated GARCH parameters violate positivity constraints.")
    if vol_model in ("GARCH", "GJR-GARCH"):
        persistence_check = alpha + beta + (gamma / 2.0 if vol_model == "GJR-GARCH" else 0.0)
        if persistence_check >= 0.999:
            raise RuntimeError(f"Estimated volatility persistence is not below 0.999: {persistence_check:.6f}")
    eps = residuals_from_mean(mu, phi)
    h = np.empty(n, dtype=float)
    h[0] = sample_var
    if vol_model in ("GARCH", "GJR-GARCH"):
        for t in range(1, n):
            h[t] = (omega + alpha*eps[t-1]**2 + beta*h[t-1]
                    + (gamma*eps[t-1]**2 if vol_model == "GJR-GARCH" and eps[t-1] < 0 else 0.0))
            h[t] = max(float(h[t]), 1e-12)
        persistence = alpha + beta + (gamma/2.0 if vol_model == "GJR-GARCH" else 0.0)
    else:
        log_h = np.empty(n, dtype=float)
        log_h[0] = np.log(sample_var)
        if dist == "t":
            expected_abs_z = (np.sqrt(nu - 2.0) *
                              np.exp(gammaln((nu - 1.0) / 2.0) -
                                     gammaln(nu / 2.0)) / np.sqrt(np.pi))
        else:
            expected_abs_z = np.sqrt(2.0/np.pi)
        z_prev = eps[0]/np.sqrt(sample_var)
        for t in range(1, n):
            log_h[t] = (omega + beta*log_h[t-1]
                        + alpha*(abs(z_prev)-expected_abs_z) + gamma*z_prev)
            log_h[t] = np.clip(log_h[t], -30.0, 30.0)
            h[t] = max(float(np.exp(log_h[t])), 1e-12)
            z_prev = eps[t]/np.sqrt(h[t])
        persistence = beta

    z = pd.Series(eps[ar_lags:]/np.sqrt(h[ar_lags:]),
                  index=x.index[ar_lags:], name=x.name).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) <= 20:
        raise RuntimeError("Too few standardized residuals for diagnostics.")

    # The standardised-residual Ljung-Box test adjusts for estimated AR mean terms.
    lb_resid = float(acorr_ljungbox(
        z, lags=[10], model_df=ar_lags, return_df=True
    )["lb_pvalue"].iloc[0])
    # Squared-residual Ljung-Box is a volatility-clustering diagnostic.
    lb_squared = float(acorr_ljungbox(
        z**2, lags=[10], model_df=0, return_df=True
    )["lb_pvalue"].iloc[0])
    # Adjust ARCH LM degrees of freedom for the estimated mean parameters.
    arch_lm = float(het_arch(z, nlags=10, ddof=1 + ar_lags)[1])

    k_params = len(fit.x)
    n_eff = len(z)
    nll = float(fit.fun)
    aic = 2*k_params + 2*nll
    bic = np.log(n_eff)*k_params + 2*nll
    mean_label = "Constant" if ar_lags == 0 else f"AR({ar_lags})"
    model_label = f"{mean_label}-{vol_model}-{dist}"

    diagnostics = {
        "Model": model_label,
        "Mean Model": mean_label,
        "Volatility Model": vol_model,
        "Distribution": dist,
        "Mean": mu/scale,
        "AR Coefficients": ", ".join(f"{v:.5f}" for v in phi) if ar_lags else "None",
        "Omega": omega,
        "Alpha": alpha,
        "Gamma": gamma,
        "Beta": beta,
        "Alpha + Beta": alpha + beta if vol_model != "EGARCH" else np.nan,
        "Volatility Persistence": persistence,
        "Student-t df": nu if dist == "t" else np.nan,
        "N": n_eff,
        "NLL": nll,
        "AIC": aic,
        "BIC": bic,
        "Convergence": bool(fit.success),
        "Optimizer Message": str(fit.message),
        "Ljung-Box mean df adjustment": ar_lags,
        "ARCH LM mean df adjustment": 1 + ar_lags,
        "LB standardized residual p(10)": lb_resid,
        "LB squared residual p(10)": lb_squared,
        "ARCH LM p(10)": arch_lm,
    }
    return z, diagnostics, fit



def _archq_nll_core(theta, y, ar_lags, q, sample_var):
    """Vectorized Student-t AR-ARCH(q) negative log likelihood (no Numba dependency)."""
    n = len(y)
    mu = float(theta[0])
    phi = np.asarray(theta[1:1+ar_lags], dtype=float)
    k = 1 + ar_lags
    omega = float(np.exp(theta[k])); k += 1
    alpha = np.asarray(theta[k:k+q], dtype=float); k += q
    nu = float(2.0 + np.exp(theta[k]))
    alpha_sum = float(alpha.sum())
    if omega <= 0.0 or np.any(alpha < 0.0) or alpha_sum >= 0.998 or nu <= 2.0:
        return 1.0e12 + 1.0e9 * max(alpha_sum-0.998, 0.0)**2

    eps = y - mu
    for lag in range(1, ar_lags+1):
        eps[lag:] -= phi[lag-1] * y[:-lag]

    burn = max(q, ar_lags)
    h = np.full(n-burn, omega, dtype=float)
    for lag in range(1, q+1):
        h += alpha[lag-1] * eps[burn-lag:n-lag]**2
    if not np.isfinite(h).all() or np.any(h <= 1.0e-12):
        return 1.0e12

    from scipy.special import gammaln
    c = gammaln((nu+1.0)/2.0) - gammaln(nu/2.0) - 0.5*np.log(np.pi*(nu-2.0))
    e = eps[burn:]
    terms = -c + 0.5*np.log(h) + ((nu+1.0)/2.0)*np.log1p(e*e/(h*(nu-2.0)))
    total = float(np.sum(terms))
    return total if np.isfinite(total) else 1.0e12


def fit_archq_candidate(series, spec, scale=100.0):
    """Fit an AR(1)-ARCH(q)-Student-t model; fully vectorized and no Numba required."""
    from scipy.optimize import minimize
    x = pd.Series(series).dropna().astype(float)*scale
    y = x.to_numpy(dtype=float)
    ar_lags, q = int(spec.get("ar_lags", 1)), int(spec["q"])
    if len(y) < max(100, q+ar_lags+30):
        raise ValueError(f"Not enough observations for AR-ARCH({q}): {len(y)}")
    variance = max(float(np.var(y, ddof=1)), 1e-8)
    alpha0 = np.full(q, 0.20/q)
    mean0 = [float(np.mean(y))] + [0.0]*ar_lags
    x0 = np.asarray(mean0 + [np.log(variance*(1-alpha0.sum()))] + alpha0.tolist() + [np.log(6.0)])
    x1 = x0.copy()
    x1[1+ar_lags+1:1+ar_lags+1+q] = np.full(q, 0.35/q)
    bounds = [(float(np.mean(y)-5*np.std(y)-1), float(np.mean(y)+5*np.std(y)+1))]
    bounds += [(-0.98,0.98)]*ar_lags
    bounds += [(np.log(max(variance*1e-8,1e-12)), np.log(max(variance*10,1e-8)))]
    bounds += [(1e-6, 0.90/q)]*q
    bounds += [(np.log(0.05), np.log(198.0))]
    sample_var = variance
    objective = lambda th: _archq_nll_core(np.asarray(th,dtype=float), y, ar_lags, q, sample_var)
    options={"maxiter":2000,"ftol":1e-9,"maxls":40}
    fit=minimize(objective,x0,method="L-BFGS-B",bounds=bounds,options=options)
    if not (fit.success and np.isfinite(fit.fun)):
        retry=minimize(objective,x1,method="L-BFGS-B",bounds=bounds,options=options)
        if retry.success and np.isfinite(retry.fun):
            fit=retry
        else:
            raise RuntimeError(f"Neither AR-ARCH({q}) starting point converged: {fit.message}; {retry.message}")
    mu=float(fit.x[0]); phi=np.asarray(fit.x[1:1+ar_lags],dtype=float)
    omega=float(np.exp(fit.x[1+ar_lags]))
    alpha=np.asarray(fit.x[2+ar_lags:2+ar_lags+q],dtype=float)
    nu=float(2+np.exp(fit.x[-1]))
    eps=y-mu
    for lag in range(1,ar_lags+1):
        eps[lag:]-=phi[lag-1]*y[:-lag]
    burn=max(q,ar_lags)
    h=np.full(len(y)-burn,omega,dtype=float)
    for lag in range(1,q+1):
        h += alpha[lag-1]*eps[burn-lag:len(y)-lag]**2
    z=pd.Series(eps[burn:]/np.sqrt(h),index=x.index[burn:],name=x.name).replace([np.inf,-np.inf],np.nan).dropna()
    lb=float(acorr_ljungbox(z,lags=[10],model_df=ar_lags,return_df=True)["lb_pvalue"].iloc[0])
    lb2=float(acorr_ljungbox(z**2,lags=[10],model_df=0,return_df=True)["lb_pvalue"].iloc[0])
    archp=float(het_arch(z,nlags=10,ddof=1+ar_lags)[1])
    k_params=len(fit.x); n_eff=len(z); nll=float(fit.fun)
    diag={
        "Model":f"AR({ar_lags})-ARCH({q})-t","Mean Model":f"AR({ar_lags})",
        "Volatility Model":f"ARCH({q})","Distribution":"t","Mean":mu/scale,
        "AR Coefficients":", ".join(f"{v:.5f}" for v in phi),
        "Omega":omega,"Alpha":float(alpha.sum()),"Alpha Coefficients":", ".join(f"{v:.5f}" for v in alpha),
        "Gamma":np.nan,"Beta":np.nan,"Alpha + Beta":float(alpha.sum()),
        "Volatility Persistence":float(alpha.sum()),"Student-t df":nu,"N":n_eff,
        "NLL":nll,"AIC":2*k_params+2*nll,"BIC":np.log(n_eff)*k_params+2*nll,
        "Convergence":bool(fit.success),"Optimizer Message":str(fit.message),
        "Ljung-Box mean df adjustment":ar_lags,"ARCH LM mean df adjustment":1+ar_lags,
        "LB standardized residual p(10)":lb,"LB squared residual p(10)":lb2,"ARCH LM p(10)":archp,
    }
    return z,diag,fit

def _candidate_specs_for_asset(asset, frequency):
    """Define reasonable candidates; model choice is based on diagnostics then BIC."""
    def s(name, ar, vol="GARCH", dist="t"):
        return {"name": name, "ar_lags": ar, "vol": vol, "dist": dist}

    if asset == "USA":
        return [s("Constant_GARCH_Normal", 0, "GARCH", "normal"),
                s("Constant_GARCH_t", 0, "GARCH", "t")]
    if asset == "USA_IT":
        # USA IT had residual ARCH effects under the first AR(1)-GARCH-t attempt.
        # Expand mean and asymmetric-volatility candidates; selection remains diagnostic-first.
        candidates = [
            s("AR1_GARCH_t", 1, "GARCH", "t"),
            s("AR2_GARCH_t", 2, "GARCH", "t"),
            s("AR1_GJR_GARCH_t", 1, "GJR-GARCH", "t"),
            s("AR2_GJR_GARCH_t", 2, "GJR-GARCH", "t"),
            s("AR1_EGARCH_t", 1, "EGARCH", "t"),
            s("AR2_EGARCH_t", 2, "EGARCH", "t"),
        ]
        if frequency == "Weekly":
            # Vectorized ARCH(q) candidates are available without Numba.
            candidates += [
                {"name":"AR1_ARCH5_t", "engine":"ARCHQ", "ar_lags":1, "q":5, "dist":"t"},
                {"name":"AR1_ARCH10_t", "engine":"ARCHQ", "ar_lags":1, "q":10, "dist":"t"},
            ]
        return candidates
    if asset == "EM Small Cap Price":
        return [s("Constant_GARCH_Normal", 0, "GARCH", "normal"),
                s("AR1_GARCH_t", 1, "GARCH", "t"),
                s("AR2_GARCH_t", 2, "GARCH", "t"),
                s("AR1_GJR_GARCH_t", 1, "GJR-GARCH", "t")]
    # EM, EM_exIT and EM_exTWKR
    return [s("Constant_GARCH_Normal", 0, "GARCH", "normal"),
            s("AR1_GARCH_t", 1, "GARCH", "t"),
            s("AR2_GARCH_t", 2, "GARCH", "t")]


def fit_garch_panel(log_returns, panel_name):
    """Fit candidate models asset by asset and choose the best supported specification."""
    std_resid = pd.DataFrame(index=log_returns.index)
    selected_rows = []
    candidate_rows = []
    selected_fits = {}
    selected_names = {}

    for asset in log_returns.columns:
        series = log_returns[asset].dropna()
        candidates = _candidate_specs_for_asset(asset, panel_name)
        fitted_candidates = []
        print(f"\n{panel_name}: evaluating {asset}")

        for spec in candidates:
            try:
                if spec.get("engine") == "ARCHQ":
                    z, diag, fit = fit_archq_candidate(series, spec)
                else:
                    z, diag, fit = fit_garch_candidate(series, spec)
                row = {"Asset": asset, "Candidate": spec["name"], **diag}
                pcols = ["LB standardized residual p(10)",
                         "LB squared residual p(10)", "ARCH LM p(10)"]
                row["Diagnostic Pass Count"] = int(sum(
                    np.isfinite(row[c]) and row[c] >= 0.05 for c in pcols
                ))
                row["All Diagnostics Pass"] = row["Diagnostic Pass Count"] == 3
                candidate_rows.append(row)
                fitted_candidates.append((spec, z, diag, fit, row))
                print(f"  {spec['name']}: converged={diag['Convergence']}, "
                      f"LB={diag[pcols[0]]:.4f}, LB2={diag[pcols[1]]:.4f}, "
                      f"ARCH={diag[pcols[2]]:.4f}, BIC={diag['BIC']:.2f}")
            except Exception as exc:
                candidate_rows.append({"Asset": asset, "Candidate": spec["name"],
                                       "Convergence": False, "Error": str(exc),
                                       "Diagnostic Pass Count": 0,
                                       "All Diagnostics Pass": False})
                print(f"  {spec['name']}: FAILED - {exc}")

        valid = [item for item in fitted_candidates if item[2]["Convergence"]]
        if not valid:
            raise RuntimeError(f"No converged GARCH candidate for {panel_name} {asset}.")

        passing = [item for item in valid if item[4]["All Diagnostics Pass"]]
        if passing:
            chosen = min(passing, key=lambda item: item[2]["BIC"])
            selection_note = "All three diagnostics pass; lowest BIC"
        else:
            # If none passes all tests, prefer the model passing the most tests,
            # then use BIC as a tie-breaker. The remaining failure is reported.
            chosen = sorted(valid,
                            key=lambda item: (-item[4]["Diagnostic Pass Count"],
                                              item[2]["BIC"]))[0]
            selection_note = "No candidate passes all tests; most passes, then lowest BIC"

        spec, z, diag, fit, chosen_row = chosen
        chosen_row["Selected"] = True
        for row in candidate_rows:
            if row.get("Asset") == asset and row.get("Candidate") != spec["name"]:
                row.setdefault("Selected", False)
        std_resid.loc[z.index, asset] = z
        selected_names[asset] = spec["name"]
        selected_fits[asset] = fit
        selected_rows.append({"Asset": asset, "Selected Candidate": spec["name"],
                              "Selection Rule": selection_note,
                              "Diagnostics Fully Pass": bool(chosen_row["All Diagnostics Pass"]), **diag})
        print(f"  SELECTED: {spec['name']} ({selection_note})")

    selected_table = pd.DataFrame(selected_rows).set_index("Asset")
    candidate_table = pd.DataFrame(candidate_rows)
    return std_resid, selected_table, candidate_table, selected_fits, selected_names


# ------------------------------------------------------------
# 13A-2. DCC(1,1) likelihood and correlation recursion
#       Numba is optional; a NumPy fallback is provided.
# ------------------------------------------------------------

try:
    from numba import njit
    NUMBA_DCC_AVAILABLE = True

    @njit
    def _dcc_nll_bivariate_numba(a, b, z, qbar):
        q11, q22, q12 = qbar[0, 0], qbar[1, 1], qbar[0, 1]
        persistence = 1.0 - a - b
        total = 0.0
        for t in range(z.shape[0]):
            if t > 0:
                z1p, z2p = z[t-1, 0], z[t-1, 1]
                q11 = persistence*qbar[0,0] + a*z1p*z1p + b*q11
                q22 = persistence*qbar[1,1] + a*z2p*z2p + b*q22
                q12 = persistence*qbar[0,1] + a*z1p*z2p + b*q12
            denom = np.sqrt(max(q11*q22, 1e-14))
            rho = q12 / denom
            if rho > 0.999999 or rho < -0.999999:
                return 1e20
            det = max(1.0-rho*rho, 1e-12)
            z1, z2 = z[t,0], z[t,1]
            quad = (z1*z1 + z2*z2 - 2.0*rho*z1*z2) / det
            total += 0.5*(np.log(det) + quad - z1*z1 - z2*z2)
        return total

    @njit
    def _dcc_recursion_bivariate_numba(a, b, z, qbar):
        out = np.empty((z.shape[0], 2, 2), dtype=np.float64)
        q11, q22, q12 = qbar[0,0], qbar[1,1], qbar[0,1]
        persistence = 1.0 - a - b
        for t in range(z.shape[0]):
            if t > 0:
                z1p, z2p = z[t-1,0], z[t-1,1]
                q11 = persistence*qbar[0,0] + a*z1p*z1p + b*q11
                q22 = persistence*qbar[1,1] + a*z2p*z2p + b*q22
                q12 = persistence*qbar[0,1] + a*z1p*z2p + b*q12
            rho = q12 / np.sqrt(max(q11*q22, 1e-14))
            out[t,0,0] = 1.0
            out[t,1,1] = 1.0
            out[t,0,1] = rho
            out[t,1,0] = rho
        return out
except Exception:
    NUMBA_DCC_AVAILABLE = False


def dcc_nll(params, z, qbar, max_persistence=0.9995):
    a, b = map(float, params)
    if a < 0 or b < 0 or a+b >= max_persistence:
        return 1e20
    z = np.asarray(z, dtype=float)
    qbar = np.asarray(qbar, dtype=float)
    if z.shape[1] != 2:
        raise ValueError("This implementation expects bivariate DCC pairs.")
    if NUMBA_DCC_AVAILABLE:
        return float(_dcc_nll_bivariate_numba(a, b, z, qbar))
    q11, q22, q12 = qbar[0,0], qbar[1,1], qbar[0,1]
    persistence = 1.0-a-b
    total = 0.0
    for t in range(len(z)):
        if t:
            z1p, z2p = z[t-1]
            q11 = persistence*qbar[0,0] + a*z1p*z1p + b*q11
            q22 = persistence*qbar[1,1] + a*z2p*z2p + b*q22
            q12 = persistence*qbar[0,1] + a*z1p*z2p + b*q12
        rho = q12 / np.sqrt(max(q11*q22, 1e-14))
        if abs(rho) >= 0.999999:
            return 1e20
        det = max(1.0-rho*rho, 1e-12)
        z1, z2 = z[t]
        quad = (z1*z1+z2*z2-2*rho*z1*z2)/det
        total += 0.5*(np.log(det)+quad-z1*z1-z2*z2)
    return float(total)


def dcc_recursion(a, b, z, qbar):
    z = np.asarray(z, dtype=float)
    qbar = np.asarray(qbar, dtype=float)
    if z.shape[1] != 2:
        raise ValueError("This implementation expects bivariate DCC pairs.")
    if NUMBA_DCC_AVAILABLE:
        return _dcc_recursion_bivariate_numba(float(a), float(b), z, qbar)
    out = np.empty((len(z), 2, 2), dtype=float)
    q11, q22, q12 = qbar[0,0], qbar[1,1], qbar[0,1]
    persistence = 1.0-float(a)-float(b)
    for t in range(len(z)):
        if t:
            z1p, z2p = z[t-1]
            q11 = persistence*qbar[0,0] + a*z1p*z1p + b*q11
            q22 = persistence*qbar[1,1] + a*z2p*z2p + b*q22
            q12 = persistence*qbar[0,1] + a*z1p*z2p + b*q12
        rho = q12 / np.sqrt(max(q11*q22, 1e-14))
        out[t] = [[1.0, rho], [rho, 1.0]]
    return out


def fit_dcc_pair(z_df, label, persistence_cap=0.998):
    """Estimate DCC parameters with a selected strict upper bound on a+b."""
    persistence_cap=float(persistence_cap)
    if not (0.95 <= persistence_cap < 0.9995):
        raise ValueError("persistence_cap must be in [0.95, 0.9995).")
    z_df = z_df.dropna()
    z = z_df.to_numpy(float)
    if z.shape[0] < 80:
        raise ValueError(f"DCC pair {label} has only {z.shape[0]} observations.")
    qbar = np.cov(z, rowvar=False)
    qbar = (qbar + qbar.T)/2 + np.eye(qbar.shape[0])*1e-8

    def unpack(theta):
        s1, s2 = expit(theta[0]), expit(theta[1])
        a = persistence_cap*s1
        b = (persistence_cap-a)*s2
        return float(a), float(b)

    def objective(theta):
        a, b = unpack(theta)
        return dcc_nll((a, b), z, qbar, max_persistence=max(0.9995, persistence_cap+1e-6))

    fits = []
    for a0, b0 in ((0.03, 0.94), (0.05, 0.88), (0.10, 0.75)):
        s1 = np.clip(a0/persistence_cap, 1e-7, 1-1e-7)
        s2 = np.clip(b0/max(persistence_cap-a0, 1e-8), 1e-7, 1-1e-7)
        theta0 = np.array([logit(s1), logit(s2)], dtype=float)
        candidate = minimize(objective, theta0, method="L-BFGS-B",
                             bounds=[(-8.0, 8.0), (-8.0, 8.0)],
                             options={"maxiter": 3000, "ftol": 1e-10, "maxls": 50})
        if np.isfinite(candidate.fun):
            fits.append(candidate)

    converged_fits = [candidate for candidate in fits
                      if candidate.success and np.isfinite(candidate.fun)]
    if not converged_fits:
        messages = "; ".join(str(candidate.message) for candidate in fits)
        raise RuntimeError(
            f"No DCC(1,1) starting point converged for {label}. "
            f"Optimizer messages: {messages or 'no finite solutions'}"
        )
    # All starts are evaluated; use the converged solution with the lowest NLL.
    fit = min(converged_fits, key=lambda candidate: candidate.fun)
    a, b = unpack(fit.x)
    correlations = dcc_recursion(a, b, z, qbar)
    return {"Pair": label, "Persistence Cap": persistence_cap, "a": a, "b": b, "a+b": a+b,
            "Near Cap (within 0.002)": bool(a+b >= persistence_cap-0.002),
            "NLL": float(fit.fun), "N": len(z),
            "Convergence": bool(fit.success),
            "Optimizer Message": str(fit.message)}, correlations, z_df.index




# ======================================================================================
# Run: data, A3 optimizer extensions, OOS backtests, DCC-GARCH, rolling correlations
# ======================================================================================

def make_a3_rolling_windows(first_month, last_month=END, years=5):
    """A3 convention: windows end each August and shift forward one year."""
    start_ts = pd.Timestamp(first_month)
    first_end = start_ts + pd.DateOffset(years=years)
    end_year = first_end.year if first_end.month <= 8 else first_end.year + 1
    first_end = pd.Timestamp(year=end_year, month=8, day=31)
    last_ts = pd.Timestamp(last_month)
    if first_end > last_ts:
        return {}
    ends = pd.date_range(first_end, last_ts, freq="12ME")
    out = {}
    for end in ends:
        start = end - pd.DateOffset(months=12 * years - 1)
        label = f"{start:%Y-%m}…{end:%Y-%m}"
        out[label] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return out


def complete_window(data, start, end, min_obs=None):
    d = data.loc[start:end].dropna()
    if min_obs is not None and len(d) < min_obs:
        return None
    return d


def fixed_em_grid(data, rf_series, exposure, start, end, min_obs=12):
    """EM weight is fixed at x; optimise the remaining allocation between SP500 and Agg by grid search."""
    cols = BASE_ASSETS + [exposure]
    d = complete_window(data[cols], start, end, min_obs=min_obs)
    if d is None: return pd.DataFrame()
    rf_w = rf_series.reindex(d.index)
    if rf_w.isna().any(): return pd.DataFrame()
    mu, cov, rf_ann, used = annualized_inputs(d, rf_w)
    s_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    rows = []
    for x in EM_STEPS:
        sp_weights = s_grid * (1 - x)
        bond_weights = (1 - s_grid) * (1 - x)
        W = np.column_stack([sp_weights, bond_weights, np.full_like(s_grid, x)])
        mean = W @ mu
        variance = np.einsum("ij,jk,ik->i", W, cov, W)
        vol = np.sqrt(np.maximum(variance, 0))
        sharpe = np.where(vol > 0, (mean - rf_ann) / vol, np.nan)
        i_sh = int(np.nanargmax(sharpe))
        i_mv = int(np.nanargmin(vol))
        rows.append({"Exposure": exposure, "EM %": int(round(x * 100)), "N months": len(used),
                     "Max-Sharpe": sharpe[i_sh], "Max-Sharpe Return %": mean[i_sh] * 100,
                     "Max-Sharpe Vol %": vol[i_sh] * 100, "S&P 500 weight %": W[i_sh, 0] * 100,
                     "US Agg weight %": W[i_sh, 1] * 100,
                     "Min-Variance Vol %": vol[i_mv] * 100,
                     "Min-Variance Return %": mean[i_mv] * 100,
                     "Min-Variance S&P 500 %": W[i_mv, 0] * 100,
                     "Min-Variance US Agg %": W[i_mv, 1] * 100})
    return pd.DataFrame(rows)


def rolling_a3_joint_country_sleeve(monthly_returns, rf_series, lookback=60, objective="sharpe"):
    """Each month, estimate the joint universe (S&P 500 + US Agg + 8 countries) over the previous 60 months
    with A3's SLSQP optimizer, normalise the country slice to 100% and apply it to the next month."""
    assets = BASE_ASSETS + COUNTRIES
    aligned = pd.concat([monthly_returns[assets], rf_series.rename("__rf__")], axis=1).dropna()
    rows_max, rows_min = {}, {}
    w_country_max, w_country_min, w_joint_max, w_joint_min = [], [], [], []
    for i in range(lookback, len(aligned)):
        hist = aligned.iloc[i - lookback:i]
        current = aligned.iloc[i]
        r_hist = hist[assets]
        mu, cov, rf_ann, _ = annualized_inputs(r_hist, hist["__rf__"])
        w_sh = solve_weights(mu, cov, rf_ann, "sharpe")
        w_mv = solve_weights(mu, cov, rf_ann, "minvar")
        dt = aligned.index[i]
        country_sh = w_sh[len(BASE_ASSETS):]
        country_mv = w_mv[len(BASE_ASSETS):]
        sh_sum = float(country_sh.sum())
        mv_sum = float(country_mv.sum())
        if sh_sum > 1e-10:
            sleeve_w = country_sh / sh_sum
            rows_max[dt] = float(current[COUNTRIES].to_numpy(dtype=float) @ sleeve_w)
            w_country_max.append(pd.Series(sleeve_w, index=COUNTRIES, name=dt))
        if mv_sum > 1e-10:
            sleeve_w_mv = country_mv / mv_sum
            rows_min[dt] = float(current[COUNTRIES].to_numpy(dtype=float) @ sleeve_w_mv)
            w_country_min.append(pd.Series(sleeve_w_mv, index=COUNTRIES, name=dt))
        w_joint_max.append(pd.Series(w_sh, index=assets, name=dt))
        w_joint_min.append(pd.Series(w_mv, index=assets, name=dt))
    ret_max = pd.Series(rows_max, name="Optimal EM Countries OOS (A3 Max Sharpe)").sort_index()
    ret_min = pd.Series(rows_min, name="Optimal EM Countries OOS (A3 Min Variance)").sort_index()
    cw_max = pd.DataFrame(w_country_max); cw_min = pd.DataFrame(w_country_min)
    jw_max = pd.DataFrame(w_joint_max); jw_min = pd.DataFrame(w_joint_min)
    for frame in (cw_max, cw_min, jw_max, jw_min):
        if len(frame): frame.index.name = "date"
    return ret_max, cw_max, ret_min, cw_min, jw_max, jw_min


def fast_small_portfolio_weights(mu, cov, rf_ann, mode="minvar", fixed_asset_idx=None, fixed_weight=None):
    """Compatibility wrapper that routes all weights through Module A3's optimizer."""
    if fixed_asset_idx is not None:
        return a3_fixed_em_weights(mu, cov, rf_ann, fixed_asset_idx=fixed_asset_idx,
                                   fixed_weight=fixed_weight, objective=mode, grid_points=GRID_POINTS)
    return solve_weights(mu, cov, rf_ann, objective=mode)


def rolling_mvo_oos(returns, rf_series, lookback=60, mode="max_sharpe", fixed_asset=None, fixed_weight=None):
    """No-look-ahead monthly OOS optimisation using the exact Module A3 optimizer."""
    cols = list(returns.columns)
    aligned = pd.concat([returns, rf_series.rename("__rf__")], axis=1).dropna()
    values = aligned[cols].to_numpy(dtype=float)
    rf_values = aligned["__rf__"].to_numpy(dtype=float)
    dates = aligned.index
    out_values = []; out_dates = []; out_weights = []
    if fixed_asset is not None and fixed_asset not in cols:
        raise ValueError(f"{fixed_asset} not in return columns")
    fixed_idx = cols.index(fixed_asset) if fixed_asset is not None else None
    for i in range(lookback, len(aligned)):
        hist = values[i - lookback:i]
        mu = hist.mean(axis=0) * 12.0
        cov = np.cov(hist, rowvar=False, ddof=1) * 12.0
        rf_ann = float(rf_values[i - lookback:i].mean() * 12.0)
        if fixed_idx is None:
            w = fast_small_portfolio_weights(mu, cov, rf_ann, mode=mode)
        else:
            w = fast_small_portfolio_weights(mu, cov, rf_ann, mode=mode, fixed_asset_idx=fixed_idx, fixed_weight=fixed_weight)
        out_values.append(float(values[i] @ w))
        out_dates.append(dates[i])
        out_weights.append(w)
    out_series = pd.Series(out_values, index=pd.DatetimeIndex(out_dates), name=mode, dtype=float)
    weight_frame = pd.DataFrame(out_weights, index=pd.DatetimeIndex(out_dates), columns=cols)
    weight_frame.index.name = "date"
    return out_series, weight_frame


def performance_table(returns, rf_series):
    rows = []
    for col in returns:
        r = returns[col].dropna(); rf = rf_series.reindex(r.index)
        if len(r) < 2 or rf.isna().any(): continue
        wealth = (1 + r).cumprod(); peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
        dd = wealth.to_numpy() / peak - 1
        rows.append({"Portfolio": col, "Start": r.index.min(), "End": r.index.max(), "OOS months": len(r),
                     "Annualized return %": ((1 + r).prod() ** (12 / len(r)) - 1) * 100,
                     "Annualized volatility %": r.std(ddof=1) * np.sqrt(12) * 100,
                     "Sharpe": ((r - rf).mean() / r.std(ddof=1)) * np.sqrt(12) if r.std(ddof=1) > 0 else np.nan,
                     "Max drawdown %": dd.min() * 100})
    return pd.DataFrame(rows).set_index("Portfolio")


def _savefig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight"); plt.close(fig)


def monthly_data():
    """B's monthly simple-return panel: U.S. benchmark, A's EM series and countries, rf, EM Small Cap net / price."""
    p = panel()
    a = p["ret"]
    monthly = pd.concat([p["us"], a[["EM", "EM_exIT", "EM_exTWKR", *COUNTRIES]], p["rf"].rename("rf"),
                         p["extra"]["EMSC_N"].rename("EM Small Cap Net"), p["extra"]["EMSC_P"].rename("EM Small Cap Price")],
                        axis=1).sort_index()
    return monthly.loc[START:END], p


def run(do_dcc=True):
    plt.rcParams.update({"figure.figsize": (11, 5), "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "font.size": 10})
    globals()["A3_OPTIMIZER_STATUS_LOG"] = []
    monthly, p = monthly_data()
    rf_monthly = monthly["rf"].copy()
    coverage = coverage_table(monthly)

    # ---------------------------------------------------------------- 4. blocks and rolling windows
    FIXED_BLOCKS = {"2001-2005": ("2001-01-31", "2005-12-31"), "2006-2010": ("2006-01-31", "2010-12-31"),
                    "2011-2015": ("2011-01-31", "2015-12-31"), "2016-2020": ("2016-01-31", "2020-12-31"),
                    "2021-2026": ("2021-01-31", END)}
    rolling_windows = {y: make_a3_rolling_windows("2001-01-31", END, y) for y in WINDOW_YEARS}

    # ---------------------------------------------------------------- 5. baseline
    baseline_rows = []
    for block, (start, end) in FIXED_BLOCKS.items():
        d = complete_window(monthly[BASE_ASSETS], start, end, min_obs=24)
        if d is None: continue
        stats, w_sh, w_mv, used = window_summary(d, rf_monthly.reindex(d.index))
        baseline_rows.append({"Window type": "Fixed 5-year block", "Window": block, **stats,
                              "Max-Sharpe S&P 500 %": w_sh["S&P 500"] * 100, "Max-Sharpe US Agg %": w_sh["US Agg"] * 100,
                              "Min-Variance S&P 500 %": w_mv["S&P 500"] * 100, "Min-Variance US Agg %": w_mv["US Agg"] * 100})
    for years, wins in rolling_windows.items():
        for label, (start, end) in wins.items():
            d = complete_window(monthly[BASE_ASSETS], start, end, min_obs=12 * years)
            if d is None: continue
            stats, w_sh, w_mv, used = window_summary(d, rf_monthly.reindex(d.index))
            baseline_rows.append({"Window type": f"Rolling {years}y", "Window": label, **stats,
                                  "Max-Sharpe S&P 500 %": w_sh["S&P 500"] * 100, "Max-Sharpe US Agg %": w_sh["US Agg"] * 100,
                                  "Min-Variance S&P 500 %": w_mv["S&P 500"] * 100, "Min-Variance US Agg %": w_mv["US Agg"] * 100})
    baseline_summary = pd.DataFrame(baseline_rows)

    # ---------------------------------------------------------------- 7. A3-style joint country optimiser
    COUNTRY_START = "2015-04-30"
    COUNTRY_UNIVERSE = BASE_ASSETS + COUNTRIES
    country_joint_data = monthly[COUNTRY_UNIVERSE].loc[COUNTRY_START:END]
    country_windows = make_a3_rolling_windows(COUNTRY_START, END, 5)
    country_windows["2015-04…2026-08 (full)"] = (COUNTRY_START, END)
    joint_country_rows, joint_country_weight_rows, country_sleeve_rows = [], [], []
    for label, (start, end) in country_windows.items():
        d = complete_window(country_joint_data, start, end, min_obs=60)
        if d is None: continue
        rf_w = rf_monthly.reindex(d.index)
        if rf_w.isna().any(): continue
        mu, cov, rf_ann, used = annualized_inputs(d, rf_w)
        w_sh = solve_weights(mu, cov, rf_ann, "sharpe")
        w_mv = solve_weights(mu, cov, rf_ann, "minvar")
        m_base = complete_window(monthly[BASE_ASSETS], start, end, min_obs=60)
        mu_b, cov_b, rf_b, _ = annualized_inputs(m_base, rf_monthly.reindex(m_base.index))
        wb = solve_weights(mu_b, cov_b, rf_b, "sharpe")
        wb_mv = solve_weights(mu_b, cov_b, rf_b, "minvar")
        base_sh = portfolio_metrics(mu_b, cov_b, rf_b, wb)["Sharpe"]
        all_sh = portfolio_metrics(mu, cov, rf_ann, w_sh)["Sharpe"]
        baseline_minvar_vol = portfolio_metrics(mu_b, cov_b, rf_b, wb_mv)["Annual Volatility %"]
        country_minvar_vol = portfolio_metrics(mu, cov, rf_ann, w_mv)["Annual Volatility %"]
        country_sh, country_mv = w_sh[2:], w_mv[2:]
        sh_sum, mv_sum = country_sh.sum(), country_mv.sum()
        joint_country_rows.append({"Window": label, "N months": len(used), "Baseline Sharpe": base_sh,
                                   "Sharpe with 8 countries": all_sh, "Sharpe gain": all_sh - base_sh,
                                   "EM total weight max-Sharpe %": sh_sum * 100, "EM total weight min-var %": mv_sum * 100,
                                   "Min-var vol baseline %": baseline_minvar_vol, "Min-var vol with countries %": country_minvar_vol,
                                   "Country sleeve available": sh_sum > 1e-8})
        for kind, w in [("Max Sharpe", w_sh), ("Min Variance", w_mv)]:
            for asset, weight in zip(COUNTRY_UNIVERSE, w):
                joint_country_weight_rows.append({"Window": label, "Portfolio": kind, "Asset": asset, "Weight %": weight * 100})
        # Normalize the country slice to 100%: an in-sample country-only sleeve implied by A3's joint max-Sharpe solution.
        if sh_sum > 1e-8:
            basket_w = country_sh / sh_sum
            basket_ret = d[COUNTRIES].mul(basket_w, axis=1).sum(axis=1)
            for date, value in basket_ret.items():
                country_sleeve_rows.append({"Window": label, "date": date, "A3-implied EM country sleeve": value})
    joint_country_summary = pd.DataFrame(joint_country_rows).set_index("Window") if joint_country_rows else pd.DataFrame()
    joint_country_weights = pd.DataFrame(joint_country_weight_rows)
    a3_country_sleeves = pd.DataFrame(country_sleeve_rows)

    (country_oos_maxsharpe, country_oos_maxsharpe_weights, country_oos_minvar, country_oos_minvar_weights,
     country_oos_joint_maxsharpe_weights, country_oos_joint_minvar_weights) = rolling_a3_joint_country_sleeve(monthly, rf_monthly, LOOKBACK_MONTHS)
    monthly["Optimal EM Countries OOS (Max Sharpe)"] = country_oos_maxsharpe.reindex(monthly.index)
    monthly["Optimal EM Countries OOS (Min Variance)"] = country_oos_minvar.reindex(monthly.index)

    full_country_label = "2015-04…2026-08 (full)"
    full_sleeve_rows = a3_country_sleeves.loc[a3_country_sleeves["Window"] == full_country_label].copy()
    if len(full_sleeve_rows):
        a3_optimal_country_sleeve = full_sleeve_rows.set_index("date")["A3-implied EM country sleeve"].sort_index()
        monthly["A3 Optimal EM Country Basket (In-Sample)"] = a3_optimal_country_sleeve.reindex(monthly.index)
    else:
        a3_optimal_country_sleeve = pd.Series(dtype=float, name="A3 Optimal EM Country Basket (In-Sample)")
        monthly["A3 Optimal EM Country Basket (In-Sample)"] = np.nan

    # ---------------------------------------------------------------- 8. all EM exposures, rolling windows and blocks
    EXPOSURE_DATA = {
        "Broad EM": monthly["EM"], "EM ex-IT": monthly["EM_exIT"], "EM ex-TWKR": monthly["EM_exTWKR"],
        "EM Small Cap Net": monthly["EM Small Cap Net"],
        "Optimal EM Countries OOS (Max Sharpe)": monthly["Optimal EM Countries OOS (Max Sharpe)"],
        "Optimal EM Countries OOS (Min Variance)": monthly["Optimal EM Countries OOS (Min Variance)"]}
    free_rows, free_w_rows, fixed_rows_all = [], [], []
    for years in WINDOW_YEARS:
        for label, (start, end) in rolling_windows[years].items():
            for exposure, series in EXPOSURE_DATA.items():
                d = pd.concat([monthly[BASE_ASSETS], series.rename(exposure)], axis=1).loc[start:end].dropna()
                if len(d) < 12 * years: continue
                rf_w = rf_monthly.reindex(d.index)
                if rf_w.isna().any(): continue
                stats, w_sh, w_mv, used = window_summary(d, rf_w)
                free_rows.append({"Window length": f"{years}y", "Window": label, "Exposure": exposure, **stats,
                                  "Max-Sharpe EM weight %": w_sh[exposure] * 100, "Min-Variance EM weight %": w_mv[exposure] * 100})
                for kind, w in [("Max Sharpe", w_sh), ("Min Variance", w_mv)]:
                    for asset, weight in w.items():
                        free_w_rows.append({"Window length": f"{years}y", "Window": label, "Exposure": exposure,
                                            "Portfolio": kind, "Asset": asset, "Weight %": weight * 100})
                fixed_data = monthly[BASE_ASSETS].join(series.rename(exposure))
                f10 = fixed_em_grid(fixed_data, rf_monthly, exposure, start, end, min_obs=12 * years)
                if len(f10):
                    f10.insert(0, "Window", label); f10.insert(0, "Window length", f"{years}y")
                    fixed_rows_all.append(f10)
    free_optimization = pd.DataFrame(free_rows)
    free_weights = pd.DataFrame(free_w_rows)
    fixed_em_all = pd.concat(fixed_rows_all, ignore_index=True) if fixed_rows_all else pd.DataFrame()
    fixed_em_10 = fixed_em_all.loc[fixed_em_all["EM %"] == 10].copy() if len(fixed_em_all) else pd.DataFrame()

    block_rows = []
    for block, (start, end) in FIXED_BLOCKS.items():
        for exposure, series in EXPOSURE_DATA.items():
            d = pd.concat([monthly[BASE_ASSETS], series.rename(exposure)], axis=1).loc[start:end].dropna()
            if len(d) < 24: continue
            rf_w = rf_monthly.reindex(d.index)
            if rf_w.isna().any(): continue
            stats, w_sh, w_mv, used = window_summary(d, rf_w)
            block_rows.append({"Block": block, "Exposure": exposure, **stats,
                               "Max-Sharpe EM weight %": w_sh[exposure] * 100, "Min-Variance EM weight %": w_mv[exposure] * 100})
    block_optimization = pd.DataFrame(block_rows)

    # ---------------------------------------------------------------- 9. efficient-frontier comparisons
    def plot_frontier_comparison(exposures, title, filename, start=None, end=END, min_obs=36):
        fig, ax = plt.subplots(figsize=(10.5, 6))
        colors = {"Benchmark": NAVY, "Broad EM": GOLD, "EM ex-IT": BLUE, "EM ex-TWKR": RED,
                  "EM Small Cap Net": GREY, "Optimal EM Countries OOS (Max Sharpe)": "#2E8B57"}
        base = monthly[BASE_ASSETS].loc[start:end].dropna()
        if len(base) >= min_obs:
            f, _ = efficient_frontier(base, rf_monthly.reindex(base.index))
            ax.plot(f["Volatility %"], f["Return %"], color=colors["Benchmark"], lw=2.2, label="Benchmark: S&P 500 + US Agg")
        for label, series in exposures.items():
            d = pd.concat([monthly[BASE_ASSETS], series.rename(label)], axis=1).loc[start:end].dropna()
            if len(d) < min_obs: continue
            f, _ = efficient_frontier(d, rf_monthly.reindex(d.index))
            ax.plot(f["Volatility %"], f["Return %"], lw=1.8, color=colors.get(label), label=f"+ {label}")
        ax.set(title=title, xlabel="Annualized volatility (%)", ylabel="Annualized expected return (%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); _savefig(fig, filename)

    frontier_exposures = {k: EXPOSURE_DATA[k] for k in ["Broad EM", "EM ex-IT", "EM ex-TWKR", "EM Small Cap Net",
                                                         "Optimal EM Countries OOS (Max Sharpe)"]}
    frontier_common_start = country_oos_maxsharpe.first_valid_index()
    if frontier_common_start is not None:
        plot_frontier_comparison(frontier_exposures,
                                 "Efficient Frontiers: U.S. Stock–Bond Benchmark vs. Rolling OOS EM Exposures (Common Sample)",
                                 "B_01_Efficient_Frontiers_OOS_Exposures.png", start=frontier_common_start, min_obs=24)
    if a3_optimal_country_sleeve.notna().sum() >= 24:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        common_start = a3_optimal_country_sleeve.first_valid_index()
        base = monthly[BASE_ASSETS].loc[common_start:END].dropna()
        f_base, _ = efficient_frontier(base, rf_monthly.reindex(base.index))
        ax.plot(f_base["Volatility %"], f_base["Return %"], color=NAVY, lw=2.2, label="Benchmark")
        for label, series, color in [("Broad EM", monthly["EM"], GOLD), ("EM ex-IT", monthly["EM_exIT"], BLUE),
                                     ("EM ex-TWKR", monthly["EM_exTWKR"], RED),
                                     ("A3 Optimal EM Country Basket (in-sample)", a3_optimal_country_sleeve, "#2E8B57")]:
            d = pd.concat([monthly[BASE_ASSETS], series.rename(label)], axis=1).loc[common_start:END].dropna()
            f, _ = efficient_frontier(d, rf_monthly.reindex(d.index))
            ax.plot(f["Volatility %"], f["Return %"], lw=1.8, color=color, label=label)
        ax.set(title="In-Sample Efficient Frontier: A3-Optimised EM Country Sleeve", xlabel="Annualized volatility (%)",
               ylabel="Annualized expected return (%)")
        ax.legend(frameon=False, fontsize=8); fig.tight_layout()
        _savefig(fig, "B_01b_Efficient_Frontier_A3_Country_Sleeve_InSample.png")

    # ---------------------------------------------------------------- 10. OOS backtest (60-month lookback)
    OOS_STRATEGIES = {}; OOS_WEIGHTS = {}
    OOS_INPUTS = {"Benchmark": BASE_ASSETS, "+ Broad EM": BASE_ASSETS + ["EM"], "+ EM ex-IT": BASE_ASSETS + ["EM_exIT"],
                  "+ EM ex-TWKR": BASE_ASSETS + ["EM_exTWKR"], "+ EM Small Cap Net": BASE_ASSETS + ["EM Small Cap Net"],
                  "+ Optimal EM Countries OOS": BASE_ASSETS + ["Optimal EM Countries OOS (Max Sharpe)"]}
    for name, cols in OOS_INPUTS.items():
        d = monthly[cols].dropna()
        if len(d) <= LOOKBACK_MONTHS: continue
        for mode in ["minvar", "sharpe"]:
            strat, weights = rolling_mvo_oos(d, rf_monthly.reindex(d.index), LOOKBACK_MONTHS, mode=mode)
            OOS_STRATEGIES[f"{name} ({mode})"] = strat
            if len(weights):
                weights.columns = cols; OOS_WEIGHTS[f"{name} ({mode})"] = weights
    portfolio_oos_returns = pd.concat(OOS_STRATEGIES, axis=1).sort_index()
    portfolio_oos_performance = performance_table(portfolio_oos_returns, rf_monthly)

    fixed10_oos = {}; fixed10_oos_weights = {}
    for label, asset in [("Broad EM", "EM"), ("EM ex-IT", "EM_exIT"), ("EM ex-TWKR", "EM_exTWKR"),
                         ("EM Small Cap Net", "EM Small Cap Net"), ("Optimal EM Countries OOS", "Optimal EM Countries OOS (Max Sharpe)")]:
        cols = BASE_ASSETS + [asset]; d = monthly[cols].dropna()
        if len(d) <= LOOKBACK_MONTHS: continue
        for mode in ["minvar", "sharpe"]:
            strat, weights = rolling_mvo_oos(d, rf_monthly.reindex(d.index), LOOKBACK_MONTHS, mode=mode, fixed_asset=asset, fixed_weight=0.10)
            strategy_label = f"10% {label} ({mode})"
            fixed10_oos[strategy_label] = strat
            if len(weights):
                weights.columns = cols; fixed10_oos_weights[strategy_label] = weights
    fixed10_oos_returns = pd.concat(fixed10_oos, axis=1).sort_index() if fixed10_oos else pd.DataFrame()
    fixed10_oos_performance = performance_table(fixed10_oos_returns, rf_monthly) if len(fixed10_oos_returns) else pd.DataFrame()
    for _strategy, _weights in fixed10_oos_weights.items():        # weight audit
        if not np.allclose(_weights.sum(axis=1).to_numpy(), 1.0, atol=1e-8):
            raise AssertionError(f"{_strategy}: fixed-10% weights do not sum to 100%")
        if not np.allclose(_weights[_weights.columns[-1]].to_numpy(), 0.10, atol=1e-8):
            raise AssertionError(f"{_strategy}: EM weight is not fixed at 10%.")

    # ---------------------------------------------------------------- 11. rolling-window charts
    if len(free_optimization):
        for years in WINDOW_YEARS:
            d = free_optimization.loc[free_optimization["Window length"] == f"{years}y"].copy()
            if d.empty: continue
            fig, ax = plt.subplots(figsize=(12, 5))
            for exposure, g in d.groupby("Exposure"):
                g = g.copy(); g["Window end"] = g["Window"].str.split("…").str[-1]
                ax.plot(pd.to_datetime(g["Window end"]), g["Max-Sharpe Sharpe"], marker="o", ms=3, label=exposure)
            ax.set(title=f"Maximum Sharpe Ratio by Rolling {years}-Year Estimation Window", xlabel="Window end", ylabel="Maximum Sharpe Ratio")
            ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
            _savefig(fig, f"B_02_Rolling_{years}Y_MaxSharpe.png")
    if len(fixed_em_10):
        fig, ax = plt.subplots(figsize=(12, 5))
        for exposure, g in fixed_em_10.groupby("Exposure"):
            g = g.loc[g["Window length"] == "5y"].copy()
            if g.empty: continue
            g["Window end"] = g["Window"].str.split("…").str[-1]
            ax.plot(pd.to_datetime(g["Window end"]), g["Max-Sharpe"], marker="o", ms=3, label=exposure)
        ax.set(title="Maximum Sharpe with EM Fixed at 10% (5-Year Rolling Windows)", xlabel="Window end", ylabel="Maximum Sharpe Ratio")
        ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
        _savefig(fig, "B_03_Fixed10_EM_Rolling_Sharpe.png")
    if len(portfolio_oos_returns):
        wealth = (1 + portfolio_oos_returns).cumprod()
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for col in wealth:
            ax.plot(wealth.index, wealth[col], label=col, lw=1.5)
        ax.set(title="Out-of-Sample Wealth: 60-Month Rolling Minimum-Variance Portfolios", xlabel="Date", ylabel="Growth of $1")
        ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
        _savefig(fig, "B_04_OOS_Portfolio_Wealth.png")
    if len(portfolio_oos_performance):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        portfolio_oos_performance["Annualized volatility %"].plot.bar(ax=axes[0], color=NAVY)
        portfolio_oos_performance["Sharpe"].plot.bar(ax=axes[1], color=GOLD)
        axes[0].set(title="OOS Annualized Volatility", ylabel="Percent", xlabel="")
        axes[1].set(title="OOS Sharpe Ratio", ylabel="Sharpe", xlabel="")
        for ax in axes: ax.tick_params(axis="x", rotation=55)
        fig.tight_layout(); _savefig(fig, "B_05_OOS_Volatility_Sharpe.png")
    if len(fixed10_oos_performance):
        fig, ax = plt.subplots(figsize=(13, 5.5))
        fixed10_oos_performance["Sharpe"].plot.bar(ax=ax, color=[NAVY if "minvar" in str(i).lower() else GOLD for i in fixed10_oos_performance.index])
        ax.set(title="OOS Sharpe: EM Fixed at 10% (Remaining 90% Re-optimised)", ylabel="Sharpe Ratio", xlabel="")
        ax.tick_params(axis="x", rotation=55)
        fig.tight_layout(); _savefig(fig, "B_10_Fixed10_OOS_Sharpe.png")

    out = {"coverage": coverage, "baseline_summary": baseline_summary, "free_optimization": free_optimization,
           "free_weights": free_weights, "fixed_em_all": fixed_em_all, "fixed_em_10": fixed_em_10,
           "block_optimization": block_optimization, "joint_country_summary": joint_country_summary,
           "joint_country_weights": joint_country_weights, "a3_country_sleeves": a3_country_sleeves,
           "portfolio_oos_performance": portfolio_oos_performance, "fixed10_oos_performance": fixed10_oos_performance}

    # ---------------------------------------------------------------- 12. full-sample DCC-GARCH (log returns; no break split)
    weekly_dcc_full = monthly_dcc_full = pd.DataFrame()
    if do_dcc:
        weekly_log_ = weekly_log()
        weekly_std, weekly_garch_diagnostics, weekly_garch_candidate_table, weekly_garch_fits, weekly_garch_selected = fit_garch_panel(weekly_log_, "Weekly")
        weekly_pair_defs = {"EM vs USA": ["EM", "USA"], "EM vs USA IT": ["EM", "USA_IT"],
                            "EM Small Cap vs USA": ["EM Small Cap Price", "USA"], "EM Small Cap vs USA IT": ["EM Small Cap Price", "USA_IT"]}
        weekly_dcc_series, weekly_dcc_params_rows = {}, []
        for label, pair in weekly_pair_defs.items():
            prow, corr_arr, idx = fit_dcc_pair(weekly_std[pair], label)
            weekly_dcc_params_rows.append(prow); weekly_dcc_series[label] = pd.Series(corr_arr[:, 0, 1], index=idx)
        weekly_dcc_full = pd.DataFrame(weekly_dcc_series)
        weekly_dcc_params = pd.DataFrame(weekly_dcc_params_rows).set_index("Pair")
        # sensitivity: the same weekly pairs under alternative persistence caps (headline cap = 0.998)
        weekly_dcc_cap_rows = []
        for cap in [0.995, 0.998, 0.999]:
            for label, pair in weekly_pair_defs.items():
                row, _, _ = fit_dcc_pair(weekly_std[pair], label, persistence_cap=cap); weekly_dcc_cap_rows.append(row)
        weekly_dcc_cap_sensitivity = pd.DataFrame(weekly_dcc_cap_rows)

        monthly_log_source = p["lret"][["EM", "EM_exIT", "EM_exTWKR", "USA", "USA_IT"]].replace([np.inf, -np.inf], np.nan)
        monthly_std, monthly_garch_diagnostics, monthly_garch_candidate_table, monthly_garch_fits, monthly_garch_selected = fit_garch_panel(monthly_log_source, "Monthly")
        monthly_pair_defs = {"EM vs USA": ["EM", "USA"], "EM ex-IT vs USA": ["EM_exIT", "USA"], "EM ex-TWKR vs USA": ["EM_exTWKR", "USA"],
                             "EM vs USA IT": ["EM", "USA_IT"], "EM ex-IT vs USA IT": ["EM_exIT", "USA_IT"], "EM ex-TWKR vs USA IT": ["EM_exTWKR", "USA_IT"]}
        monthly_dcc_series, monthly_dcc_params_rows = {}, []
        for label, pair in monthly_pair_defs.items():
            prow, corr_arr, idx = fit_dcc_pair(monthly_std[pair], label)
            monthly_dcc_params_rows.append(prow); monthly_dcc_series[label] = pd.Series(corr_arr[:, 0, 1], index=idx)
        monthly_dcc_full = pd.DataFrame(monthly_dcc_series)
        monthly_dcc_params = pd.DataFrame(monthly_dcc_params_rows).set_index("Pair")

        fig, ax = plt.subplots(figsize=(12, 5.5))
        for col in weekly_dcc_full:
            ax.plot(weekly_dcc_full.index, weekly_dcc_full[col], lw=1.05, label=col)
        ax.set(title="Full-Sample Weekly DCC-GARCH (Log Returns; Small Cap Price Proxy)", ylabel="Conditional correlation", xlabel="Date")
        ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
        _savefig(fig, "B_06_Weekly_DCC_FullSample.png")
        for benchmark, cols in [("USA", ["EM vs USA", "EM ex-IT vs USA", "EM ex-TWKR vs USA"]),
                                ("USA IT", ["EM vs USA IT", "EM ex-IT vs USA IT", "EM ex-TWKR vs USA IT"])]:
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in cols:
                ax.plot(monthly_dcc_full.index, monthly_dcc_full[col], lw=1.1, label=col)
            ax.set(title=f"Full-Sample Monthly DCC-GARCH vs {benchmark} (Log Returns)", ylabel="Conditional correlation", xlabel="Date")
            ax.legend(frameon=False, fontsize=8); fig.tight_layout()
            _savefig(fig, f"B_07_Monthly_DCC_{benchmark.replace(' ', '_')}_FullSample.png")
        out.update({"weekly_dcc_params": weekly_dcc_params, "weekly_dcc_cap_sensitivity": weekly_dcc_cap_sensitivity,
                    "weekly_garch_diagnostics": weekly_garch_diagnostics, "weekly_garch_candidate_table": weekly_garch_candidate_table,
                    "monthly_dcc_params": monthly_dcc_params, "monthly_garch_diagnostics": monthly_garch_diagnostics,
                    "monthly_garch_candidate_table": monthly_garch_candidate_table,
                    "weekly_dcc_full": weekly_dcc_full, "monthly_dcc_full": monthly_dcc_full})

    # ---------------------------------------------------------------- 13. rolling-correlation diagnostics (no break split)
    rolling_corr_rows = []
    monthly_log_for_corr = p["lret"][["EM", "EM_exIT", "EM_exTWKR", "USA", "USA_IT"]]
    for window in [36, 60]:
        for benchmark in ["USA", "USA_IT"]:
            broad = monthly_log_for_corr["EM"].rolling(window).corr(monthly_log_for_corr[benchmark])
            for asset in ["EM_exIT", "EM_exTWKR"]:
                alt = monthly_log_for_corr[asset].rolling(window).corr(monthly_log_for_corr[benchmark])
                rolling_corr_rows.append(pd.DataFrame({"Window": window, "Benchmark": benchmark, "Alternative": asset,
                                                       "Broad DCC-independent rolling correlation": broad,
                                                       "Alternative correlation": alt, "Gap": alt - broad}))
    rolling_corr_diagnostics = pd.concat(rolling_corr_rows).reset_index(names="date")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, (window, benchmark) in zip(axes.flat, [(36, "USA"), (36, "USA_IT"), (60, "USA"), (60, "USA_IT")]):
        sub = rolling_corr_diagnostics.loc[(rolling_corr_diagnostics["Window"] == window) & (rolling_corr_diagnostics["Benchmark"] == benchmark)]
        for asset, g in sub.groupby("Alternative"):
            ax.plot(g["date"], g["Gap"], label=f"{asset} minus Broad EM")
        ax.axhline(0, color="black", lw=.7); ax.set_title(f"{window}-month correlation gap vs {benchmark}")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); _savefig(fig, "B_08_Rolling_Correlation_Gaps_FullSample.png")
    out["rolling_corr_diagnostics"] = rolling_corr_diagnostics

    # ---------------------------------------------------------------- 14. export
    readme = pd.DataFrame({"Item": ["U.S. benchmark", "Return convention", "Risk-free rate", "Time windows",
                                    "Structural break handling", "Country optimizer", "A3 optimizer convention",
                                    "GARCH / DCC robustness", "Small Cap convention"],
                           "Definition": [
                               "S&P 500 total return (SPXT) + Bloomberg U.S. Aggregate (LBUSTRUU).",
                               "Simple returns for mean-variance portfolios; log returns for DCC-GARCH.",
                               "Lagged monthly 3-month T-bill (A's rf).",
                               "A's five-year blocks plus 5-year / 10-year rolling windows ending each August.",
                               "No structural break is applied in this module.",
                               "A3 joint optimizer for in-sample windows and as a 60-month rolling OOS extension; country weights are taken from the joint U.S. stock/bond/eight-country solution and normalised to an EM sleeve.",
                               "Monthly arithmetic mean x12, covariance x12, average monthly 3M T-bill x12; free weights: SLSQP, long-only, weights sum to 1, ftol=1e-12, maxiter=1000; fixed EM weights: 2,001-point grid for the S&P 500 / US Agg split.",
                               "Univariate model chosen by residual diagnostics then BIC; weekly USA_IT also tests AR-ARCH(5/10)-t. Bivariate DCC(1,1) with persistence cap 0.998; caps 0.995/0.998/0.999 as sensitivity.",
                               "EM Small Cap Net Total Return is main; the long-history price index is used only for the weekly DCC."]})
    sheets = {"README": readme, "Data_Coverage": coverage.reset_index(), "Baseline": baseline_summary,
              "Free_Optimization": free_optimization, "Free_Weights": free_weights, "Fixed_EM_Grid": fixed_em_all,
              "Fixed_10pct": fixed_em_10, "Five_Year_Blocks": block_optimization,
              "A3_Country_Summary": joint_country_summary.reset_index(), "A3_Country_Weights": joint_country_weights,
              "A3_EM_Sleeves": a3_country_sleeves, "A3_Optimizer_Status": pd.DataFrame(A3_OPTIMIZER_STATUS_LOG),
              "Country_OOS_Joint_MaxSharpe": country_oos_joint_maxsharpe_weights, "Country_OOS_Joint_MinVar": country_oos_joint_minvar_weights,
              "Country_OOS_Weights": country_oos_maxsharpe_weights, "OOS_Performance": portfolio_oos_performance.reset_index(),
              "Fixed10_OOS_Perf": fixed10_oos_performance.reset_index()}
    for k in ["weekly_dcc_params", "weekly_dcc_cap_sensitivity", "weekly_garch_diagnostics", "weekly_garch_candidate_table",
              "monthly_dcc_params", "monthly_garch_diagnostics", "monthly_garch_candidate_table"]:
        if k in out:
            sheets[k.replace("_", " ").title().replace(" ", "_")[:31]] = out[k].reset_index() if out[k].index.name else out[k]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_DIR / "B_Module_Results.xlsx", engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            if frame is None: continue
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    detailed_csv = {"OOS_Returns": portfolio_oos_returns, "Fixed10_OOS_Returns": fixed10_oos_returns,
                    "A3_EM_Sleeve_InSample": a3_optimal_country_sleeve.to_frame(), "Country_OOS_MaxSharpe": country_oos_maxsharpe.to_frame(),
                    "Country_OOS_MinVar": country_oos_minvar.to_frame(), "Weekly_DCC_Full": weekly_dcc_full,
                    "Monthly_DCC_Full": monthly_dcc_full, "Rolling_Correlations": rolling_corr_diagnostics}
    for name, frame in detailed_csv.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=True, date_format="%Y-%m-%d", encoding="utf-8-sig")
    return out
