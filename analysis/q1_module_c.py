"""Module C (v2): when did EM's link to the U.S. change, how sure are we, and does it matter? (Question 1)

| Section | Question |
|---|---|
| 0 | Data, analysis plan |
| 1 | When did the relationship change? Tests, validation, break table |
| 1b | Complete Bai-Perron table (supF(m|0), UDmax, WDmax, BIC, LWZ) |
| 1c | Is the date a break in alpha, in beta, or both? |
| 1d | Test statistic at every candidate date (Fig C3) |
| 1e | Does the date survive other samples and benchmarks? |
| 2 | Did non-tech EM's U.S. sensitivity fall, and more than tech's? (H1-H3, Holm) |
| 2b | Results with and without the break; step vs trend; A's five-year blocks |
| 3 | Are the correlation numbers beyond sampling noise? (moving-block bootstrap) |
| 4 | Robustness of H1-H3, including +-1 month and other break dates |
| 4c | Is the Sharpe gain from adding EM beyond chance? |
| 5 | Summary for the team |
| 6, 8 | Tech share; China (appendix) |
| 7 | EM Small Cap (exploratory) |

Break rule (fixed before running): EM on the U.S., alpha and beta both allowed to break, sequential sup-F(l+1|l) at 5%
with a heteroskedasticity-robust sup-LM statistic and simulated critical values, trimming 15%, at most 5 breaks;
a break date is the last month of the earlier regime. Monthly data 2001-01 to 2026-08, log returns ln(1 + r).
Section 4b (the crisis regression of Question 2) belongs to the Question 2 code.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, linear_reset  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.stattools import durbin_watson, jarque_bera  # noqa: E402

from . import config as Cfg  # noqa: E402
from .q1_data import BLUE, GOLD, GREY, NAVY, RED, load, panel, to_month  # noqa: E402

OUT = Cfg.OUT_DIR / "q1" / "module_C"

START, END = '2001-01', '2026-08'
ALT_BREAKS = ['2009-10', '2018-12']           # alternative split: data-selected break + GICS change
ROBUST_END = '2024-12'                        # robustness: sample ends here
TRIM, MAX_BREAKS, ALPHA = 0.15, 5, 0.05       # Bai-Perron: min segment = 15% of sample, up to 5 breaks, 5% tests
BLOCK, SEED = 12, 0                           # block bootstrap: 12-month blocks
CRISIS_DD = -0.10                             # crisis month: MSCI World > 10% below its running peak


def month_end(b): return pd.Period(b, 'M').to_timestamp('M')


def display(*args, **kwargs):                 # notebook display calls are not needed when run as a script
    return None


def run(fast=False):
    """Run Module C. fast=True cuts the Monte Carlo and bootstrap draws (for testing only; the formal status
    checks need the full counts)."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    plt.rcParams.update({'figure.figsize': (10, 4), 'axes.spines.top': False, 'axes.spines.right': False})
    OUT.mkdir(parents=True, exist_ok=True)
    BOOT = 300 if fast else 2000                         # block bootstrap draws
    MC_SIZE, MC_REPS = (100, 40) if fast else (2000, 500)  # Monte Carlo replications (size / other checks)
    MFIX_REPS, MC_UD, MC_PART = (400, 40, 60) if fast else (4000, 500, 1000)
    B_FREE, B_CTRY = (199, 49) if fast else (1999, 499)
    results = {}

    # --------------------------------------------------------------------------------
    # 0. Data (Module A panel)
    # --------------------------------------------------------------------------------
    # --- A's monthly panel (simple USD net total returns; EM_exIT / EM_exTWKR built by A with last month's weights) ---
    _p = panel()
    A = _p['ret'].join(_p['tech_share'].rename('tech_share')).join(_p['W'].add_prefix('w_')).join(_p['rf'].rename('rf'))
    A = A.loc[START:END]
    Wl = A.filter(like='w_').rename(columns=lambda c: c[2:]).shift(1)          # last month's country weights

    def ex(parts):
        # EM without `parts`, A's formula on simple returns: (EM - sum w*X) / (1 - sum w); empty if any input is missing
        w, x = Wl[parts], A[parts]
        return ((A['EM'] - (w * x).sum(axis=1)) / (1 - w.sum(axis=1))).where(w.notna().all(axis=1) & x.notna().all(axis=1))

    if 'EM_exTWKR' not in A: A['EM_exTWKR'] = ex(['Taiwan', 'Korea'])      # older A files call this EM_exSel; rebuilt with A's formula

    # --- extra raw series ---
    lv = to_month(pd.DataFrame({'EMSC_P': load('EMSC.xlsx', 'EMSC_Price_MXEFSC'),            # EM Small Cap price
                                'EMSC_N': load('EMSC.xlsx', 'EMSC_Net_M1EFSC'),              # EM Small Cap net TR (from 2010-05)
                                'EM_P':   load('MSCI EM price index.xlsx'),                   # EM price (pairs with EMSC_P)
                                'WORLD':  load('MSCI WORLD net total return.xlsx')}))
    simple = A[['EM', 'EM_IT', 'USA', 'USA_IT', 'Taiwan', 'Korea', 'China', 'EM_exIT', 'EM_exTWKR']].copy()
    simple = simple.assign(EM_exTW=ex(['Taiwan']), EM_exKR=ex(['Korea']), EM_exCN=ex(['China']),
                           EM_exTWKRCN=ex(['Taiwan', 'Korea', 'China']))
    simple = simple.join(lv[['EMSC_P', 'EMSC_N', 'EM_P']].pct_change(fill_method=None))
    logr = np.log1p(simple)                                          # log returns: used for every correlation, beta and test
    logr['SPR_IT_exIT'] = logr['EM_IT'] - logr['EM_exIT']            # tech minus non-tech
    logr['SPR_EM_exTWKR'] = logr['EM'] - logr['EM_exTWKR']           # EM minus EM without Taiwan & Korea
    simple['SPR_IT_exIT'] = simple['EM_IT'] - simple['EM_exIT']      # simple-return versions: robustness only
    simple['SPR_EM_exTWKR'] = simple['EM'] - simple['EM_exTWKR']

    dd = lv['WORLD'] / lv['WORLD'].cummax() - 1                     # running peak from 1990
    crisis = (dd <= CRISIS_DD).reindex(simple.index)

    pd.DataFrame({'first month': simple.apply(lambda s: s.first_valid_index().strftime('%Y-%m')),
                  'months': simple.count()}).T

    # --------------------------------------------------------------------------------
    # 0.2 What C checks in A's statements
    # --------------------------------------------------------------------------------
    PRIMARY = pd.DataFrame([
        ('H1-USA', 'EM_exIT', 'USA', 'less'), ('H1-IT', 'EM_exIT', 'USA_IT', 'less'),
        ('H2-USA', 'SPR_IT_exIT', 'USA', 'greater'), ('H2-IT', 'SPR_IT_exIT', 'USA_IT', 'greater'),
        ('H3-USA', 'SPR_EM_exTWKR', 'USA', 'greater'), ('H3-IT', 'SPR_EM_exTWKR', 'USA_IT', 'greater')],
        columns=['id', 'dependent', 'benchmark', 'H1']).set_index('id')
    PRIMARY

    # --------------------------------------------------------------------------------
    # 1. When did EM's relationship with the U.S. change? (Bai–Perron)
    # --------------------------------------------------------------------------------
    def nw_lags(T): return int(np.floor(4 * (T / 100) ** (2 / 9)))

    def nw_cov(X, u, L):
        # Newey-West (Bartlett) covariance of OLS coefficients; L = 0 gives White
        g = X * u[:, None]; S = g.T @ g
        for l in range(1, L + 1):
            G = g[l:].T @ g[:-l]; S += (1 - l / (L + 1)) * (G + G.T)
        Q = np.linalg.inv(X.T @ X); return Q @ S @ Q

    def seg_ssr(y, X, h):
        # SSR of OLS on every segment [i, j] with length >= h, from cumulative sums
        T, q = X.shape
        cXX = np.vstack([np.zeros((1, q, q)), np.cumsum(X[:, :, None] * X[:, None, :], 0)])
        cXy = np.vstack([np.zeros((1, q)), np.cumsum(X * y[:, None], 0)]); cyy = np.concatenate([[0], np.cumsum(y ** 2)])
        S = np.full((T, T), np.inf)
        for i in range(T - h + 1):
            j = np.arange(i + h - 1, T); Am = cXX[j + 1] - cXX[i]; b = cXy[j + 1] - cXy[i]
            S[i, j] = cyy[j + 1] - cyy[i] - np.einsum('ni,ni->n', b, np.linalg.solve(Am, b[..., None])[..., 0])
        return S

    def dp_breaks(S, m, h):
        # global SSR-minimising dates for 1..m breaks; {k: (SSR, [last index of each of the first k segments])}
        T = S.shape[0]; out = {0: (S[0, T - 1], [])}; best = S[0].copy(); paths = {j: [] for j in range(T)}
        for k in range(1, m + 1):
            new, newp = np.full(T, np.inf), {}
            for j in range((k + 1) * h - 1, T):
                cand = np.arange(k * h - 1, j - h + 1)
                if len(cand) == 0: continue
                v = best[cand] + S[cand + 1, j]; b = cand[np.argmin(v)]; new[j] = v.min(); newp[j] = paths[b] + [b]
            best, paths = new, newp
            if np.isfinite(best[T - 1]): out[k] = (best[T - 1], paths[T - 1])
        return out

    def sup_lm(y, X, h, lo=0, hi=None, L=0):
        # sup over admissible dates in [lo, hi] of the robust test of "all coefficients equal before/after";
        # variance from the restricted (no-break) residuals
        hi = len(y) - 1 if hi is None else hi; ys, Xs = y[lo:hi + 1], X[lo:hi + 1]; n, q = Xs.shape
        if n < 2 * h: return -np.inf, None
        u0 = ys - Xs @ np.linalg.lstsq(Xs, ys, rcond=None)[0]; best, arg = -np.inf, None
        for tau in range(h - 1, n - h):
            Z = np.hstack([Xs, (np.arange(n) > tau)[:, None] * Xs]); b = np.linalg.lstsq(Z, ys, rcond=None)[0]
            V = nw_cov(Z, u0, L)[q:, q:]; w = float(b[q:] @ np.linalg.solve(V, b[q:]))
            if w > best: best, arg = w, lo + tau
        return best, arg

    def sim_sup_dist(q, eps, n=1000, reps=20000, seed=SEED):
        # asymptotic null distribution of the sup statistic: sup |B(r) - rB(1)|^2 / (r(1-r)), B q-dim Brownian motion
        rng = np.random.default_rng(seed); r = np.arange(1, n + 1) / n; keep = (r >= eps) & (r <= 1 - eps); out = []
        for _ in range(reps // 500):
            B = np.cumsum(rng.standard_normal((500, n, q)), 1) / np.sqrt(n); BB = B - r[None, :, None] * B[:, -1:, :]
            out.append(((BB ** 2).sum(2)[:, keep] / (r[keep] * (1 - r[keep]))).max(1))
        return np.sort(np.concatenate(out))

    def cv_seq(dist, l, alpha=ALPHA): return float(np.quantile(dist, (1 - alpha) ** (1 / (l + 1))))

    def bai_perron(y, X, trim=TRIM, max_breaks=MAX_BREAKS, alpha=ALPHA, dist=None):
        y, X = np.asarray(y, float), np.asarray(X, float); T, q = X.shape; h = int(np.ceil(trim * T))
        dp = dp_breaks(seg_ssr(y, X, h), max_breaks, h); crit, seq, chosen = [], [], 0
        for m, (ssr, _) in dp.items():
            p = (m + 1) * q + m
            crit.append({'breaks': m, 'SSR': ssr, 'BIC': np.log(ssr / T) + p * np.log(T) / T,
                         'LWZ': np.log(ssr / (T - p)) + p * 0.299 * np.log(T) ** 2.1 / T})
        for l in range(max(dp)):
            bnd = [-1] + dp[l][1] + [T - 1]
            stat = max(sup_lm(y, X, h, bnd[i] + 1, bnd[i + 1])[0] for i in range(l + 1)); cv = cv_seq(dist, l, alpha)
            seq.append({'test': f'supF({l + 1}|{l})', 'stat': stat, 'cv 5%': cv, 'reject': stat > cv})
            if stat <= cv: break
            chosen = l + 1
        return {'n': chosen, 'dates': dp[chosen][1] if chosen else [], 'dp': dp, 'seq': pd.DataFrame(seq),
                'crit': pd.DataFrame(crit).set_index('breaks'), 'h': h}

    def break_ci(y, X, dates, L, level=0.90):
        # Bai (1997): (T^ - T0)(D'QD)^2/(D'Omega D) -> argmax W(s) - |s|/2; c(90%) = 7.687, c(95%) = 11.033
        c = {0.90: 7.687, 0.95: 11.033}[level]; y, X = np.asarray(y, float), np.asarray(X, float)
        T = len(y); b = [-1] + list(dates) + [T - 1]; out = []
        for i, tb in enumerate(dates):
            lo, hi = b[i] + 1, b[i + 2]; ys, Xs = y[lo:hi + 1], X[lo:hi + 1]; t = tb - lo; n = len(ys)
            b1 = np.linalg.lstsq(Xs[:t + 1], ys[:t + 1], rcond=None)[0]; b2 = np.linalg.lstsq(Xs[t + 1:], ys[t + 1:], rcond=None)[0]
            u = np.concatenate([ys[:t + 1] - Xs[:t + 1] @ b1, ys[t + 1:] - Xs[t + 1:] @ b2])
            D = b2 - b1; Q = Xs.T @ Xs / n; g = Xs * u[:, None]; Om = g.T @ g / n
            for l in range(1, L + 1):
                G = g[l:].T @ g[:-l] / n; Om += (1 - l / (L + 1)) * (G + G.T)
            w = int(np.ceil(c * (D @ Om @ D) / (D @ Q @ D) ** 2)); out.append((max(tb - w, 0), min(tb + w, T - 1)))
        return out

    DIST = {q: sim_sup_dist(q, TRIM) for q in (1, 2)}
    pd.DataFrame({f'q={q}': [cv_seq(DIST[q], l) for l in range(4)] for q in (1, 2)},
                 index=[f'supF({l + 1}|{l}) 5% cv' for l in range(4)]).round(2)

    # --------------------------------------------------------------------------------
    # *Check:* the simulated 5% critical values for q = 2 (intercept + slope) should be close to Bai & Perron's (2003) published values (≈ 11.5, 13.0, 14.0, 14.9).
    # --------------------------------------------------------------------------------
    # --- Validation 1: Nile (known break 1898; strucchange: 90% CI ≈ 1895–1902) ---
    nile = sm.datasets.nile.load_pandas().data; yrs = nile['year'].astype(int).values
    yN, XN = nile['volume'].values, np.ones((len(nile), 1)); rN = bai_perron(yN, XN, dist=DIST[1])
    ciN = break_ci(yN, XN, rN['dates'], nw_lags(len(yN)))
    nile_date = [int(yrs[i]) for i in rN['dates']]; nile_ci = [(int(yrs[a]), int(yrs[b])) for a, b in ciN]
    print('Nile breaks:', nile_date, '| 90% CI:', nile_ci)

    # --------------------------------------------------------------------------------
    # *Check:* the simulated 5% critical values for q = 2 (intercept + slope) should be close to Bai & Perron's (2003) published values (≈ 11.5, 13.0, 14.0, 14.9).
    # --------------------------------------------------------------------------------
    # --- Validation 2: Monte Carlo, calibrated to EM ~ USA (log returns) ---
    rng = np.random.default_rng(SEED)
    x_us = logr['USA'].values; T = len(x_us); X_us = np.column_stack([np.ones(T), x_us]); y_em = logr['EM'].values
    b0 = np.linalg.lstsq(X_us, y_em, rcond=None)[0]; u0 = y_em - X_us @ b0; s0 = u0.std()
    rho = np.corrcoef(u0[1:], u0[:-1])[0, 1]
    print(f'EM~USA residuals: kurtosis {stats.kurtosis(u0, fisher=False):.2f}, AR(1) {rho:.3f}, '
          f'AR(1) of squares {np.corrcoef(u0[1:]**2, u0[:-1]**2)[0, 1]:.3f}')
    DGP = {'realistic': dict(a=0.05, b=0.90, df=None), 'stress': dict(a=0.10, b=0.85, df=5)}

    def sim_err(sig, a, b, df, burn=100):
        # AR(1) + GARCH(1,1) errors, unconditional sd = sig (scalar or array)
        n = T + burn; z = rng.standard_normal(n) if df is None else rng.standard_t(df, n) / np.sqrt(df / (df - 2))
        h, e, u = np.ones(n), np.zeros(n), np.zeros(n)
        for t in range(1, n):
            h[t] = (1 - a - b) + a * e[t - 1] ** 2 + b * h[t - 1]; e[t] = np.sqrt(h[t]) * z[t]; u[t] = rho * u[t - 1] + e[t]
        return u[burn:] * np.sqrt(1 - rho ** 2) * sig

    val = {}
    for name, kw in DGP.items():      # size of supF(1|0): data generated with no break
        val[f'size supF(1|0), {name}'] = np.mean([bai_perron(X_us @ b0 + sim_err(s0, **kw), X_us, max_breaks=1, dist=DIST[2])['n'] > 0
                                                  for _ in range(MC_SIZE)])

    r_main = bai_perron(y_em, X_us, dist=DIST[2]); tb = r_main['dates'][0] if r_main['dates'] else T // 2
    pre, post = np.arange(T) <= tb, np.arange(T) > tb
    b1 = np.linalg.lstsq(X_us[pre], y_em[pre], rcond=None)[0]; b2 = np.linalg.lstsq(X_us[post], y_em[post], rcond=None)[0]
    sig = np.where(pre, (y_em - X_us @ b1)[pre].std(), (y_em - X_us @ b2)[post].std())
    n_br, near, cover = [], [], []
    for _ in range(MC_REPS):          # one break at the estimated date with the estimated coefficients
        ys = np.where(pre, X_us @ b1, X_us @ b2) + sim_err(sig, **DGP['realistic']); r = bai_perron(ys, X_us, max_breaks=3, dist=DIST[2])
        n_br.append(r['n'])
        if r['n'] >= 1: near.append(min(abs(d - tb) for d in r['dates']) <= 12)
        if r['n'] == 1: lo, hi = break_ci(ys, X_us, r['dates'], nw_lags(T))[0]; cover.append(lo <= tb <= hi)
    n_br = np.array(n_br)
    val['power supF(1|0), one break'] = np.mean(n_br >= 1); val['size supF(2|1), one break'] = np.mean(n_br >= 2)
    val['date within ±12m, one break'] = np.mean(near); val['90% CI coverage, one break'] = np.mean(cover)

    t1, t2 = T // 3, 2 * T // 3; power2 = {}
    for lv in [(1.3, 0.9, 1.2), (1.5, 0.8, 1.5), (1.6, 0.6, 1.6)]:   # two slope breaks, intercept fixed
        beta = np.select([np.arange(T) <= t1, np.arange(T) <= t2], lv[:2], lv[2]); sel, ok = [], []
        for _ in range(300):
            r = bai_perron(beta * x_us + sim_err(s0, **DGP['realistic']), X_us, max_breaks=3, dist=DIST[2]); d = r['dp'][2][1]
            sel.append(r['n'] == 2); ok.append(abs(d[0] - t1) <= 12 and abs(d[1] - t2) <= 12)
        power2[f'β {lv[0]}→{lv[1]}→{lv[2]}'] = {'selects 2 breaks': np.mean(sel), 'DP dates within ±12m (m=2 given)': np.mean(ok)}
    power2 = pd.DataFrame(power2).T
    val['DP dates within ±12m, 2 large slope breaks'] = power2.iloc[-1, 1]

    ok = lambda k, lo, hi: lo <= val[k] <= hi
    FORMAL = {
        'dates': nile_date == [1898] and val['date within ±12m, one break'] >= 0.8 and val['DP dates within ±12m, 2 large slope breaks'] >= 0.8,
        'tests': ok('size supF(1|0), realistic', .025, .075) and ok('size supF(1|0), stress', .025, .075) and val['size supF(2|1), one break'] <= .075,
        'CIs':   len(nile_ci) == 1 and nile_ci[0][0] <= 1898 <= nile_ci[0][1] and ok('90% CI coverage, one break', .85, .95)}
    validation = pd.Series(val, name='Monte Carlo').round(3).to_frame()
    results['C1_validation'] = validation; results['C1_power_two_breaks'] = power2.round(3)
    print('Formal status:', {k: 'formal' if v else 'EXPLORATORY' for k, v in FORMAL.items()})
    display(validation); power2.round(3)

    # --------------------------------------------------------------------------------
    # *Check:* the simulated 5% critical values for q = 2 (intercept + slope) should be close to Bai & Perron's (2003) published values (≈ 11.5, 13.0, 14.0, 14.9).
    # --------------------------------------------------------------------------------
    # --- Break table: all indices x benchmarks, same rule ---
    def fmt(i, idx): return idx[i].strftime('%Y-%m')
    rows, BP = [], {}
    for dep in ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR', 'EMSC_P']:
        for bench in ['USA', 'USA_IT']:
            d = logr[[dep, bench]].dropna(); y, X = d[dep].values, np.column_stack([np.ones(len(d)), d[bench].values])
            r = bai_perron(y, X, dist=DIST[2]); ci = break_ci(y, X, r['dates'], nw_lags(len(d))) if r['dates'] else []
            BP[(dep, bench)] = r
            rows.append({'dependent': dep, 'benchmark': bench, 'sample': f'{d.index[0]:%Y-%m}…{d.index[-1]:%Y-%m}',
                         'breaks (sequential)': r['n'], 'dates': ', '.join(fmt(i, d.index) for i in r['dates']) or '—',
                         '90% CI': '; '.join(f'{fmt(a, d.index)}…{fmt(b, d.index)}' for a, b in ci) or '—',
                         'sup-F stats': ', '.join(f'{s:.1f}' for s in r['seq']['stat']),
                         'BIC': int(r['crit']['BIC'].idxmin()), 'LWZ': int(r['crit']['LWZ'].idxmin())})
    breaks_tab = pd.DataFrame(rows).set_index(['dependent', 'benchmark'])
    status = 'formal' if FORMAL['dates'] and FORMAL['tests'] else 'exploratory'
    breaks_tab['status'] = f"dates/tests {status}; CIs {'formal' if FORMAL['CIs'] else 'exploratory'}"

    main = BP[('EM', 'USA')]
    BREAKS_C = [fmt(i, logr.index) for i in main['dates']]          # main model; [] means no clear break
    PRIMARY_BREAKS, PRIMARY_LABEL = (BREAKS_C, 'C (data-selected)') if BREAKS_C else (ALT_BREAKS, 'alternative split (exploratory)')
    results['C1_breaks'] = breaks_tab
    print('Main model EM ~ USA:', main['seq'].round(2).to_dict('records'))
    print('BREAKS_C =', BREAKS_C, '| primary regimes from:', PRIMARY_LABEL, PRIMARY_BREAKS)
    breaks_tab

    # --------------------------------------------------------------------------------
    # *Check:* the simulated 5% critical values for q = 2 (intercept + slope) should be close to Bai & Perron's (2003) published values (≈ 11.5, 13.0, 14.0, 14.9).
    # --------------------------------------------------------------------------------
    def regimes(index, breaks):
        # regime number for each month: count of break dates strictly before it (break date = last month of a regime)
        return np.searchsorted(np.array([month_end(b) for b in breaks], dtype='datetime64[ns]'), index.values, side='left')

    def regime_names(breaks, index):
        e = [month_end(b) for b in breaks] + [index[-1]]; s = [index[0]] + [month_end(b) + pd.offsets.MonthEnd(1) for b in breaks]
        return [f'{a:%Y-%m}…{z:%Y-%m}' for a, z in zip(s, e)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (bench, deps) in zip(axes, [('USA', ['EM', 'EM_exIT']), ('USA_IT', ['EM', 'EM_IT', 'EM_exIT'])]):
        for dep, col, ls in zip(deps, [NAVY, GOLD, GREY] if len(deps) == 3 else [NAVY, GREY], ['-', '-', '--'] if len(deps) == 3 else ['-', '--']):
            rb = logr[dep].rolling(36).cov(logr[bench]) / logr[bench].rolling(36).var()
            rb.plot(ax=ax, color=col, ls=ls, lw=2, label=f'{dep} (36m rolling β)')
        for b in BREAKS_C: ax.axvline(month_end(b), color=RED, lw=1.2)
        for a, z in [(logr.index[i], logr.index[j]) for i, j in break_ci(y_em, X_us, main['dates'], nw_lags(T))]:
            ax.axvspan(a, z, color=RED, alpha=0.08)
        ax.axvline(pd.Timestamp('2018-12-31'), color=GREY, ls=':', lw=1)          # GICS change, as in A's figures
        ax.set(title=f'36-month rolling β on {bench}', xlabel=''); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig C1. Rolling β with C's break (red line, 90% CI shaded); GICS change Dec-2018 (dotted)", y=1.02)
    plt.savefig(OUT / 'figC1_breaks_rolling_beta_v2.png', dpi=200, bbox_inches='tight'); plt.close('all')

    # --------------------------------------------------------------------------------
    # 1b. The complete Bai–Perron table
    # --------------------------------------------------------------------------------
    # --- 1b. Standard Bai–Perron table: supF(m|0), UDmax, WDmax, on top of the sequential tests in the cell above ---


    def sim_mfix(q, eps, M=MAX_BREAKS, n=500, reps=MFIX_REPS, seed=SEED):
        # limit of supF(m|0): (1/m)[sum over segments of |dW|^2/dlambda − |W(1)|^2], maximised over partitions by DP
        rng = np.random.default_rng(seed); h = int(np.ceil(eps * n)); out = np.zeros((reps, M))
        gap = np.arange(n + 1)[None, :] - np.arange(n + 1)[:, None]; ok = gap >= h; lam = np.where(ok, gap / n, 1.0)
        for r in range(reps):
            W = np.vstack([np.zeros(q), np.cumsum(rng.standard_normal((n, q)), 0) / np.sqrt(n)])
            S = np.where(ok, ((W[None, :, :] - W[:, None, :]) ** 2).sum(2) / lam, -np.inf)
            V = S[0].copy(); base = (W[-1] ** 2).sum()
            for m in range(1, M + 1):
                V = (V[:, None] + S).max(0); out[r, m - 1] = (V[n] - base) / m
        return out

    def f_at_dates(y, X, dates):
        # robust test of "no change at these dates", same variance as sup_lm (White, restricted residuals), divided by m
        n, q = X.shape; u0 = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        Z = np.hstack([X] + [(np.arange(n) > tb)[:, None] * X for tb in dates]); b = np.linalg.lstsq(Z, y, rcond=None)[0]
        V = nw_cov(Z, u0, 0)[q:, q:]; return float(b[q:] @ np.linalg.solve(V, b[q:])) / len(dates)

    Fm = sim_mfix(2, TRIM); cv_m = np.quantile(Fm, 0.95, axis=0); a_m = cv_m[0] / cv_m
    cv_ud, cv_wd = np.quantile(Fm.max(1), 0.95), np.quantile((Fm * a_m).max(1), 0.95)
    def double_max(F): F = np.asarray(F); return F.max(), (F * a_m[:len(F)]).max()

    F_main = [f_at_dates(y_em, X_us, main['dp'][m][1]) for m in range(1, MAX_BREAKS + 1)]
    ud, wd = double_max(F_main)
    bp_std = pd.DataFrame({'statistic': F_main + [ud, wd], '5% cv': list(cv_m) + [cv_ud, cv_wd],
                           'dates (global SSR minimisers)': [', '.join(fmt(i, logr.index) for i in main['dp'][m][1]) for m in range(1, MAX_BREAKS + 1)] + ['', ''],
                           }, index=[f'supF({m}|0)' for m in range(1, MAX_BREAKS + 1)] + ['UDmax', 'WDmax'])
    bp_std['reject 5%'] = bp_std['statistic'] > bp_std['5% cv']

    seq_tab = main['seq'].set_index('test').rename(columns={'stat': 'statistic'})
    crit_tab = main['crit'].copy(); crit_tab.index.name = 'breaks'
    results['C1b_supF_m0_UD_WD'] = bp_std; results['C1b_sequential'] = seq_tab; results['C1b_BIC_LWZ'] = crit_tab

    # size of the double-maximum tests, same two null DGPs as above
    h_ = int(np.ceil(TRIM * T)); ud_rej = {}
    for name, kw in DGP.items():
        hits = []
        for _ in range(MC_UD):
            ys = X_us @ b0 + sim_err(s0, **kw); dp_ = dp_breaks(seg_ssr(ys, X_us, h_), MAX_BREAKS, h_)
            u_, w_ = double_max([f_at_dates(ys, X_us, dp_[m][1]) for m in range(1, MAX_BREAKS + 1)]); hits.append((u_ > cv_ud, w_ > cv_wd))
        ud_rej[name] = np.mean(hits, axis=0)
    ud_size = pd.DataFrame(ud_rej, index=['size UDmax', 'size WDmax']); results['C1b_UD_size'] = ud_size
    FORMAL['UD/WD'] = bool(((ud_size >= .025) & (ud_size <= .075)).all().all())
    print('Simulated supF(m|0) 5% cv (q=2):', cv_m.round(2), '| UDmax', round(cv_ud, 2), '| WDmax', round(cv_wd, 2))
    print('UD/WD status:', 'formal' if FORMAL['UD/WD'] else 'EXPLORATORY')
    display(bp_std.round(2)); display(seq_tab.round(2)); display(crit_tab.round(4)); ud_size.round(3)

    # --------------------------------------------------------------------------------
    # 1c. What is the date made of? α only, β only, or both
    # --------------------------------------------------------------------------------
    # --- 1c. What does the 2009-10 date consist of? Break in α only, β only, or both ---
    def search_partial(y, Xr, Zb, h):
        # one break in the coefficients on Zb; all columns of Xr stay in the model. Returns LM and SSR for every candidate date
        n, q = len(y), Xr.shape[1]; u0 = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]; lm, ssr = {}, {}
        for t in range(h - 1, n - h):
            Z = np.column_stack([Xr, (np.arange(n) > t)[:, None] * Zb]); b = np.linalg.lstsq(Z, y, rcond=None)[0]
            V = nw_cov(Z, u0, 0)[q:, q:]; lm[t] = float(b[q:] @ np.linalg.solve(V, b[q:])); e = y - Z @ b; ssr[t] = e @ e
        return pd.Series(lm), pd.Series(ssr)

    def ci_partial(y, Xr, Zb, tb, L, level=0.90):
        # Bai (1997) with the regressors whose coefficients change, same constants as break_ci
        c = {0.90: 7.687, 0.95: 11.033}[level]; n, q = len(y), Xr.shape[1]
        Z = np.column_stack([Xr, (np.arange(n) > tb)[:, None] * Zb]); b = np.linalg.lstsq(Z, y, rcond=None)[0]; u = y - Z @ b; D = b[q:]
        Q = Zb.T @ Zb / n; g = Zb * u[:, None]; Om = g.T @ g / n
        for l in range(1, L + 1):
            G = g[l:].T @ g[:-l] / n; Om += (1 - l / (L + 1)) * (G + G.T)
        w = int(np.ceil(c * (D @ Om @ D) / (D @ Q @ D) ** 2)); return max(tb - w, 0), min(tb + w, n - 1)

    KINDS = {'α and β': lambda X: X, 'α only': lambda X: X[:, [0]], 'β only': lambda X: X[:, [1]]}
    rows, PATHS = [], {}
    for dep in ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR']:
        for bench in ['USA', 'USA_IT']:
            d = logr[[dep, bench]].dropna(); y, X = d[dep].values, np.column_stack([np.ones(len(d)), d[bench].values]); h = int(np.ceil(TRIM * len(d)))
            at = d.index.get_loc(month_end(BREAKS_C[0]))
            for kind, pick in KINDS.items():
                Zb = pick(X); lm, ssr = search_partial(y, X, Zb, h); PATHS[(dep, bench, kind)] = (lm, ssr, d.index)
                cv = cv_seq(DIST[Zb.shape[1]], 0); a, z = ci_partial(y, X, Zb, int(ssr.idxmin()), nw_lags(len(d)))
                rows.append({'dependent': dep, 'benchmark': bench, 'what may break': kind, 'sup LM': lm.max(), 'at (LM peak)': fmt(lm.idxmax(), d.index),
                             '5% cv': cv, 'reject 5%': lm.max() > cv, 'min SSR at': fmt(ssr.idxmin(), d.index),
                             '90% CI around min SSR': f'{fmt(a, d.index)}…{fmt(z, d.index)}', f'LM at {BREAKS_C[0]}': lm[at],
                             'SSR within 1% of minimum': f'{fmt(ssr[ssr <= ssr.min() * 1.01].index.min(), d.index)}…{fmt(ssr[ssr <= ssr.min() * 1.01].index.max(), d.index)}'})
    decomp = pd.DataFrame(rows).set_index(['dependent', 'benchmark', 'what may break']); results['C1c_alpha_beta_split'] = decomp

    # Monte Carlo for the partial models on EM ~ USA: size, and 90% CI coverage at the estimated date
    rng_p = np.random.default_rng(SEED + 1); rng_keep = rng; rng = rng_p          # sim_err reads the global rng
    part_val = {}
    for kind in ['α only', 'β only']:
        Zb = KINDS[kind](X_us); cv = cv_seq(DIST[1], 0); h = int(np.ceil(TRIM * T))
        for name, kw in DGP.items():
            part_val[(kind, f'size, {name}')] = np.mean([search_partial(X_us @ b0 + sim_err(s0, **kw), X_us, Zb, h)[0].max() > cv for _ in range(MC_PART)])
        lm0, ssr0 = search_partial(y_em, X_us, Zb, h); tb0 = int(ssr0.idxmin())
        Z0 = np.column_stack([X_us, (np.arange(T) > tb0)[:, None] * Zb]); mean0 = Z0 @ np.linalg.lstsq(Z0, y_em, rcond=None)[0]; s1 = (y_em - mean0).std()
        near, cover = [], []
        for _ in range(MC_REPS):
            ys = mean0 + sim_err(s1, **DGP['realistic']); _, ss = search_partial(ys, X_us, Zb, h); tbh = int(ss.idxmin())
            a, z = ci_partial(ys, X_us, Zb, tbh, nw_lags(T)); near.append(abs(tbh - tb0) <= 12); cover.append(a <= tb0 <= z)
        part_val[(kind, 'date within ±12m')] = np.mean(near); part_val[(kind, '90% CI coverage')] = np.mean(cover)
    rng = rng_keep
    part_val = pd.Series(part_val).unstack(); results['C1c_validation'] = part_val
    FORMAL['α/β split'] = bool(part_val.filter(like='size').stack().between(.025, .075).all() and part_val['90% CI coverage'].between(.85, .95).all() and (part_val['date within ±12m'] >= .8).all())
    print('α/β split status:', 'formal' if FORMAL['α/β split'] else 'EXPLORATORY')
    display(decomp.round(2)); part_val.round(3)

    # --------------------------------------------------------------------------------
    # 1d. The test statistic at every candidate date (Fig C3)
    # --------------------------------------------------------------------------------
    # --- 1d. Fig C3: test statistic for every candidate date ---
    d_ = logr[['EM', 'USA']].dropna(); idx_ = d_.index
    lmJ, ssrJ, _ = PATHS[('EM', 'USA', 'α and β')]; lmA, ssrA, _ = PATHS[('EM', 'USA', 'α only')]; lmB, ssrB, _ = PATHS[('EM', 'USA', 'β only')]
    s0_ = ((y_em - X_us @ np.linalg.lstsq(X_us, y_em, rcond=None)[0]) ** 2).sum()
    classicJ = (s0_ - ssrJ) / (ssrJ / (T - 4))                                # classical Wald form (q × F), peaks at the SSR minimum by construction
    ci_a, ci_z = break_ci(y_em, X_us, main['dates'], nw_lags(T))[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]; x_ = idx_[lmJ.index]
    ax.plot(x_, classicJ.values, color=GOLD, lw=2, ls='--', label='Classical Wald (largest at the SSR minimum)')
    ax.plot(x_, lmJ.values, color=NAVY, lw=2, label='Robust LM (the test used in section 1)')
    ax.axhline(cv_seq(DIST[2], 0), color=GREY, lw=1, ls=':', label=f'5% critical value, 2 coefficients ({cv_seq(DIST[2], 0):.1f})')
    ax.set(title='α and β may both break: how strong is the evidence at each candidate date?', ylabel='Test statistic'); ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax = axes[1]
    ax.plot(x_, lmA.values, color=BLUE, lw=2, label='Only α may break'); ax.plot(x_, lmB.values, color=GOLD, lw=2, ls='--', label='Only β may break')
    ax.axhline(cv_seq(DIST[1], 0), color=GREY, lw=1, ls=':', label=f'5% critical value, 1 coefficient ({cv_seq(DIST[1], 0):.1f})')
    ax.set(title='The same search with one coefficient at a time (robust LM)', ylabel='Test statistic', xlabel='Candidate break date (last month of the first regime)'); ax.legend(frameon=False, fontsize=9, loc='upper right')
    for ax in axes:
        ax.axvspan(idx_[ci_a], idx_[ci_z], color=RED, alpha=0.08); ax.axvline(month_end(BREAKS_C[0]), color=RED, lw=1.2)
        ax.grid(axis='y', color='#E6E8EC', lw=0.6); ax.set_ylim(bottom=0)
    axes[0].annotate(f'{BREAKS_C[0]}: minimum SSR\n(shaded: 90% CI)', (month_end(BREAKS_C[0]), classicJ.max()), xytext=(-150, -22), textcoords='offset points', fontsize=9, color='#333333')
    for ax, s, lab, off in [(axes[0], lmJ, 'LM peak', (4, -40)), (axes[1], lmA, 'α peak', (-92, 2)), (axes[1], lmB, 'β peak', (8, 7))]:
        i = s.idxmax(); ax.plot(idx_[i], s.max(), 'o', ms=8, color='white', mec='#333333', zorder=5)
        ax.annotate(f'{lab} {fmt(i, idx_)}', (idx_[i], s.max()), xytext=off, textcoords='offset points', fontsize=9, color='#333333')
    plt.tight_layout(); plt.savefig(OUT / 'figC3_break_statistic_by_date_v2.png', dpi=200, bbox_inches='tight'); plt.close('all')
    profile = pd.DataFrame({'robust LM, α and β': lmJ.values, 'classical Wald, α and β': classicJ.values, 'SSR, α and β': ssrJ.values,
                            'robust LM, α only': lmA.values, 'robust LM, β only': lmB.values}, index=idx_[lmJ.index].strftime('%Y-%m')); profile.index.name = 'candidate date'
    results['C1d_statistic_by_date'] = profile
    print('LM at', BREAKS_C[0], round(lmJ[d_.index.get_loc(month_end(BREAKS_C[0]))], 2), '| LM peak', round(lmJ.max(), 2), fmt(lmJ.idxmax(), idx_))

    # --------------------------------------------------------------------------------
    # 1e. Does the date survive other choices?
    # --------------------------------------------------------------------------------
    # --- 1e. Does the date survive other choices? Every row re-runs the date search ---
    lv_world = to_month(load('MSCI WORLD net total return.xlsx').to_frame('v'))['v']        # `lv` is reused by the Monte Carlo cell
    logr['WORLD'] = np.log(lv_world).diff().reindex(logr.index)
    lv_local = to_month(load('MSCI Emerging Net Total Return Local Index.xlsx').to_frame('v'))['v']
    logr['EM_LOCAL'] = np.log(lv_local).diff().reindex(logr.index)
    DIST_TRIM = {t: sim_sup_dist(2, t) for t in (0.10, 0.20)} | {TRIM: DIST[2]}
    def drop(d, a, z): return d.drop(d.loc[a:z].index)
    CASES = {'main: EM ~ USA, log, trim 15%': ('EM', 'USA', logr, TRIM, lambda d: d), 'trim 10%': ('EM', 'USA', logr, 0.10, lambda d: d), 'trim 20%': ('EM', 'USA', logr, 0.20, lambda d: d),
             'simple returns': ('EM', 'USA', simple, TRIM, lambda d: d), f'sample ends {ROBUST_END}': ('EM', 'USA', logr, TRIM, lambda d: d.loc[:ROBUST_END]),
             'sample starts 2003-01': ('EM', 'USA', logr, TRIM, lambda d: d.loc['2003-01':]), 'without 2008-09…2009-03': ('EM', 'USA', logr, TRIM, lambda d: drop(d, '2008-09', '2009-03')),
             'without 2008-01…2009-12': ('EM', 'USA', logr, TRIM, lambda d: drop(d, '2008-01', '2009-12')), 'benchmark MSCI World': ('EM', 'WORLD', logr, TRIM, lambda d: d),
             'benchmark U.S. tech': ('EM', 'USA_IT', logr, TRIM, lambda d: d), 'EM in local currency': ('EM_LOCAL', 'USA', logr, TRIM, lambda d: d), 'EM ex-IT': ('EM_exIT', 'USA', logr, TRIM, lambda d: d)}
    rows = {}
    for name, (dep, bench, frame, tr, f) in CASES.items():
        d = f(frame[[dep, bench]].dropna()); y, X = d[dep].values, np.column_stack([np.ones(len(d)), d[bench].values])
        r = bai_perron(y, X, trim=tr, dist=DIST_TRIM[tr]); ci = break_ci(y, X, r['dates'], nw_lags(len(d))) if r['dates'] else []
        stat, arg = sup_lm(y, X, r['h'])
        rows[name] = {'months': len(d), 'sup LM': stat, 'at (LM peak)': fmt(arg, d.index), '5% cv': cv_seq(DIST_TRIM[tr], 0), 'breaks (sequential)': r['n'],
                      'dates': ', '.join(fmt(i, d.index) for i in r['dates']) or '—', '90% CI': '; '.join(f'{fmt(a, d.index)}…{fmt(b, d.index)}' for a, b in ci) or '—',
                      'min-SSR date if 1 break': fmt(r['dp'][1][1][0], d.index), 'BIC': int(r['crit']['BIC'].idxmin()), 'LWZ': int(r['crit']['LWZ'].idxmin())}
    date_rob = pd.DataFrame(rows).T.infer_objects(); results['C1e_date_robustness'] = date_rob
    date_rob.round(2)

    # --------------------------------------------------------------------------------
    # 2. Did non-tech EM's U.S. sensitivity fall, and more than tech's? (primary tests)
    # --------------------------------------------------------------------------------
    def regime_reg(df, dep, bench, breaks, L=None, mask=None):
        d = df[[dep, bench]].dropna()
        if mask is not None: d = d[mask.reindex(d.index).fillna(False).values]
        g = regimes(d.index, breaks); present = sorted(set(g))
        X = pd.DataFrame({'const': 1.0, 'b': d[bench]}, index=d.index)
        for k in present[1:]:
            X[f'R{k}'] = (g == k) * 1.0; X[f'R{k}:b'] = X[f'R{k}'] * d[bench]
        res = sm.OLS(d[dep], X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(len(d)) if L is None else L})
        return res, present

    def delta_row(df, dep, bench, breaks, alt=None, **kw):
        res, present = regime_reg(df, dep, bench, breaks, **kw); k = present[-1]; out = {'n': int(res.nobs), 'β first': res.params['b']}
        if len(present) > 1:
            d, se = res.params[f'R{k}:b'], res.bse[f'R{k}:b']; bl = res.t_test(f'b + R{k}:b = 0')
            ci = np.asarray(bl.conf_int()).ravel()
            out.update({'β last': float(np.squeeze(bl.effect)), 'β last 95% CI': f'[{ci[0]:.2f}, {ci[1]:.2f}]', 'δ = last − first': d, 'SE (NW)': se, 'z': d / se})
            if alt: out['p one-sided'] = stats.norm.cdf(d / se) if alt == 'less' else stats.norm.sf(d / se)
        return out

    prim = pd.DataFrame({i: delta_row(logr, r.dependent, r.benchmark, PRIMARY_BREAKS, r.H1) for i, r in PRIMARY.iterrows()}).T
    prim.insert(0, 'H1', PRIMARY['H1']); prim.insert(0, 'benchmark', PRIMARY['benchmark']); prim.insert(0, 'dependent', PRIMARY['dependent'])
    prim['p Holm'] = multipletests(prim['p one-sided'].astype(float), method='holm')[1]
    prim['reject at 5% (Holm)'] = prim['p Holm'] < 0.05
    prim['regimes'] = f"{regime_names(PRIMARY_BREAKS, logr.index)[-1]} vs {regime_names(PRIMARY_BREAKS, logr.index)[0]} ({PRIMARY_LABEL})"
    prim['inference'] = 'conditional on break dates'
    results['C2_primary'] = prim
    prim.drop(columns=['regimes', 'inference']).round(3)

    # --------------------------------------------------------------------------------
    # 2. Did non-tech EM's U.S. sensitivity fall, and more than tech's? (primary tests)
    # --------------------------------------------------------------------------------
    # β by regime for each index (descriptive, same regimes) + correlation by regime
    names = regime_names(PRIMARY_BREAKS, logr.index); g_all = regimes(logr.index, PRIMARY_BREAKS)
    by_reg = {}
    for bench in ['USA', 'USA_IT']:
        for dep in ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR']:
            row = delta_row(logr, dep, bench, PRIMARY_BREAKS)
            for k, nm in enumerate(names):
                m = g_all == k; row[f'corr {nm}'] = logr.loc[m, dep].corr(logr.loc[m, bench])
            row['corr change'] = row[f'corr {names[-1]}'] - row[f'corr {names[0]}']
            by_reg[(bench, dep)] = row
    by_reg = pd.DataFrame(by_reg).T; results['C2_beta_corr_by_regime'] = by_reg

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    deps = ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR']
    for ax, bench in zip(axes, ['USA', 'USA_IT']):
        for j, (nm, col, mk) in enumerate([(names[0], GREY, 'o'), (names[-1], NAVY, 's')]):
            est, lo, hi = [], [], []
            for dep in deps:
                res, present = regime_reg(logr, dep, bench, PRIMARY_BREAKS)
                if j == 0: e = res.params['b']; c = res.conf_int().loc['b'].values
                else:
                    tt = res.t_test(f'b + R{present[-1]}:b = 0'); e = float(np.squeeze(tt.effect)); c = np.asarray(tt.conf_int()).ravel()
                est.append(e); lo.append(e - c[0]); hi.append(c[1] - e)
            xs = np.arange(len(deps)) + (j - 0.5) * 0.25
            ax.errorbar(xs, est, yerr=[lo, hi], fmt=mk, color=col, ms=8, capsize=3, lw=1.5, label=f'β {nm}')
        ax.set_xticks(range(len(deps))); ax.set_xticklabels(deps); ax.axhline(0, color='black', lw=0.6)
        ax.set(title=f'Fig C2. β on {bench} by regime (95% CI, NW)'); ax.legend(frameon=False, fontsize=8)
    plt.savefig(OUT / 'figC2_beta_by_regime_v2.png', dpi=200, bbox_inches='tight'); plt.close('all')
    by_reg.drop(columns=['n']).round(3)

    # --------------------------------------------------------------------------------
    # 2b. Results with and without the break
    # --------------------------------------------------------------------------------
    # --- 2b. With and without the break, on the same data ---
    def diagnostics(y, X):
        r = sm.OLS(y, X).fit(); u = r.resid
        return {'adj R²': r.rsquared_adj, 'BIC': r.bic, 'SSR': r.ssr, 'DW': durbin_watson(u), 'Ljung–Box(12) p': acorr_ljungbox(u, [12]).lb_pvalue.iloc[0],
                'ARCH(12) p': het_arch(u, nlags=12)[1], 'Breusch–Pagan p': het_breuschpagan(u, X)[1], 'Jarque–Bera p': jarque_bera(u)[1],
                'RESET p': float(linear_reset(r, power=2, use_f=True).pvalue)}

    def link_design(d, bench, kind, brk=BREAKS_C[0]):
        b = d[bench]; t = np.arange(len(d)) / (len(d) - 1); P = (d.index > month_end(brk)) * 1.0; X = pd.DataFrame({'const': 1.0, 'β': b}, index=d.index)
        if kind in ('step', 'step + trend'): X['Post'] = P; X['Post × r'] = P * b
        if kind in ('trend', 'step + trend'): X['t'] = t; X['t × r'] = t * b
        return X

    MODELS = ['no break', 'step', 'trend', 'step + trend']
    rows = {}
    for dep in ['EM', 'EM_exIT', 'EM_IT']:
        for bench in ['USA', 'USA_IT']:
            d = logr[[dep, bench]].dropna()
            for k in MODELS:
                X = link_design(d, bench, k); r = sm.OLS(d[dep], X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(len(d))})
                row = {'β (first month / first regime)': r.params['β'], 'SE β': r.bse['β']}
                for c, lab in [('Post', 'α shift'), ('Post × r', 'β shift'), ('t', 'α trend'), ('t × r', 'β trend')]:
                    row[lab] = r.params.get(c, np.nan); row[f't {lab}'] = r.tvalues.get(c, np.nan)
                rows[(dep, bench, k)] = row | diagnostics(d[dep], X)
    with_without = pd.DataFrame(rows).T; with_without.index.names = ['dependent', 'benchmark', 'model']; results['C2b_with_without'] = with_without

    # the six primary questions: no break, step, trend
    rows = {}
    for i, r_ in PRIMARY.iterrows():
        d = logr[[r_.dependent, r_.benchmark]].dropna(); fit = lambda k: sm.OLS(d[r_.dependent], link_design(d, r_.benchmark, k)).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(len(d))})
        n_, s_, t_ = fit('no break'), fit('step'), fit('trend')
        rows[i] = {'dependent': r_.dependent, 'benchmark': r_.benchmark, 'no break: β': n_.params['β'], 'SE': n_.bse['β'], 'step: δ': s_.params['Post × r'], 'SE δ': s_.bse['Post × r'], 'z δ': s_.tvalues['Post × r'],
                   'trend: β change, first to last month': t_.params['t × r'], 'SE trend': t_.bse['t × r'], 'z trend': t_.tvalues['t × r'], 'BIC no break': n_.bic, 'BIC step': s_.bic, 'BIC trend': t_.bic}
    prim_nb = pd.DataFrame(rows).T; results['C2b_primary_no_break'] = prim_nb

    # A's five-year blocks, for comparison with the step model
    BLOCKS = {'2001–05': ('2001-01', '2005-12'), '2006–10': ('2006-01', '2010-12'), '2011–15': ('2011-01', '2015-12'), '2016–20': ('2016-01', '2020-12'), '2021–26': ('2021-01', END)}
    rows = {}
    for bench in ['USA', 'USA_IT']:
        for dep in ['EM', 'EM_exIT', 'EM_IT']:
            for nm, (a, z) in BLOCKS.items():
                s = logr.loc[a:z, [dep, bench]].dropna(); r = sm.OLS(s[dep], sm.add_constant(s[bench])).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(len(s))})
                rows[(bench, dep, nm)] = {'months': len(s), 'β': r.params[bench], 'SE': r.bse[bench], 'α %/month': r.params['const'] * 100, 'SE α': r.bse['const'] * 100, 'corr': s[dep].corr(s[bench])}
    blocks = pd.DataFrame(rows).T; blocks.index.names = ['benchmark', 'dependent', 'block']; results['C2b_five_year_blocks'] = blocks

    mid = [pd.Timestamp(f'{a[:4]}-01-01') + (pd.Timestamp(z + '-28') - pd.Timestamp(a + '-01')) / 2 for a, z in BLOCKS.values()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, dep, title in zip(axes, ['EM', 'EM_exIT'], ['MSCI EM on the U.S. market', 'EM ex-IT on the U.S. market']):
        b_ = blocks.loc[('USA', dep)]
        ax.errorbar(mid, b_['β'].astype(float).values, yerr=1.96 * b_['SE'].astype(float).values, fmt='o', ms=8, color=NAVY, capsize=3, lw=1.5, label="β in A's five-year block (95% CI)")
        res_, _ = regime_reg(logr, dep, 'USA', BREAKS_C); first, last = res_.params['b'], res_.params['b'] + res_.params['R1:b']
        ax.hlines(first, logr.index[0], month_end(BREAKS_C[0]), color=GOLD, lw=2.5, label='β in the step model'); ax.hlines(last, month_end(BREAKS_C[0]), logr.index[-1], color=GOLD, lw=2.5)
        ax.axvline(month_end(BREAKS_C[0]), color=RED, lw=1.2, label=BREAKS_C[0]); ax.set(title=title, ylabel='β'); ax.grid(axis='y', color='#E6E8EC', lw=0.6); ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / 'figC4_step_vs_five_year_blocks_v2.png', dpi=200, bbox_inches='tight'); plt.close('all')
    display(with_without.loc[('EM', 'USA')].round(3).T); display(with_without.loc[('EM_exIT', 'USA')].round(3).T); display(prim_nb.infer_objects().round(3)); blocks.infer_objects().round(3)

    # --------------------------------------------------------------------------------
    # 3. Are the correlation numbers beyond sampling noise? (moving-block bootstrap)
    # --------------------------------------------------------------------------------
    def block_boot(df, stat, groups, B=BOOT, block=BLOCK, seed=SEED):
        rng = np.random.default_rng(seed); pos = [np.flatnonzero(groups == g) for g in np.unique(groups)]; out = []
        for _ in range(B):
            idx = []
            for p in pos:
                n = len(p); st = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
                idx.append(p[(st[:, None] + np.arange(block)).ravel()[:n]])
            out.append(stat(df.iloc[np.concatenate(idx)]))
        return pd.DataFrame(out)

    def corr_in(df, g, a, b): m = df['g'] == g; return df.loc[m, a].corr(df.loc[m, b])

    def a_stats(df, first, last):
        c = lambda s, g: corr_in(df, g, s, 'USA_IT')
        return {'gap EM − EM_exIT, last period': c('EM', last) - c('EM_exIT', last),
                'EM_exIT change, last − first': c('EM_exIT', last) - c('EM_exIT', first),
                'EM change, last − first': c('EM', last) - c('EM', first),
                'joint − sum of single TW/KR drops, last period': c('EM_exTW', last) + c('EM_exKR', last) - c('EM', last) - c('EM_exTWKR', last)}

    def boot_table(df, groups, first, last):
        d = df.assign(g=groups); point = pd.Series(a_stats(d, first, last)); bs = block_boot(d, lambda x: a_stats(x, first, last), groups)
        return pd.DataFrame({'estimate': point, '2.5%': bs.quantile(.025), '97.5%': bs.quantile(.975),
                             'share of draws ≤ 0': (bs <= 0).mean()})

    cols = ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR', 'EM_exTW', 'EM_exKR', 'EMSC_P', 'USA', 'USA_IT']
    dL = logr[cols].dropna()                                          # log returns, same months for every series
    gA, gB = regimes(dL.index, PRIMARY_BREAKS), regimes(dL.index, ALT_BREAKS)
    bootA = boot_table(dL, gA, 0, gA.max())
    bootB = boot_table(dL, gB, 0, gB.max())
    boot = pd.concat({f'Panel A: primary regimes ({PRIMARY_LABEL})': bootA,
                      f'Panel B: alternative split {ALT_BREAKS} (last = 2019-01…2026-08)': bootB})
    results['C3_bootstrap'] = boot
    boot.round(3)

    # --------------------------------------------------------------------------------
    # 3. Are the correlation numbers beyond sampling noise? (moving-block bootstrap)
    # --------------------------------------------------------------------------------
    # correlation change (last − first regime) with bootstrap CI, log returns; Small Cap added
    def corr_changes(df):
        out = {}
        for bench in ['USA', 'USA_IT']:
            for dep in ['EM', 'EM_IT', 'EM_exIT', 'EM_exTWKR', 'EMSC_P']:
                out[f'{dep} ~ {bench}'] = corr_in(df, df['g'].max(), dep, bench) - corr_in(df, 0, dep, bench)
            out[f'EMSC_P − EM (difference of changes) ~ {bench}'] = out[f'EMSC_P ~ {bench}'] - out[f'EM ~ {bench}']
        return out
    dC = dL.assign(g=gA)
    bs = block_boot(dC, corr_changes, gA)
    corr_ch = pd.DataFrame({'corr change': pd.Series(corr_changes(dC)), '2.5%': bs.quantile(.025), '97.5%': bs.quantile(.975)})
    results['C3_corr_change'] = corr_ch
    corr_ch.round(3)

    # --------------------------------------------------------------------------------
    # 4. Robustness of the primary tests
    # --------------------------------------------------------------------------------
    pre_rob = pd.Series(logr.index <= month_end(ROBUST_END), index=logr.index)
    print('Non-crisis months per primary regime:', dict(zip(regime_names(PRIMARY_BREAKS, logr.index),
          pd.Series(~crisis.values).groupby(regimes(logr.index, PRIMARY_BREAKS)).sum().astype(int).tolist())))
    EARLIER = [str(pd.Period(b, 'M') - 1) for b in PRIMARY_BREAKS]   # Oct-2009 in the post period
    LATER = [str(pd.Period(b, 'M') + 1) for b in PRIMARY_BREAKS]
    VARIANTS = {'main (log, NW auto)': dict(df=logr), 'simple returns': dict(df=simple), 'NW lags = 12': dict(df=logr, L=12),
                'non-crisis months only': dict(df=logr, mask=~crisis), f'sample ending {ROBUST_END}': dict(df=logr, mask=pre_rob),
                f'break one month earlier ({", ".join(EARLIER)})': dict(df=logr, breaks=EARLIER),
                f'break one month later ({", ".join(LATER)})': dict(df=logr, breaks=LATER)}
    if PRIMARY_BREAKS != ALT_BREAKS:
        VARIANTS[f'adds GICS split 2018-12 (2019–26 vs first)'] = dict(df=logr, breaks=ALT_BREAKS)
    rob = {}
    for name, kw in VARIANTS.items():
        kw = dict(kw); df = kw.pop('df'); brk = kw.pop('breaks', PRIMARY_BREAKS)
        for i, r in PRIMARY.iterrows():
            row = delta_row(df, r.dependent, r.benchmark, brk, r.H1, **kw)
            rob[(name, 'δ')] = rob.get((name, 'δ'), {}) | {i: row.get('δ = last − first', np.nan)}
            rob[(name, 'p')] = rob.get((name, 'p'), {}) | {i: row.get('p one-sided', np.nan)}
    rob = pd.DataFrame(rob).T
    results['C4_robustness'] = rob

    post19 = {}
    for i, r in PRIMARY.iterrows():
        res, _ = regime_reg(logr.loc['2019-01':], r.dependent, r.benchmark, [])
        post19[i] = {'β 2019-01…2026-08': res.params['b'], 'SE (NW)': res.bse['b'], 'β last regime (main)': prim.loc[i, 'β last']}
    post19 = pd.DataFrame(post19).T; results['C4_post2019'] = post19
    display(rob.round(3)); post19.round(3)

    # --------------------------------------------------------------------------------
    # 4. Robustness of the primary tests
    # --------------------------------------------------------------------------------
    # --- δ for the primary tests at other break dates inside the transition window (raw one-sided p, no Holm) ---
    DATES_ = ['2007-10', '2009-10', '2010-09', '2012-12']
    rows = {}
    for b_ in DATES_:
        for i, r_ in PRIMARY.iterrows():
            o = delta_row(logr, r_.dependent, r_.benchmark, [b_], r_.H1); rows[(b_, 'δ', i)] = o['δ = last − first']; rows[(b_, 'p', i)] = o['p one-sided']
    across = pd.Series(rows).unstack(2)[PRIMARY.index]; results['C4_across_break_dates'] = across
    across.astype(float).round(3)

    # --------------------------------------------------------------------------------
    # 4c. Is the Sharpe gain from adding EM beyond chance? (exploratory, added in v2)
    # --------------------------------------------------------------------------------
    # --- 4c. Is the Sharpe gain from adding EM beyond chance? A3's portfolios, tested ---
    idx_x = load('index.xlsx', 'S&P500 daily'); agg_x = load('index.xlsx', 'LBUSTRUU Index daily')
    lv_us = to_month(pd.DataFrame({'SPX': idx_x, 'AGG': agg_x})); r_us = lv_us.pct_change(fill_method=None).loc[START:END]
    rf_m = (to_month(load('UST3M daily.xlsx').to_frame('v'))['v'] / 100 / 12).shift(1).reindex(r_us.index)
    COUNTRIES = ['Taiwan', 'Korea', 'China', 'India', 'Brazil', 'South Africa', 'Mexico', 'Saudi Arabia']
    PF = r_us.join(A[['EM', 'EM_exIT'] + COUNTRIES]); PF['rf'] = rf_m                      # simple returns: portfolio arithmetic, as A3
    BOOT_BLOCK = {60: 6, 120: 12}

    STEP = 0.01; w1, w2 = np.meshgrid(np.arange(0, 1 + STEP / 2, STEP), np.arange(0, 1 + STEP / 2, STEP)); keep = w1 + w2 <= 1 + 1e-9
    W3 = np.column_stack([w1[keep], w2[keep], (1 - w1[keep] - w2[keep]).clip(0)]); W3[:, 2] = np.round(W3[:, 2], 2)   # S&P 500, Agg, EM
    IS0, IS10 = W3[:, 2] == 0, W3[:, 2] == 0.10

    def sharpes3(R, rf):
        mu = R.mean(0) * 12 - rf.mean() * 12; S = np.cov(R, rowvar=False) * 12
        return (W3 @ mu) / np.sqrt(np.einsum('ij,jk,ik->i', W3, S, W3))
    def gains(R, rf):
        s = sharpes3(R, rf); b = s[IS0].max(); return s.max() - b, s[IS10].max() - b, b, W3[s.argmax(), 2]

    def grs(E, F):
        # exact F test (Gibbons–Ross–Shanken) that the N test assets add nothing to the K benchmark assets. E: T×N, F: T×K, both excess returns
        T_, N, K = E.shape[0], E.shape[1], F.shape[1]; X = np.column_stack([np.ones(T_), F]); B = np.linalg.lstsq(X, E, rcond=None)[0]; U = E - X @ B
        Sig = U.T @ U / T_; a = B[0]; m = F.mean(0); V = np.cov(F, rowvar=False, bias=True); th2 = m @ np.linalg.solve(V, m); q = a @ np.linalg.solve(Sig, a)
        Fs = (T_ - N - K) / N * q / (1 + th2); return Fs, stats.f.sf(Fs, N, T_ - N - K), (np.sqrt(th2 + q) - np.sqrt(th2)) * np.sqrt(12)

    def blocks_idx(n, L, r_):
        st = r_.integers(0, n - L + 1, size=int(np.ceil(n / L))); return (st[:, None] + np.arange(L)).ravel()[:n]

    def test_em(asset, a, z, B=B_FREE, seed=SEED):
        d = PF.loc[a:z, ['SPX', 'AGG', asset, 'rf']].dropna(); R = d[['SPX', 'AGG', asset]].values; rf = d['rf'].values; n = len(d)
        Fe = R[:, :2] - rf[:, None]; Ee = (R[:, 2] - rf)[:, None]; g_free, g10, base, w_em = gains(R, rf); Fg, pg, dS = grs(Ee, Fe)
        X = np.column_stack([np.ones(n), Fe]); bb = np.linalg.lstsq(X, Ee[:, 0], rcond=None)[0]; e = Ee[:, 0] - X @ bb; r_ = np.random.default_rng(seed)
        L = BOOT_BLOCK.get(n, max(6, n // 10)); nf, n10 = np.zeros(B), np.zeros(B)
        for i in range(B):
            k = blocks_idx(n, L, r_); rk = rf[k]; em0 = rk + Fe[k] @ bb[1:] + e[k]                       # α imposed to be zero
            nf[i], n10[i], *_ = gains(np.column_stack([R[k, 0], R[k, 1], em0]), rk)
        res = sm.OLS(Ee[:, 0], X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(n)})
        return {'months': n, 'baseline Sharpe': base, 'best EM %': w_em * 100, 'Sharpe gain, free EM': g_free, 'p free': (1 + (nf >= g_free - 1e-12).sum()) / (B + 1),
                'chance gain, 95th pct': np.quantile(nf, .95), 'Sharpe gain, EM = 10%': g10, 'p 10%': (1 + (n10 >= g10 - 1e-12).sum()) / (B + 1),
                'α %/yr': bb[0] * 1200, 't α (NW)': res.tvalues[0], 'GRS p': pg}

    def roll(y):
        first = pd.Timestamp(START + '-01') + pd.DateOffset(years=y); first = pd.Timestamp(f'{first.year if first.month <= 8 else first.year + 1}-08-31')
        return {f"{(e - pd.DateOffset(months=12 * y - 1)):%Y-%m}…{e:%Y-%m}": (f"{(e - pd.DateOffset(months=12 * y - 1)):%Y-%m}", f'{e:%Y-%m}') for e in pd.date_range(first, month_end(END), freq='12ME')}

    fixed = {(asset, nm): test_em(asset, a, z) for asset in ['EM', 'EM_exIT'] for nm, (a, z) in BLOCKS.items()}
    fixed = pd.DataFrame(fixed).T.infer_objects(); fixed.index.names = ['asset', 'block']
    for col, new in [('p free', 'Holm p free'), ('p 10%', 'Holm p 10%'), ('GRS p', 'Holm GRS p')]: fixed[new] = multipletests(fixed[col].astype(float), method='holm')[1]
    rolling = pd.DataFrame({(y, asset, w): test_em(asset, a, z) for y in (5, 10) for asset in ['EM', 'EM_exIT'] for w, (a, z) in roll(y).items()}).T; rolling.index.names = ['years', 'asset', 'window']
    results['C4c_sharpe_blocks'] = fixed; results['C4c_sharpe_rolling'] = rolling

    # Check against A3's printed baseline and step-4 gains
    a3_base = [0.82, 1.02, 0.70, 0.55, 0.93, 1.33, 1.66, 1.31, 2.31, 1.86]; got = rolling.loc[(5, 'EM'), 'baseline Sharpe'].astype(float).values[:10]
    print('max |baseline Sharpe − A3|:', round(np.abs(got - a3_base).max(), 3)); assert np.abs(got - a3_base).max() < 0.02

    # Size of the bootstrap test under no added value: DGP calibrated to the 2016–20 block, EM
    def size_check(asset='EM', a='2016-01', z='2020-12', sims=300, B=299, df=None, seed=SEED + 3):
        d = PF.loc[a:z, ['SPX', 'AGG', asset, 'rf']].dropna(); rf = d['rf'].values; n = len(d); Fe = d[['SPX', 'AGG']].values - rf[:, None]; Ee = d[asset].values - rf
        X = np.column_stack([np.ones(n), Fe]); bb = np.linalg.lstsq(X, Ee, rcond=None)[0]; se = (Ee - X @ bb).std(); r_ = np.random.default_rng(seed); mf, Vf = Fe.mean(0), np.cov(Fe, rowvar=False); L_ = np.linalg.cholesky(Vf); hit = []
        for _ in range(sims):
            zf = r_.standard_normal((n, 2)) if df is None else r_.standard_t(df, (n, 2)) / np.sqrt(df / (df - 2)); ze = r_.standard_normal(n) if df is None else r_.standard_t(df, n) / np.sqrt(df / (df - 2))
            Fs = mf + zf @ L_.T; es = se * ze; Rs = np.column_stack([Fs + rf[:, None], rf + Fs @ bb[1:] + es]); gf, g10, *_ = gains(Rs, rf); X_ = np.column_stack([np.ones(n), Fs]); b_ = np.linalg.lstsq(X_, Fs @ bb[1:] + es, rcond=None)[0]; e_ = Fs @ bb[1:] + es - X_ @ b_
            nf, n10 = np.zeros(B), np.zeros(B)
            for i in range(B):
                k = blocks_idx(n, 6, r_); rk = rf[k]; nf[i], n10[i], *_ = gains(np.column_stack([Fs[k] + rk[:, None], rk + Fs[k] @ b_[1:] + e_[k]]), rk)
            hit.append(((1 + (nf >= gf - 1e-12).sum()) / (B + 1) <= .05, (1 + (n10 >= g10 - 1e-12).sum()) / (B + 1) <= .05, grs((Fs @ bb[1:] + es)[:, None], Fs)[1] <= .05))
        return np.mean(hit, axis=0)
    sharpe_size = pd.DataFrame({'normal': size_check(), 't(5)': size_check(df=5)}, index=['size, free EM gain', 'size, EM = 10% gain', 'size, GRS']); results['C4c_size'] = sharpe_size
    FORMAL['GRS'] = bool(sharpe_size.loc['size, GRS'].between(.025, .075).all())
    FORMAL['Sharpe bootstrap'] = bool(sharpe_size.drop('size, GRS').stack().between(.025, .075).all())

    # Country step: A3's eight countries, joint GRS and bootstrap of the long-only gain
    def solve_ms(R, rf):
        mu = R.mean(0) * 12 - rf.mean() * 12; S = np.cov(R, rowvar=False) * 12; n = len(mu)
        res = minimize(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-10, 'maxiter': 300})
        return -res.fun, res.x
    def base2(R, rf):
        s = np.linspace(0, 1, 501); W = np.column_stack([s, 1 - s]); mu = R.mean(0) * 12 - rf.mean() * 12; S = np.cov(R, rowvar=False) * 12
        return ((W @ mu) / np.sqrt(np.einsum('ij,jk,ik->i', W, S, W))).max()
    def test_ctry(a, z, B=B_CTRY, seed=SEED):
        d = PF.loc[a:z, ['SPX', 'AGG'] + COUNTRIES + ['rf']].dropna(); rf = d['rf'].values; n = len(d); RU = d[['SPX', 'AGG']].values; RC = d[COUNTRIES].values
        Fe = RU - rf[:, None]; Ee = RC - rf[:, None]; X = np.column_stack([np.ones(n), Fe]); bb = np.linalg.lstsq(X, Ee, rcond=None)[0]; e = Ee - X @ bb; b0_ = base2(RU, rf)
        sf, wf = solve_ms(np.column_stack([RU, RC]), rf); gain = sf - b0_; Fg, pg, dS = grs(Ee, Fe); r_ = np.random.default_rng(seed); L = BOOT_BLOCK.get(n, max(6, n // 10)); ng = np.zeros(B)
        for i in range(B):
            k = blocks_idx(n, L, r_); rk = rf[k]; Rn = rk[:, None] + Fe[k] @ bb[1:] + e[k]; ng[i] = solve_ms(np.column_stack([RU[k], Rn]), rk)[0] - base2(RU[k], rk)
        return {'months': n, 'baseline Sharpe': b0_, 'Sharpe with countries': sf, 'Sharpe gain': gain, 'p gain': (1 + (ng >= gain - 1e-9).sum()) / (B + 1), 'chance gain, 95th pct': np.quantile(ng, .95), 'EM total weight %': wf[2:].sum() * 100, 'largest country': COUNTRIES[int(wf[2:].argmax())], 'its weight %': wf[2:].max() * 100, 'GRS p (8 countries jointly)': pg}
    A3_STEP4 = {'2015-09…2020-08': .010, '2016-09…2021-08': .172, '2017-09…2022-08': .158, '2018-09…2023-08': .033, '2019-09…2024-08': .098, '2020-09…2025-08': .135, '2021-09…2026-08': .177, '2015-04…2026-08 (full)': .083}
    ctry = pd.DataFrame({w: test_ctry(*w.replace(' (full)', '').split('…')) for w in A3_STEP4}).T.infer_objects(); results['C4c_sharpe_countries'] = ctry
    print('max |country gain − A3|:', round((ctry['Sharpe gain'].astype(float) - pd.Series(A3_STEP4)).abs().max(), 3))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex='col', sharey='row')
    for row, (gc, pc, yl) in enumerate([('Sharpe gain, free EM', 'p free', 'Sharpe gain, EM weight free'), ('Sharpe gain, EM = 10%', 'p 10%', 'Sharpe gain, EM fixed at 10%')]):
        for ax, y in zip(axes[row], (5, 10)):
            for asset, col, ls, lab in [('EM', NAVY, '-', 'MSCI EM'), ('EM_exIT', GOLD, '--', 'EM ex-IT')]:
                s = rolling.loc[(y, asset)]; xx = pd.to_datetime([w.split('…')[1] for w in s.index]); gv = s[gc].astype(float).values; sig = s[pc].astype(float).values <= .05
                ax.plot(xx, gv, color=col, ls=ls, lw=2, label=lab); ax.plot(xx[sig], gv[sig], 'o', ms=8, color=col, mec='white', mew=1.5, zorder=5, label=f'{lab}: p ≤ 0.05')
            ax.axhline(0, color='#333333', lw=0.8); ax.grid(axis='y', color='#E6E8EC', lw=0.6); ax.legend(frameon=False, fontsize=9)
            if row == 0: ax.set_title(f'{y}-year windows')
            else: ax.set_xlabel('Window end')
        axes[row, 0].set_ylabel(yl)
    plt.tight_layout(); plt.savefig(OUT / 'figC5_sharpe_gain_tests_v2.png', dpi=200, bbox_inches='tight'); plt.close('all')
    display(fixed.astype({c: float for c in fixed.columns}).round(3)); display(sharpe_size.round(3)); ctry.round(3)

    # --------------------------------------------------------------------------------
    # 5. Summary for the team
    # --------------------------------------------------------------------------------
    def verdict(r):
        if r['p Holm'] < 0.05: return 'supported (Holm 5%)'
        if r['p one-sided'] < 0.05: return 'nominal 5% only (not after Holm)'
        return 'no evidence of change' if r['dependent'] == 'EM_exIT' else 'no evidence of a difference'
    QUESTION = {'H1': 'Did non-tech EM sensitivity fall?', 'H2': 'Did non-tech fall more than tech?', 'H3': 'Does removing TW/KR lower the change?'}
    summary = pd.DataFrame({
        'question': [QUESTION[i[:2]] for i in prim.index], 'benchmark': prim['benchmark'],
        'δ': prim['δ = last − first'].astype(float).round(3), 'SE': prim['SE (NW)'].astype(float).round(3),
        'p one-sided': prim['p one-sided'].astype(float).round(4), 'p Holm': prim['p Holm'].astype(float).round(4),
        'verdict': prim.apply(verdict, axis=1)}, index=prim.index)
    summary['note'] = f'regimes: {PRIMARY_LABEL}; conditional on break dates'
    data_notes = pd.DataFrame({'value': {
        'sample': f'{START} … {END}, monthly, {len(logr)} months',
        'returns': 'USD net total return; ex-series built on simple returns with t−1 weights (A), then ln(1+r); all correlations, betas, regressions and bootstraps on log returns; simple returns only for the robustness row and the exact covariance split (8b); EM Small Cap = price index vs EM price index',
        'break convention': 'break month = last month of the first regime: 2001-01…2009-10 vs 2009-11…2026-08 (±1 month checked in section 4)',
        'break rule': f'EM ~ USA, α and β break, sequential sup-F at 5%, trim {TRIM}; BREAKS_C = {BREAKS_C or "[] (no clear break)"}',
        'break validation': '; '.join(f"{k}: {'formal' if v else 'exploratory'}" for k, v in FORMAL.items()),
        'primary regimes': f'{PRIMARY_LABEL}: {PRIMARY_BREAKS}',
        'standard errors': 'Newey–West, lag ⌊4(T/100)^(2/9)⌋',
        'multiple testing': 'Holm over 6 primary tests; does not cover break-search uncertainty',
        'GICS 2018': 'EM_IT / EM_exIT membership changes at Nov-2018 review; comparisons across 2018 carry this caveat',
        'crisis months': f'MSCI World ≤ {CRISIS_DD:.0%} below running peak: {int(crisis.sum())} months'}})
    results['C5_summary'] = summary; results['C5_data_notes'] = data_notes
    summary

    # --------------------------------------------------------------------------------
    # 6. Appendix: does tech share account for EM's sensitivity to U.S. tech?
    # --------------------------------------------------------------------------------
    ts = A['tech_share']; factor = ts['2018-12'].iloc[0] / ts['2018-11'].iloc[0]
    W_EXP = pd.DataFrame({'IT share (raw)': ts, 'IT share (GICS-adj.)': ts.where(ts.index >= '2018-12-31', ts * factor),
                          'TW+KR weight': A['w_Taiwan'] + A['w_Korea']}).shift(1)

    def share_reg(wcols, sample=slice(None), bench='USA_IT'):
        d = logr[['EM', bench]].join(W_EXP[wcols]).loc[sample].dropna(); X = pd.DataFrame({'const': 1.0, 'b': d[bench]}, index=d.index)
        for w in wcols: X[w] = d[w]; X[f'{w} × r'] = d[w] * d[bench]
        res = sm.OLS(d['EM'], X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(len(d))})
        out = {'n': int(res.nobs), 'adj R²': res.rsquared_adj}
        for w in wcols: out[f'δ[{w}]'] = res.params[f'{w} × r']; out[f'SE[{w}]'] = res.bse[f'{w} × r']
        if len(wcols) == 2:
            inter = X[[f'{w} × r' for w in wcols]]; out['corr of interactions'] = inter.corr().iloc[0, 1]
        return out

    tech = pd.DataFrame({
        'IT share raw, full': share_reg(['IT share (raw)']), 'IT share adj, full': share_reg(['IT share (GICS-adj.)']),
        'TW+KR weight, full': share_reg(['TW+KR weight']), 'both (horse race), full': share_reg(['IT share (GICS-adj.)', 'TW+KR weight']),
        'IT share raw, 2019+': share_reg(['IT share (raw)'], slice('2019-01', None)),
        'both, 2019+': share_reg(['IT share (raw)', 'TW+KR weight'], slice('2019-01', None))}).T
    results['C6_techshare'] = tech
    tech.round(3)

    # --------------------------------------------------------------------------------
    # 7. Exploratory: is EM Small Cap different?
    # --------------------------------------------------------------------------------
    sc = {}
    for bench in ['USA', 'USA_IT']:
        for dep in ['EM_P', 'EMSC_P']:
            sc[(bench, dep)] = delta_row(logr, dep, bench, PRIMARY_BREAKS)
    logr['SPR_SC_EM'] = logr['EMSC_P'] - logr['EM_P']; logr['SPR_SCN_EM'] = logr['EMSC_N'] - logr['EM']
    for bench in ['USA', 'USA_IT']:
        sc[(bench, 'spread EMSC_P − EM_P')] = delta_row(logr, 'SPR_SC_EM', bench, PRIMARY_BREAKS)
    sc = pd.DataFrame(sc).T
    col = {'EM_P': 'EM_P', 'EMSC_P': 'EMSC_P', 'spread EMSC_P − EM_P': 'SPR_SC_EM'}
    for k, nm in enumerate(names):
        m = g_all == k; sc[f'corr {nm}'] = [logr.loc[m, col[d]].corr(logr.loc[m, b]) for b, d in sc.index]

    # net-return check: one window (2010-06 onward), EM net vs EM Small Cap net, plain β and correlation
    net = logr.loc['2010-06':]
    sc_net = pd.DataFrame({(b, d): {'β': regime_reg(net, d, b, [])[0].params['b'], 'SE (NW)': regime_reg(net, d, b, [])[0].bse['b'],
                                    'corr': net[d].corr(net[b])} for b in ['USA', 'USA_IT'] for d in ['EM', 'EMSC_N', 'EM_P', 'EMSC_P']}).T
    results['C7_smallcap'] = sc; results['C7_smallcap_net_2010'] = sc_net
    display(sc.drop(columns=['n']).round(3)); print('2010-06 … 2026-08, net vs price versions:'); sc_net.round(3)

    # --------------------------------------------------------------------------------
    # 8. Appendix: how much of the non-tech decoupling is China?
    # --------------------------------------------------------------------------------
    logr['SPR_TW_CN'] = logr['Taiwan'] - logr['China']
    cn = pd.DataFrame({k: delta_row(logr, k, 'USA_IT', PRIMARY_BREAKS) for k in ['China', 'Taiwan', 'Korea', 'SPR_TW_CN']}).T
    for k, nm in enumerate(names):
        m = g_all == k; cn[f'corr {nm}'] = [logr.loc[m, c].corr(logr.loc[m, 'USA_IT']) for c in cn.index]

    contrib = pd.DataFrame({'USA_IT': simple['USA_IT'], 'total': simple['EM_exTWKR'],
                            'China': Wl['China'] / (1 - Wl['Taiwan'] - Wl['Korea']) * simple['China']}).dropna()
    contrib['rest'] = contrib['total'] - contrib['China']
    def cov_attr(df):
        out = {}
        for part in ['total', 'China', 'rest']:
            c = [df.loc[df['g'] == g, part].cov(df.loc[df['g'] == g, 'USA_IT']) for g in (0, df['g'].max())]
            out[f'{part}: first'], out[f'{part}: last'], out[f'{part}: change'] = c[0], c[1], c[1] - c[0]
        return out
    cA = contrib.assign(g=regimes(contrib.index, PRIMARY_BREAKS)); bs = block_boot(cA, cov_attr, cA['g'].values)
    attr = pd.DataFrame({'estimate': pd.Series(cov_attr(cA)), '2.5%': bs.quantile(.025), '97.5%': bs.quantile(.975)}) * 1e4
    attr.loc['China share of total change'] = [cov_attr(cA)['China: change'] / cov_attr(cA)['total: change'], np.nan, np.nan]
    results['C8_china_country'] = cn; results['C8_china_cov_attr'] = attr
    display(cn.drop(columns=['n']).round(3)); print('Covariances × 10⁴ (share row is a ratio):'); attr.round(3)

    # --------------------------------------------------------------------------------
    # 9. Save outputs
    # --------------------------------------------------------------------------------
    with pd.ExcelWriter(OUT / 'module_C_results_v2.xlsx') as xl:
        pd.DataFrame({'BREAKS_C': [', '.join(BREAKS_C) or '[] no clear break'], 'primary regimes': [f'{PRIMARY_LABEL}: {PRIMARY_BREAKS}'],
                      'status': [', '.join(f"{k}: {'formal' if v else 'exploratory'}" for k, v in FORMAL.items())],
                      'convention': ['break month = last month of the first regime (2009-10 in pre; post starts 2009-11)'],
                      'reading': ['2009-10 is the point estimate of the joint α + β model inside a post-GFC transition; see C1c and C1e']}).T.to_excel(xl, sheet_name='README', header=False)
        for name, df in results.items():
            df.to_excel(xl, sheet_name=name[:31])
    return {'results': results, 'BREAKS_C': BREAKS_C, 'FORMAL': FORMAL, 'logr': logr}
