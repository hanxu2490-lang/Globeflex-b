# GlobeFlex B — Is there still a diversification benefit to emerging markets?

Team Delta. Code for the full analysis in the methods page (Steps 1–4 and robustness).

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

It runs in about 10 seconds. Then `python make_report.py` builds the results report in `report/index.html`. It writes every table to `output/tables/` (CSV), every chart to
`output/figures/` (PNG) and a combined report to `output/results_summary.md`.

## Layout

| File | Methods page |
|---|---|
| `analysis/config.py` | All settings: sample, crisis rules, crisis list, portfolio rules |
| `analysis/data_prep.py` | Data: calendar-date alignment, log returns, EM ex-IT, Fin / Geo / Current, ΔVIX, ΔDXY, T-bill |
| `analysis/step1_dcc.py` | Step 1 — DCC-GARCH(1,1) by maximum likelihood, weekly |
| `analysis/step2_beta_tech.py` | Step 2 — Bai–Perron break, equation (1) with Post and with w_IT,t−1, Newey–West |
| `analysis/step3_crisis.py` | Step 3 — equation (2) with and without X, EM ex-IT, daily event study (−5 to +20 days) |
| `analysis/step4_portfolio.py` | Step 4 — USA + EM / EM Small Cap / EM ex-IT, rolling 60-month minimum variance, out of sample from 2006 |
| `analysis/robustness.py` | Robustness — Small Cap net returns post-2010, local-currency EM, current period from 2022 |
| `analysis/q2_report.py` | Question 2 write-up: Table 1 (all versions of equation 2), Table 2 (one row per crisis, with currency and tech parts), two-panel event figure, crisis-month drivers |
| `run_all.py` | Runs everything in order and writes `output/results_summary.md` |
| `make_report.py` | Builds the report page `report/index.html` from the tables and figures |

## Data (`data/`)

| Series | File | Used as |
|---|---|---|
| MSCI EM, USD net | `MSCI EM Returns daily.xlsx` | EM (main) |
| MSCI EM, USD price | `MSCI EM price index.xlsx` | Same-basis comparator for EM Small Cap |
| MSCI EM, local-currency net | `MSCI Emerging Net Total Return Local Index.xlsx` | Robustness (currency) |
| MSCI EM Small Cap, price / net | `EMSC.xlsx` | Price series is the main one; net series (from 2010) is the robustness check |
| MSCI EM IT returns and market cap; MSCI EM market cap | `MSCI EM IT market cap & returns.xlsx`, `MSCI EM INDX MARKET_CAP_MONTHLY.xlsx` | w_IT and EM ex-IT |
| MSCI USA net, MSCI USA IT, MSCI World net | `MSCI USA  net total return.xlsx`, `MSCI USA IT.xlsx`, `MSCI WORLD net total return.xlsx` | Benchmarks |
| GPR, TPU | `GPR monthly.xls` (column `GPR`), `TPU monthly.xlsx` (sheet `TPU_MONTHLY`) | Geo_t |
| VIX, DXY, 3-month T-bill | `VIX daily.xlsx`, `DXY daily.xlsx`, `UST3M daily.xlsx` | X_t and the risk-free rate |

## How each rule is implemented

**Confirmed by the team**
- **Named crises for the event study:** 15 events, listed in `config.CRISES`. There are 6 past events (3 financial, 3 war) and 9 current ones (2 financial, 3 war, 4 trade). Day 0 is the first trading day on or after each date.
- **Geo_t top decile:** computed over the 2001–2026 study sample. Thresholds are GPR ≥ 157.8 or TPU ≥ 168.0, which gives 53 Geo months.
- **Step 4 weights:** long-only. The EM weight stays between 0% and 100%.

**Defaults set in code (all changeable in `config.py`)**
- **Alignment:** each month uses the last available observation in that calendar month, and each week ends on Friday. Daily series are joined on dates both series share. All daily series are cut at 25 Sep 2026, the last MSCI date, so September 2026 runs only to the 25th.
- **Returns:** log returns in %. Exceptions are EM ex-IT and the portfolios, which use simple returns (see below).
- **EM ex-IT:** backed out from simple returns as R_exIT = (R_EM − w_{t−1} R_IT) / (1 − w_{t−1}), then converted to a log return. w = EM IT market cap / EM market cap at the end of the previous month.
- **Fin_t:** month-end MSCI World net level more than 10% below its running peak. The peak is taken from 1990 onward, so Jan 2001 already counts the 2000 peak.
- **X_t:** ΔVIX is the month-end change in index points. ΔDXY is the month-end log change in %.
- **Risk-free rate:** the 3-month T-bill yield at the previous month end, divided by 12.
- **Step 1:** a single joint DCC(1,1) on EM, EM Small Cap (price), USA and USA IT. It is estimated in two steps: a GARCH(1,1) for each series, then (a, b) by maximum likelihood. Standard errors come from the numerical Hessian of the second stage.
- **Step 2, Bai–Perron:**
  - The regression is EM on USA; both the constant and the slope may change at the break.
  - Each regime must cover at least 15% of the sample.
  - Break dates come from the global SSR minimum.
  - Post_t uses the single-break model. The 2–5 break models and the BIC are reported for reference.
  - supF(1) is compared with the Bai–Perron (2003) 5% critical value of 11.47.
  - The same break date is used for every index and in Step 3.
- **Same-basis comparison:** wherever EM Small Cap (price) appears, EM (price) is reported next to it.
- **Newey–West lags:** floor(4 (T/100)^(2/9)), which is 5 for the full monthly sample.
- **Non-crisis months:** months where both Fin_t = 0 and Geo_t = 0.

## Points to note when reading the results
- **Fin_t marks long recovery periods.** World stayed more than 10% below its peak from Jan 2001 to Jul 2005 and from Jun 2008 to Dec 2012. Fin_t = 1 in 134 of 309 months, and only 140 months count as non-crisis.
- **w_IT drops mechanically in Dec 2018.** It falls from 25.7% to 14.3% because the 2018 GICS change moved Tencent, Alibaba and similar names out of IT.
- **Step 4 weights are zero for a decade.** The long-only minimum-variance weight on every EM index is 0% from 2006 to 2016, so those portfolios equal USA-only through the global financial crisis.
- **Post2010 robustness:** the break falls in Oct 2009, so Post_t = 1 in every post-2010 month. Only the w_IT version of equation (1) can be estimated there.
