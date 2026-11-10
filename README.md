# GlobeFlex B — Is there still a diversification benefit to emerging markets?

Team Delta (Jun Wang, Yujia Chen, Xu Han, Yaqi Xiao, Luqi Yang). One code base for the whole project:
Question 1 follows the team's Modules A, A3, A tests, B and C; Question 2 holds the crisis regression and the event study.

## Run

```bash
pip install -r requirements.txt
python run_all.py            # full run; Module C's Monte Carlo checks and bootstraps take the longest
python run_all.py --fast     # fewer Monte Carlo / bootstrap draws in Module C, for testing only
python make_report.py        # report page in report/index.html
```

Question 1 results go to `output/q1/<module>/` (each module's Excel workbook, figures and CSV files, as in the notebooks).
Question 2 tables and figures go to `output/q2/`, and `output/results_summary.md` collects the key tables of both questions.

## Data processing

There are two data layers, one per question; every analysis file gets its data through one of them, so no module has its own reading or cleaning code.
Both read the same raw files in `data/`, use the last observation of each calendar month and the same sample (2001-01 to 2026-08).

- **Question 1 — `analysis/q1_data.py`.** `panel()` builds Module A's monthly panel once (it is cached) and Modules A, A3, A tests, B and C
  all use it: month-end simple returns in decimals, EM ex-IT and EM ex-Taiwan & Korea backed out with last month's weights, log returns
  ln(1 + r) for correlations, betas and tests, country weights, the 3-month T-bill, and the S&P 500 / U.S. Aggregate. `weekly_log()` gives
  Module B's weekly (Friday) returns for weeks ending January 2001 to August 2026. Module C also reads a few extra series for its robustness sections (MSCI World, local-currency EM,
  EM Small Cap) with the same `q1_data.load` reader.
- **Question 2 — `analysis/q2_data_prep.py`.** Builds the monthly panel in log returns in % (100 · ln P<sub>t</sub>/P<sub>t−1</sub>), as the
  methods page sets for equation (2), with EM ex-IT, the crisis flags (Fin, Geo, Current), ΔVIX, ΔDXY and the risk-free rate, plus the daily
  EM and World returns for the event study. `q2_step3_crisis.py`, `q2_robustness.py` and `report_q2.py` all use it.
- The two layers differ in units (Question 1 in decimal simple / log returns, Question 2 in log %) because each follows its own methods.
  The only link between them is Post<sub>t</sub> in equation (2), which takes Module C's break date.

## Layout

| File | Content |
|---|---|
| `analysis/q1_data.py` | Question 1 data, built as Module A builds it: month-end simple returns, EM without a part backed out with last month's weights (EM ex-IT, EM ex-Taiwan & Korea), log returns ln(1 + r), country weights, 3-month T-bill, S&P 500 and U.S. Aggregate, EM Small Cap, crisis months (World ≤ −10% from its running peak); weekly returns for Module B |
| `analysis/q1_module_a.py` | **Module A** — EM's composition (IT share, country weights), IT share of EM risk, performance by block, link to the U.S. and U.S. tech (rolling, by block, trend regression), country leave-one-out and Taiwan & Korea (T0–T7, Fig 1–7) |
| `analysis/q1_module_a3.py` | **Module A3** — mean-variance optimizer: S&P 500 + U.S. Aggregate baseline, MSCI EM / EM ex-IT fixed at 0–100% in rolling 5- and 10-year windows, optimizer-chosen EM countries (T0–T6, Fig 1–4) |
| `analysis/q1_module_a_tests.py` | **Module A tests** — block-bootstrap tests of Module A's latest-block findings (T1–T8, Holm) and of A3 (10% EM ex-IT vs EM; stability of the country choice) |
| `analysis/q1_module_b.py` | **Module B** — out-of-sample tests with A3's optimizer (MSCI EM and EM ex-IT; 60- and 120-month estimation windows; free, fixed 10% and fixed 20% EM; max Sharpe and min variance; by five-year block), joint S&P 500 + U.S. Aggregate + eight-country portfolio out of sample from 2020-04; GARCH models with `arch` (diagnostics, then BIC), weekly and monthly DCC with a 0.998 persistence cap and cap sensitivity; 36/60-month rolling-correlation gaps. A3's in-sample results are not repeated. |
| `analysis/q1_module_c.py` | **Module C** — Bai–Perron break search (robust sup-LM, simulated critical values, Monte Carlo validation), full Bai–Perron table, α/β split, date robustness, primary tests H1–H3 with Holm, with/without the break, correlation bootstrap, robustness, section 4b: equation (2) of Question 2 re-estimated in all 14 versions and checked against the Question 2 code, with the usual diagnostics (adjusted R², joint Wald test, Durbin–Watson, Ljung–Box, ARCH LM, Breusch–Pagan, Jarque–Bera, RESET, VIF), a test for a break inside the equation and sensitivity to the Post date; Sharpe-gain tests (GRS and bootstrap), tech share, EM Small Cap, China |
| `analysis/config.py` | Project paths (both questions); Question 2 settings: sample, crisis rules, crisis list, placebo settings |
| `analysis/utils.py` | Shared helpers: saving tables and figures, chart style, Markdown tables, Newey–West OLS and linear combinations |
| `analysis/q2_data_prep.py` | Question 2 data: calendar-date alignment, log returns in %, EM ex-IT, Fin / Geo / Current, ΔVIX, ΔDXY |
| `analysis/q2_step3_crisis.py` | Question 2 — equation (2) with and without X and with and without Post (Post = after Module C's break); EM ex-IT. Module E event study: crisis selection (each crisis checked against the crisis rule), daily event windows (−5 to +20 days), crisis vs non-crisis placebo test |
| `analysis/q2_robustness.py` | Question 2 robustness — local-currency EM, current period from 2022 |
| `analysis/report_q2.py` | For the report page only: the Question 2 regression table in all versions (formatted) and the two-panel event figure |
| `run_all.py` | Runs Question 1, then Question 2, and writes `output/results_summary.md` |
| `make_report.py` | Builds the report page `report/index.html` |

## Data (`data/`)

| Series | File | Used in |
|---|---|---|
| MSCI EM, USD net | `MSCI EM Returns daily.xlsx` | Q1, Q2 |
| MSCI EM IT returns and market cap; MSCI EM market cap | `MSCI EM IT market cap & returns.xlsx`, `MSCI EM INDX MARKET_CAP_MONTHLY.xlsx` | IT share, EM ex-IT |
| MSCI USA net, MSCI USA IT, MSCI World net | `MSCI USA  net total return.xlsx`, `MSCI USA IT.xlsx`, `MSCI WORLD net total return.xlsx` | Q1 benchmarks; World for Q2 and crisis months |
| MSCI country indices, 8 largest EM markets: daily levels and monthly market cap | `Country_returns_daily.xlsx`, `Country_mktcap_monthly.xlsx` | Q1 (weights, leave-one-out, optimizer) |
| S&P 500 total return and Bloomberg U.S. Aggregate | `SP500 and US Agg.xlsx` (the team's `US_index.xlsx` / `index.xlsx`) | Q1 optimizer |
| MSCI EM Small Cap price / net; MSCI EM price | `EMSC.xlsx`, `MSCI EM price index.xlsx` | Q1 (Modules B, C) |
| MSCI EM, local-currency net | `MSCI Emerging Net Total Return Local Index.xlsx` | Q1 (Module C 1e), Q2 robustness |
| GPR, TPU | `GPR monthly.xls`, `TPU monthly.xlsx` | Q2 Geo_t |
| VIX, DXY, 3-month T-bill | `VIX daily.xlsx`, `DXY daily.xlsx`, `UST3M daily.xlsx` | Q2 X_t; risk-free rate |

The teammates' notebooks used underscore file names (e.g. `MSCI_EM_Returns_daily.xlsx`); `q1_data.FILES` maps them to the files above.

## Question 2 rules
- **Sample:** January 2001 to August 2026, as in Question 1; daily data end on 31 Aug 2026.
- **Named crises:** 15 events in `config.CRISES`; day 0 is the first trading day on or after each date.
- **Crisis selection (Module E):** Fin events must have MSCI World at least 10% below its running peak at the event-month end (core event); if the 10% level is only reached between Day 0 and Day +20 on daily data, the event is retained as shock-triggered. Geo events need GPR, and trade events TPU, in the top decile in the event month. Result: 12 core events, 2 shock-triggered (China devaluation, COVID), 1 additional case that does not meet the rule (Crimea).
- **Geo_t:** GPR or TPU in its top decile over 2001–2026 (GPR ≥ 158.0 or TPU ≥ 165.2).
- **Fin_t:** month-end MSCI World more than 10% below its running peak (peak from 1990).
- **Post_t:** 1 after Module C's break (2009-10, last month of the first regime). Equation (2) is reported with and without Post.
- **X_t:** ΔVIX in points, ΔDXY as a log change in %. Newey–West lags floor(4 (T/100)^(2/9)) = 5.
- **Placebo test (Module E):** the mean CAR[−5,+20] of the crises against 5,000 placebo sets of non-crisis dates, matched by Past / Current, drawn without replacement from months with no Fin, GPR or TPU flag and outside ±25 trading days of every named event (seed 20261103). Two-sided empirical p-value centred on the placebo mean. Samples: all 15, the 12 core events, without Crimea (14), past, current.
