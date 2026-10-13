# -*- coding: utf-8 -*-
"""Build the results report (report/index.html) from the tables in output/.

Run `python run_all.py` first. Tables are filled from the CSV files; the written text describes
the results of the run of 13 Oct 2026 (data to 25 Sep 2026).
"""
import html
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
T = ROOT / "output" / "tables"
F = ROOT / "output" / "figures"
OUT = ROOT / "report"
FIGS = ["data_em_it_weight", "step1_dcc_correlations", "q2_event_paths", "step4_portfolios"]


def num(v, d=2, sign=False):
    if pd.isna(v):
        return "–"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def table(head, rows, cls="", groups=None):
    """head: list of header cells; rows: list of lists (already formatted HTML)."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = []
    for r in rows:
        if isinstance(r, str):                       # group heading row
            body.append(f'<tr class="grp"><td colspan="{len(head)}">{r}</td></tr>')
        else:
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def figure(name, caption):
    return (f'<figure><div class="plate"><img src="figures/{name}.png" alt="{html.escape(caption)}"></div>'
            f"<figcaption>{caption}</figcaption></figure>")


# ------------------------------------------------------------------ tables
def q1_table():
    d = pd.read_csv(T / "step2_regressions.csv", encoding="utf-8-sig")
    rows = []
    for _, r in d.iterrows():
        rows.append([r["dependent"], r["interaction"].replace("w_IT,t-1", "w<sub>IT,t−1</sub>"), r["sample"],
                     int(r["N"]), num(r["beta"]), num(r["delta"]) + stars(r["p_delta"]),
                     num(r["t_delta"]), num(r["R2"], 3)])
    return table(["Dependent", "Interaction", "Sample", "Months", "β", "δ", "t(δ)", "R²"], rows, "num")


def q2_table1():
    d = pd.read_csv(T / "q2_table1_formatted.csv", index_col=0, encoding="utf-8-sig")
    cols = list(d.columns)
    order = [("γ1 Fin", "γ<sub>1</sub> Fin"), ("γ2 Geo", "γ<sub>2</sub> Geo"),
             ("γ3 Fin × Current", "γ<sub>3</sub> Fin × Current"), ("γ4 Geo × Current", "γ<sub>4</sub> Geo × Current"),
             ("γ1+γ3 current Fin", "γ<sub>1</sub>+γ<sub>3</sub> (current Fin)"),
             ("γ2+γ4 current Geo", "γ<sub>2</sub>+γ<sub>4</sub> (current Geo)"),
             ("θ Post", "θ Post"), ("φ ΔVIX", "φ ΔVIX"), ("φ ΔDXY", "φ ΔDXY"), ("α", "α constant"),
             ("R²", "R²"), ("Months", "Months")]
    head = ["", *[c.replace(") ", ")<br>") for c in cols]]
    rows = []
    for key, lab in order:
        cells = [("" if pd.isna(v) else v) for v in d.loc[key, cols]]
        rows.append([f'<span class="rl">{lab}</span>', *cells])
    th_top = ('<tr class="panel"><th></th><th colspan="6">Panel A · Current = 1 from Jan 2018</th>'
              '<th colspan="4">Panel B · Current = 1 from Jan 2022</th></tr>')
    t = table(head, rows, "num t1")
    t = t.replace("<thead>", "<thead>" + th_top, 1)
    # highlight the core test row
    return t.replace('<tr><td><span class="rl">γ<sub>4</sub>', '<tr class="core"><td><span class="rl">γ<sub>4</sub>')


def q2_table2():
    d = pd.read_csv(T / "q2_table2_event_study.csv", encoding="utf-8-sig")
    g = pd.read_csv(T / "q2_table2_event_study_by_group.csv", index_col=0, encoding="utf-8-sig")
    heads = {"Past Fin": "Past financial crises (before 2018)", "Current Fin": "Current financial crises (from 2018)",
             "Past Geo": "Past war crises (before 2018)", "Current Geo": "Current war crises (from 2018)",
             "Current Trade": "Current trade conflicts (from 2018)"}
    rows = []
    for grp, title in heads.items():
        rows.append(title)
        sub = d[(d["period"] + " " + d["type"]) == grp]
        for _, r in sub.iterrows():
            rows.append([html.escape(r["crisis"]), r["day 0"], num(r["CAR[-5,0]"], sign=True),
                         num(r["CAR[-5,+5]"], sign=True), f"<b>{num(r['CAR[-5,+20]'], sign=True)}</b>",
                         num(r["currency part"], sign=True), num(r["local-ccy CAR[-5,+20]"], sign=True),
                         num(r["EM ex-IT CAR[-5,+20]"], sign=True), num(r["tech part"], sign=True)])
        a = g.loc[grp]
        rows.append([f"<i>Average of {int(a['events'])}</i>", "", num(a["CAR[-5,0]"], sign=True),
                     num(a["CAR[-5,+5]"], sign=True), f"<b>{num(a['CAR[-5,+20]'], sign=True)}</b>",
                     num(a["currency part"], sign=True), num(a["local-ccy CAR[-5,+20]"], sign=True),
                     num(a["EM ex-IT CAR[-5,+20]"], sign=True), num(a["tech part"], sign=True)])
    a = g.loc["Current Geo + Trade"]
    rows.append("Current war and trade together (= Geo in equation 2)")
    rows.append([f"<i>Average of {int(a['events'])}</i>", "", num(a["CAR[-5,0]"], sign=True),
                 num(a["CAR[-5,+5]"], sign=True), f"<b>{num(a['CAR[-5,+20]'], sign=True)}</b>",
                 num(a["currency part"], sign=True), num(a["local-ccy CAR[-5,+20]"], sign=True),
                 num(a["EM ex-IT CAR[-5,+20]"], sign=True), num(a["tech part"], sign=True)])
    head = ["Crisis", "Day 0", "CAR[−5, 0]", "CAR[−5, +5]", "CAR[−5, +20]", "of which currency",
            "Local-ccy CAR[−5, +20]", "EM ex-IT CAR[−5, +20]", "of which tech"]
    t = table(head, rows, "num t2")
    return t.replace("<tr><td><i>Average", '<tr class="avg"><td><i>Average')


def drivers_table():
    d = pd.read_csv(T / "q2_drivers_by_crisis_cell.csv", index_col=0, encoding="utf-8-sig")
    rows = [[k, int(r["months"]), num(r["EM − World"], sign=True), num(r["currency part (USD − local)"], sign=True),
             num(r["tech part (EM − EM ex-IT)"], sign=True), num(r["ΔDXY (%)"], sign=True),
             num(r["ΔVIX (points)"], sign=True), num(r["IT weight (%)"], 1)] for k, r in d.iterrows()]
    return table(["Months flagged", "Months", "EM − World", "Currency part", "Tech part", "ΔDXY %",
                  "ΔVIX pts", "IT weight %"], rows, "num")


def portfolio_table():
    d = pd.read_csv(T / "step4_portfolio_performance.csv", index_col=0, encoding="utf-8-sig")
    rows = [[k, num(r["ann. return %"]), num(r["ann. volatility %"]), num(r["Sharpe"]), num(r["max drawdown %"]),
             num(r["Fin months, current %"], sign=True), num(r["Geo months, current %"], sign=True),
             num(r["avg EM weight %"], 1), num(r["last EM weight %"], 1)] for k, r in d.iterrows()]
    return table(["Portfolio", "Return %/yr", "Volatility %/yr", "Sharpe", "Max drawdown %",
                  "Avg return, current Fin months %", "Avg return, current Geo months %",
                  "Avg EM weight %", "EM weight Sep 2026 %"], rows, "num")


# ------------------------------------------------------------------ page
CSS = """
:root{
  /* layout: one reading column (~70ch) with tables and figures allowed to run wider */
  --bg:#fbfbf9; --surface:#ffffff; --fg:#17181a; --fg2:#4d4f53; --muted:#7c7f85;
  --line:#e2e1dc; --accent:#1f5fae; --accent-soft:#e8f0fa; --warm:#c4521f; --core:#fff4d6;
  --plate:#fcfcfb;
  --f-display:"Source Serif 4", Georgia, "Times New Roman", serif;
  --f-body:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono:"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141517; --surface:#1c1d20; --fg:#ecebe7; --fg2:#c1c0ba; --muted:#8d8f94; --line:#2f3034;
  --accent:#78a9ea; --accent-soft:#1d2a3b; --warm:#ec8a5d; --core:#3a321c; --plate:#fcfcfb; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#141517; --surface:#1c1d20; --fg:#ecebe7; --fg2:#c1c0ba; --muted:#8d8f94; --line:#2f3034;
  --accent:#78a9ea; --accent-soft:#1d2a3b; --warm:#ec8a5d; --core:#3a321c; --plate:#fcfcfb; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.6 var(--f-body);}
.wrap{max-width:1080px;margin:0 auto;padding-inline:clamp(16px,4vw,40px);padding-block:40px 80px}
.col{max-width:72ch}
header.top{display:grid;gap:10px;padding-bottom:28px;border-bottom:1px solid var(--line);margin-bottom:8px}
.eyebrow{font:500 12px/1.4 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(28px,4.4vw,42px)/1.15 var(--f-display);margin:0;text-wrap:balance}
.lede{font-size:18px;color:var(--fg2);margin:0;max-width:70ch}
.meta{font-size:14px;color:var(--muted)}
h2{font:600 26px/1.25 var(--f-display);margin:56px 0 10px;text-wrap:balance}
h2 .q{font:500 13px var(--f-mono);color:var(--accent);letter-spacing:.06em;display:block;margin-bottom:6px;text-transform:uppercase}
h3{font:600 18px/1.35 var(--f-body);margin:34px 0 6px}
p{margin:10px 0}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px;margin:26px 0 8px}
.answer{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px;min-width:0}
.answer.main{border-color:var(--accent);background:var(--accent-soft)}
.answer h4{margin:0 0 6px;font:600 13px var(--f-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--accent)}
.answer p{margin:0;font-size:15.5px}
.verdict{border-left:3px solid var(--accent);padding:4px 0 4px 16px;margin:18px 0;font:500 19px/1.5 var(--f-display);max-width:72ch}
.tw{overflow-x:auto;margin:14px 0 6px;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font:600 12px/1.35 var(--f-body);color:var(--fg2);background:var(--bg);white-space:nowrap}
table.num td:not(:first-child),table.num th:not(:first-child){text-align:right;font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
table.num td:nth-child(2),table.num td:nth-child(3){font-family:var(--f-body)}
table.t1 td{font-family:var(--f-mono)!important;text-align:right}
table.t1 td:first-child{text-align:left;font-family:var(--f-body)!important;white-space:nowrap}
table.t1 thead th{text-align:right;white-space:normal}
table.t2 td:nth-child(2){font-family:var(--f-mono)}
table.t2 thead th{white-space:normal;min-width:64px}
table.t2 td:first-child{min-width:170px}
table.t2 td,table.t2 th{padding:7px 8px;font-size:13px}
tr.panel th{text-align:center!important;color:var(--accent);border-bottom:1px solid var(--line)}
tr.core td{background:var(--core)}
tr.grp td{font:600 12px var(--f-body);letter-spacing:.03em;color:var(--accent);background:var(--bg);padding-top:12px}
tr.avg td{background:var(--bg)}
.t{color:var(--muted);font-size:12px}
.note{font-size:13px;color:var(--muted);margin:4px 0 0;max-width:90ch}
figure{margin:22px 0 8px}
.plate{background:var(--plate);border:1px solid var(--line);border-radius:6px;padding:10px}
.plate img{display:block;width:100%;height:auto}
figcaption{font-size:13.5px;color:var(--fg2);margin-top:8px;max-width:90ch}
ul.tight{padding-left:20px;margin:8px 0}
ul.tight li{margin:5px 0}
code,.mono{font-family:var(--f-mono);font-size:.92em}
.eq{font:15px/1.6 var(--f-mono);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px;overflow-x:auto;white-space:nowrap}
.toc{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px;margin-top:6px}
.toc a{color:var(--accent);text-decoration:none}
.toc a:hover,.toc a:focus-visible{text-decoration:underline}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
"""


def build():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for f in FIGS:
        shutil.copy(F / f"{f}.png", OUT / "figures" / f"{f}.png")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GlobeFlex B Results Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="top">
  <div class="eyebrow">GlobeFlex B, San Diego · Team Delta · Results report</div>
  <h1>Is there still a diversification benefit to emerging markets?</h1>
  <p class="lede">Results of the four-step analysis on MSCI data from January 2001 to September 2026, with the focus on
  Question 2: does EM do better relative to developed markets in today's war and trade crises than in past crises?</p>
  <div class="meta">Jun Wang, Yujia Chen, Xu Han, Yaqi Xiao, Luqi Yang · Data to 25 Sep 2026 · 309 months, 1,343 weeks, 6,715 days</div>
  <nav class="toc" aria-label="Sections">
    <a href="#answers">Answers</a><a href="#data">Data</a><a href="#q1">Question 1</a>
    <a href="#q2">Question 2</a><a href="#t1">Table 1</a><a href="#t2">Table 2</a><a href="#drivers">Drivers</a>
    <a href="#limits">Limitations</a><a href="#portfolio">Portfolio</a><a href="#robust">Robustness</a>
  </nav>
</header>

<section id="answers">
<div class="answers">
  <div class="answer main"><h4>Question 2 · crises</h4>
  <p>EM's relative performance in today's war and trade crises is <b>not significantly different</b> from past crises
  (γ<sub>4</sub> = 0.19, t = 0.16). What moves it is mainly <b>the dollar</b>. Recent shocks come with a stronger dollar
  and weaker EM currencies, which costs EM about as much as its larger tech weight gains.</p></div>
  <div class="answer"><h4>Question 1 · co-movement</h4>
  <p>EM has <b>not</b> become more like the US. Weekly correlation with US equities stays around 0.6 with no trend.
  After the one structural break (Oct 2009), EM's beta to the US <i>fell</i> from 1.30 to 0.85. A higher IT weight is not
  linked to more US sensitivity.</p></div>
  <div class="answer"><h4>Portfolio · Step 4</h4>
  <p>For a US investor, minimum-variance weights gave EM <b>0% from 2006 to 2016</b>, then about 50% from 2022. Over 2006–2026,
  adding EM barely lowered volatility (15.2% to 15.1%), and the Sharpe ratio fell from 0.63 to 0.60.</p></div>
</div>
</section>

<section id="data" class="col">
<h2>Data and design</h2>
<ul class="tight">
<li>MSCI USD net total-return indices for EM, USA, World and USA IT. EM Small Cap uses the price index, so it is compared with the EM
price index on the same basis. A local-currency EM index tests the currency channel.</li>
<li>EM ex-IT is backed out from EM and EM IT returns with the IT weight w<sub>IT</sub> = EM IT / EM market value at the previous month end.</li>
<li>Crisis months. <b>Fin</b> = MSCI World more than 10% below its running peak (134 months). <b>Geo</b> = GPR or TPU in its
top decile over 2001–2026, i.e. GPR ≥ 157.8 or TPU ≥ 168.0 (53 months). <b>Current</b> = from January 2018.</li>
<li>All series are aligned by calendar date. Returns are log returns in %. Regressions use Newey–West standard errors (5 lags).</li>
</ul>
</section>
{figure("data_em_it_weight", "The IT weight in MSCI EM rose from 18% (2001) to 42% (Aug 2026). The drop in Dec 2018 is the GICS "
       "reclassification that moved Tencent, Alibaba and similar names out of IT, not a real fall in tech exposure.")}

<section id="q1">
<h2><span class="q">Question 1 · Steps 1 and 2</span>Is EM more like the US, and is technology the cause?</h2>
<div class="col">
<p>The DCC-GARCH(1,1) model (a = 0.026, b = 0.951) tracks weekly correlations. EM's correlation with the US moves between
about 0.45 and 0.75. It peaked in 2009–10 and in 2020, and was 0.57–0.58 on average in 2025–26. Correlation with US IT is lower than with
the US as a whole, which does not fit the idea of EM tracking US tech. EM Small Cap moves the same way at a slightly lower level.</p>
</div>
{figure("step1_dcc_correlations", "Weekly DCC-GARCH(1,1) correlations, 2001–2026.")}
<div class="col">
<p>The Bai–Perron test finds one significant break in EM's relation to US returns: the last month of the old regime is
<b>October 2009</b> (supF = 17.2 against a 5% critical value of 11.47; the BIC also picks one break). In equation (1),
δ is negative, not positive. EM's beta to the US was 1.30 before the break and 0.85 after it (δ = −0.45, t = −3.6).
EM ex-IT and EM Small Cap show the same fall. Replacing Post with the lagged IT weight also gives a negative δ. In non-crisis
months it is not significant. So the growing tech weight did not make EM more sensitive to US returns.</p>
</div>
{q1_table()}
<p class="note">Equation (1): r<sub>EM</sub> = α + β r<sub>US</sub> + δ (interaction × r<sub>US</sub>). w<sub>IT,t−1</sub> is a fraction,
so δ is the change in beta per 100 percentage points of IT weight. Non-crisis months: Fin = 0 and Geo = 0 (140 months).
* 10%, ** 5%, *** 1%.</p>
</section>

<section id="q2">
<h2><span class="q">Question 2 · Step 3 · main topic</span>Does EM do better relative to DM in today's crises than in past ones?</h2>
<div class="col">
<p>We regress EM's monthly return relative to MSCI World on the two crisis types and their interaction with the current period:</p>
</div>
<div class="eq">r<sub>EM,t</sub> − r<sub>World,t</sub> = α + γ<sub>1</sub>Fin<sub>t</sub> + γ<sub>2</sub>Geo<sub>t</sub> + γ<sub>3</sub>(Fin<sub>t</sub>×Current<sub>t</sub>) + γ<sub>4</sub>(Geo<sub>t</sub>×Current<sub>t</sub>) + θPost<sub>t</sub> + φ′X<sub>t</sub> + ε<sub>t</sub></div>
<div class="col">
<p>γ<sub>4</sub> is the core test. It measures how much better (positive) or worse EM does relative to World in a
war or trade crisis month today than in one in the past, in percentage points per month. γ<sub>3</sub> does the same for financial
crises. X<sub>t</sub> holds the monthly changes in the VIX and the dollar index (DXY). The sample has 116 past and 18 current Fin months, and
16 past and 37 current Geo months.</p>

<h3 id="t1">Table 1 · Crisis regressions, all versions</h3>
</div>
{q2_table1()}
<p class="note">Dependent variable: monthly log return of the named index minus MSCI World (USD net), in %. Coefficients with Newey–West
t-statistics in brackets. "+ X" adds ΔVIX (points) and ΔDXY (%). Columns 5–6 use EM in local currency, still minus World in USD. Panel B moves
the start of the current period to 2022. * 10%, ** 5%, *** 1%.</p>

<div class="col">
<h3>What γ<sub>4</sub> says</h3>
<ul class="tight">
<li><b>No significant difference.</b> In the main version, γ<sub>4</sub> = 0.19 (t = 0.16). With the VIX and dollar controls it is 0.40
(t = 0.31). In a war or trade crisis month today, EM's relative return is within a fraction of a percentage point of what it was in past
episodes, and the estimate is far from significant. γ<sub>3</sub> for financial crises is just as close to zero (|t| &lt; 0.4 in every version).</li>
<li><b>Crisis months themselves are not special.</b> γ<sub>1</sub> and γ<sub>2</sub> are close to zero too. The one strong term is θ Post
(−1.3 to −1.9, t between −2.9 and −3.9): after the 2009 break, EM's return relative to World is about 1.5 percentage points a month lower than
before, in crisis and normal months alike.</li>
<li><b>From 2022 the sign is positive but still not significant.</b> With Current starting in 2022, γ<sub>4</sub> rises to 1.32 (t = 1.52)
and to 1.45 with controls (t = 1.63). For EM ex-IT it is only 0.54–0.66. Most of the better showing since 2022 comes through the tech sector.</li>
</ul>

<h3 id="t2">Table 2 · Event study, one row per crisis</h3>
<p>Cumulative daily EM-minus-World log returns from 5 trading days before to 20 days after each named crisis. Day 0 is the first trading
day on or after the event. The two decomposition columns split the 26-day result into a <b>currency part</b> (EM in USD minus EM in local currency;
negative means EM currencies fell against the dollar) and a <b>tech part</b> (EM minus EM ex-IT).</p>
</div>
{q2_table2()}
<p class="note">All values in log %. Local-ccy CAR = CAR minus the currency part. Daily EM ex-IT uses the IT weight at the previous month end.</p>

{figure("q2_event_paths", "Average cumulative EM minus World from day −5 to +20, past (before 2018) vs current (from 2018). Thick lines "
        "are group averages; thin lines are the individual crises. The right panel puts war and trade crises together, as Geo does in equation (2).")}

<div class="col">
<h3>Does the event study agree?</h3>
<ul class="tight">
<li><b>War and trade crises: yes.</b> Over 26 trading days EM lost 2.4% against World in past war crises. It lost 1.9% in current war and trade
crises (2.6% in wars, 1.5% in trade conflicts). The two paths in the right panel almost coincide by day +20, matching γ<sub>4</sub> ≈ 0.</li>
<li><b>Financial crises: only partly.</b> EM lost 4.4% in past financial crises but gained 1.1% in current ones. The event study suggests
EM now holds up better in the first weeks of a financial shock. The monthly regression (γ<sub>3</sub> ≈ 0) does not confirm this. There are only
two current events (COVID +3.9%, SVB −1.8%). Also, Fin in the regression covers whole drawdown periods, not just the first weeks.</li>
<li><b>The spread across crises is large.</b> Individual results range from −8.4% (Russia invades Ukraine) to +3.9% (COVID). The Ukraine result
includes Russia's removal from MSCI EM in March 2022, which is specific to that event.</li>
</ul>

<h3 id="drivers">What drives the difference: the dollar, currencies or tech?</h3>
</div>
{drivers_table()}
<p class="note">Average monthly values (in %, ΔVIX in points) in each set of flagged months; a month can be both Fin and Geo. Currency part = EM in USD minus EM in local
currency. Tech part = EM minus EM ex-IT. IT weight at the previous month end.</p>
<div class="col">
<ul class="tight">
<li><b>The dollar is the clearest driver.</b> In equation (2), φ<sub>ΔDXY</sub> = −0.43 (t = −3.9): a 1% rise in the dollar in a month
lowers EM's return relative to World by 0.43 percentage points. It is the only control that is strongly significant. Past Geo months
(2001–2005) came with a falling dollar (−0.66% a month). Current Geo months came with a rising one (+0.25%). Current Fin months also saw the
dollar rise (+0.34% vs −0.15% in the past). Holding the dollar fixed, γ<sub>4</sub> doubles from 0.19 to 0.40, though it stays insignificant.</li>
<li><b>It works through EM currencies.</b> In local currency the dollar coefficient disappears (0.01, t = 0.09). The currency part of EM's
return is −0.33 points a month in current Geo months, against +0.11 in past ones. In the event study, almost all of EM's past
financial-crisis underperformance is currency: −4.9 of the −4.4 total, while the local-currency result is +0.6. In local currency, γ<sub>4</sub> is
0.55 instead of 0.19: in local terms the gain over past crises is larger, and weaker currencies take part of it away.</li>
<li><b>Tech partly offsets the dollar.</b> EM ex-IT has lower γ<sub>4</sub> than EM in every version (−0.25 vs 0.19; 0.54 vs 1.32 from
2022). In current Geo months the tech part adds +0.69 points a month (IT weight 27% on average), against −0.10 in past ones. In the event study,
tech added +1.0 points in current wars but −0.6 in trade conflicts, where tariffs hit EM's tech exporters (Section 301 −1.0, Liberation Day −1.3).</li>
</ul>

<h3 id="limits">Limitations</h3>
<ul class="tight">
<li><b>Few current crises.</b> The current period has only 18 Fin months and 37 Geo months, and the event study has 2 current financial, 3 current war
and 4 trade events. Single events (COVID, Russia–Ukraine) swing the group averages. Standard errors are wide: the 95% interval for γ<sub>4</sub> runs from about −2.1 to +2.5 points a month.</li>
<li><b>Crisis type overlaps with time period.</b> All 16 past Geo months fall in 2001–2005 (9/11, the Iraq war). They are war shocks in a falling-dollar,
pre-tech-boom era. 31 of the 37 current Geo months are flagged by trade-policy uncertainty (TPU), and no past month is, so there is no past trade crisis to compare with.
γ<sub>4</sub> therefore mixes three things: the type of shock (war vs trade), the dollar regime, and index composition. The Post dummy (break in 2009) and
the rising IT weight also change with time.</li>
<li><b>The Fin flag is a drawdown state.</b> It is on for long recovery periods (Jan 2001–Jul 2005, Jun 2008–Dec 2012), so "past financial crisis
months" include years of EM recovery. That is one reason the monthly γ<sub>3</sub> and the event study differ.</li>
<li><b>Low explanatory power.</b> R² is 0.04–0.16. Crisis flags explain little of EM's monthly relative return; most of the variation is outside the model.</li>
</ul>

<h3>Answer to Question 2</h3>
<p class="verdict">EM's relative performance in today's war and trade crises is about the same as in past crises. Whether it does better or worse depends
mainly on the dollar: today's shocks have come with a stronger dollar and weaker EM currencies, and the larger tech weight only partly offsets this.</p>
</div>
</section>

<section id="portfolio">
<h2><span class="q">Step 4 · Questions 1 and 2</span>Portfolio evaluation</h2>
<div class="col">
<p>A US investor holds MSCI USA and adds one EM index. Minimum-variance weights are re-estimated each month from the previous 60 months, long-only,
and evaluated out of sample from January 2006 to September 2026 (249 months). Because EM was much more volatile than the US and highly correlated
with it, the minimum-variance weight on every EM index was 0% from 2006 to 2016. These portfolios are therefore identical to USA only through
the 2008 crisis, which is why they share the same maximum drawdown. As EM's beta and volatility fell, the weight rose to about 50% from 2022.</p>
</div>
{portfolio_table()}
<p class="note">Sharpe ratio uses the 3-month T-bill. No past Geo months fall in the out-of-sample period (all are 2001–2005), so only current
crisis-month averages are shown. EM Small Cap and EM (price) exclude dividends and understate returns; compare them with each other.</p>
{figure("step4_portfolios", "Growth of $1 and the weight in each EM index, out of sample 2006–2026.")}
<div class="col">
<p>Adding EM lowered volatility only slightly (EM ex-IT the most, to 14.7%) and lowered the Sharpe ratio in every case. In current Geo months the
USA + EM did slightly better than USA only (+0.80% vs +0.66% a month), while USA + EM ex-IT and USA + EM Small Cap did slightly worse. In current
financial-crisis months every portfolio with EM did worse than USA only.</p>
</div>
</section>

<section id="robust" class="col">
<h2>Robustness</h2>
<ul class="tight">
<li><b>EM Small Cap net returns, post-2010.</b> Results with the net index are almost identical to the price index: the mean DCC correlation
with the US is 0.602 in both. Equation (1) post-2010 (w<sub>IT</sub> version) is also nearly unchanged. The Post version cannot be estimated after
2010 because Post = 1 throughout. In portfolios from June 2015, USA + EM Small Cap has a Sharpe ratio of 0.69 (net) vs 0.67 (price), and USA only has 0.76.</li>
<li><b>Local-currency EM.</b> See Table 1, columns 5–6: γ<sub>3</sub> and γ<sub>4</sub> stay insignificant, and the dollar effect disappears.</li>
<li><b>Current period from 2022.</b> See Table 1, Panel B: γ<sub>4</sub> becomes positive (1.32, t = 1.52) but is not significant at 10%.</li>
</ul>
</section>

<footer>
Produced from <span class="mono">python run_all.py</span> and <span class="mono">python make_report.py</span> in the project repository.
Settings are in <span class="mono">analysis/config.py</span>; every table here has a CSV in <span class="mono">output/tables/</span>.
</footer>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"report written to {OUT / 'index.html'}")


if __name__ == "__main__":
    build()
