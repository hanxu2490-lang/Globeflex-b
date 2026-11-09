# -*- coding: utf-8 -*-
"""Build the results report (report/index.html) from the tables in output/.

Run `python run_all.py` first. Tables are filled from the result files; the written text describes
the results of the latest run (data to 31 Aug 2026).
"""
import html
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
T = ROOT / "output" / "tables"
F = ROOT / "output" / "figures"
OUT = ROOT / "report"
Q1 = ROOT / "output" / "q1"
FIGS = {"q2_event_paths": F / "q2_event_paths.png",
        "a_fig1_tech_share": Q1 / "module_A" / "fig1_tech_share.png",
        "b_weekly_dcc": Q1 / "module_B" / "B_06_Weekly_DCC_FullSample.png",
        "c_fig4_step_vs_blocks": Q1 / "module_C" / "figC4_step_vs_five_year_blocks_v2.png",
        "a3_fig2_sharpe_gain_5y": Q1 / "module_A3" / "fig2_sharpe_gain_5y.png"}


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
def q1_breaks_table():
    d = pd.read_excel(Q1 / "module_C" / "module_C_results_v2.xlsx", sheet_name="C1_breaks", index_col=[0, 1])
    rows = [[dep, b, int(r["breaks (sequential)"]), r["dates"], r["90% CI"], r["sup-F stats"]] for (dep, b), r in d.iterrows()]
    return table(["Dependent", "Benchmark", "Breaks", "Dates", "90% CI", "sup-F statistics"], rows, "num")


def q1_primary_table():
    d = pd.read_excel(Q1 / "module_C" / "module_C_results_v2.xlsx", sheet_name="C2_primary", index_col=0)
    rows = [[i, r["dependent"], r["benchmark"], num(r["β first"]), num(r["β last"]), num(r["δ = last − first"], sign=True),
             num(r["SE (NW)"]), num(r["p one-sided"], 3), num(r["p Holm"], 3)] for i, r in d.iterrows()]
    return table(["Test", "Dependent", "Benchmark", "β 2001-01…2009-10", "β 2009-11…2026-08", "δ", "SE (NW)",
                  "p one-sided", "p Holm"], rows, "num")


def q1_blocks_table():
    d = pd.read_excel(Q1 / "module_C" / "module_C_results_v2.xlsx", sheet_name="C2b_five_year_blocks", index_col=[0, 1, 2])
    rows = []
    for (b, dep), g in d.groupby(level=[0, 1], sort=False):
        rows.append([b, dep, *[f"{num(r['β'])} <span class='t'>({num(r['corr'])})</span>" for _, r in g.iterrows()]])
    blocks = list(d.index.get_level_values(2).unique())
    return table(["Benchmark", "Dependent", *blocks], rows, "num")


def q1_sharpe_table():
    d = pd.read_excel(Q1 / "module_C" / "module_C_results_v2.xlsx", sheet_name="C4c_sharpe_blocks", index_col=[0, 1])
    rows = [[a, blk, num(r["best EM %"], 0), num(r["Sharpe gain, free EM"], 3), num(r["Holm p free"], 3),
             num(r["Sharpe gain, EM = 10%"], 3, sign=True), num(r["Holm p 10%"], 3), num(r["GRS p"], 3)] for (a, blk), r in d.iterrows()]
    return table(["Asset", "Block", "Best EM %", "Sharpe gain, free", "Holm p", "Sharpe gain, 10% EM", "Holm p", "GRS p"], rows, "num")


def q1_oos_table():
    d = pd.read_excel(Q1 / "module_B" / "B_Module_Results.xlsx", sheet_name="OOS_Performance", index_col=0)
    rows = [[i, int(r["OOS months"]), num(r["Annualized return %"]), num(r["Annualized volatility %"]), num(r["Sharpe"]),
             num(r["Max drawdown %"])] for i, r in d.iterrows()]
    return table(["Portfolio (60-month lookback)", "Months", "Return %/yr", "Volatility %/yr", "Sharpe", "Max drawdown %"], rows, "num")


def q2_table1(panels, headers):
    """Crisis regressions; panels = which panel letters to show, headers = {letter: panel heading}."""
    d = pd.read_csv(T / "q2_table1_formatted.csv", index_col=0, encoding="utf-8-sig")
    cols = [c for c in d.columns if d.loc["panel", c] in panels]
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
        if key == "θ Post":
            cells = [c if c != "" else "<span class='t'>not included</span>" for c in cells]
        rows.append([f'<span class="rl">{lab}</span>', *cells])
    th_top = '<tr class="panel"><th></th>' + "".join(
        f'<th colspan="{sum(d.loc["panel", c] == p for c in cols)}">{headers[p]}</th>' for p in panels) + "</tr>"
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


def placebo_table():
    d = pd.read_csv(T / "step3_placebo_summary.csv", index_col=0, encoding="utf-8-sig")
    rows = [[k, int(r["n events"]), num(r["observed mean"], sign=True), num(r["placebo mean"], sign=True),
             num(r["difference"], sign=True), num(r["p two-sided"], 3),
             f'[{num(r["placebo 2.5%"], sign=True)}, {num(r["placebo 97.5%"], sign=True)}]'] for k, r in d.iterrows()]
    return table(["Events", "n", "Observed mean", "Placebo mean", "Difference", "p (two-sided)", "Placebo 95% range"], rows, "num")


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
table.t2 thead th:not(:first-child),table.t2 thead th{white-space:normal;min-width:64px}
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
    for name, src in FIGS.items():
        shutil.copy(src, OUT / "figures" / f"{name}.png")

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
  <p class="lede">Team results on MSCI data from January 2001 to August 2026, with the focus on
  Question 2: does EM do better relative to developed markets in today's war and trade crises than in past crises?</p>
  <div class="meta">Jun Wang, Yujia Chen, Xu Han, Yaqi Xiao, Luqi Yang · Question 1 and Question 2: 2001-01 to 2026-08</div>
  <nav class="toc" aria-label="Sections">
    <a href="#answers">Answers</a><a href="#data">Data</a><a href="#q1">Question 1</a>
    <a href="#q2">Question 2</a><a href="#t1">Table 1</a><a href="#t2">Table 2</a><a href="#drivers">Drivers</a>
    <a href="#limits">Limitations</a><a href="#portfolio">Optimizer</a><a href="#robust">Robustness</a>
  </nav>
</header>

<section id="answers">
<div class="answers">
  <div class="answer main"><h4>Question 2 · crises</h4>
  <p>EM's relative performance in today's war and trade crises is <b>not significantly different</b> from past crises
  (γ<sub>4</sub> = −0.08, t = −0.07). What moves it is mainly <b>the dollar</b>. Recent shocks come with a stronger dollar
  and weaker EM currencies, which costs EM about as much as its larger tech weight gains.</p></div>
  <div class="answer"><h4>Question 1 · co-movement</h4>
  <p>EM's link to the U.S. <b>changed once</b>, in a post-GFC transition (point estimate Oct 2009). After it, non-tech EM's
  sensitivity to the U.S. <b>fell</b> (β 1.27 → 0.80, Holm p = 0.008), while EM's link to U.S. tech shows no significant change.
  In 2021–26, Taiwan and Korea carry much of EM's link to U.S. tech.</p></div>
  <div class="answer"><h4>Optimizer · Modules A3 and B</h4>
  <p>EM's gain is significant only in the <b>early 2000s</b>. Since then the optimizer gives EM <b>0%</b>, including EM ex-IT today,
  and adding EM does not reliably help out of sample. The optimizer's country gains cannot be told apart from chance.</p></div>
</div>
</section>

<section id="data" class="col">
<h2>Data and design</h2>
<ul class="tight">
<li>MSCI USD net total-return indices for EM, USA, World and USA IT. EM Small Cap uses the price index, so it is compared with the EM
price index on the same basis. A local-currency EM index tests the currency channel.</li>
<li>EM ex-IT is backed out from EM and EM IT returns with the IT weight w<sub>IT</sub> = EM IT / EM market value at the previous month end.</li>
<li>Crisis months (Question 2). <b>Fin</b> = MSCI World more than 10% below its running peak (134 months). <b>Geo</b> = GPR or TPU in its
top decile over 2001–2026, i.e. GPR ≥ 158.0 or TPU ≥ 165.2 (53 months). <b>Current</b> = from January 2018.</li>
<li>Question 1 (Modules A, A3, B, C): monthly data 2001-01 to 2026-08. Series are built in simple returns with last month's weights
(EM ex-IT, EM ex-Taiwan &amp; Korea) and converted to log returns for correlations, betas and tests. The optimizer uses the S&amp;P 500 total
return, the Bloomberg US Aggregate and MSCI country indices for the eight largest EM markets.</li>
<li>Question 2: monthly data 2001-01 to 2026-08 (daily data to 31 Aug 2026), log returns in %, Newey–West standard errors (5 lags).</li>
</ul>
</section>
{figure("a_fig1_tech_share", "Module A, Fig 1. IT share of MSCI EM. The Dec 2018 drop is the GICS reclassification that moved internet firms out of IT; "
       "the dashed line rescales the earlier values to the post-2018 definition (approximation).")}

<section id="q1">
<h2><span class="q">Question 1 · Modules A, B and C</span>Is EM more like the US, and is technology the cause?</h2>
{figure("b_weekly_dcc", "Module B. Full-sample weekly DCC-GARCH correlations (log returns; EM Small Cap price index).")}
<div class="col">
<ul class="tight">
<li><b>EM's link to the U.S. changed, and this is well supported.</b> The sequential, UDmax and WDmax tests all reject no break, in all 12
re-searches, with or without the GFC. BIC picks one break; LWZ, the most conservative, picks none.</li>
<li><b>The date is a range.</b> Oct 2009 is the joint model's point estimate (90% CI 2008-07…2011-01), inside a post-GFC transition.
α moved first and β later (exploratory).</li>
<li><b>Non-tech EM's sensitivity to the U.S. fell</b> (β 1.27 → 0.80, Holm p = 0.008), for break dates from 2009-10 to 2012-12 but not at 2007-10.</li>
<li><b>EM's link to U.S. tech shows no significant change between regimes</b>, and EM IT stays strongly linked (β ≈ 0.9).
Whether non-tech fell more than tech, and whether Taiwan &amp; Korea raised EM's sensitivity, is not established after Holm.</li>
<li><b>Correlations fell for every index</b>, but no single change is significant. A lower β is not the same as a lower correlation.</li>
<li><b>Taiwan and Korea in 2021–26.</b> Removing both lowers EM's correlation with U.S. tech by 0.25 (Holm p = 0.004), against 0.02–0.03 in every
earlier block (Module A tests).</li>
</ul>
</div>
<div class="col"><h3>Structural breaks (Module C)</h3></div>
{q1_breaks_table()}
<div class="col"><h3>Primary tests H1–H3 (Module C)</h3></div>
{q1_primary_table()}
<p class="note">Regime regression with Newey–West SEs; δ = β(last regime) − β(first regime); one-sided p-values, Holm over the six tests;
conditional on the break date. H1: EM ex-IT; H2: spread EM IT − EM ex-IT; H3: spread EM − EM ex-Taiwan &amp; Korea.</p>
<div class="col"><h3>β on the U.S. by five-year block (correlation in brackets)</h3></div>
{q1_blocks_table()}
{figure("c_fig4_step_vs_blocks", "Module C, Fig C4. β of EM and EM ex-IT on the U.S. market: step model at the break vs Module A's five-year blocks.")}
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
17 past and 36 current Geo months.</p>

<h3 id="t1">Table 1 · Crisis regressions, with and without Post</h3>
</div>
{q2_table1(["A", "B"], {"A": "Panel A · with Post (break Oct 2009)", "B": "Panel B · without Post"})}
<p class="note">Dependent variable: monthly log return of the named index minus MSCI World (USD net), in %. Coefficients with Newey–West
t-statistics in brackets. "+ X" adds ΔVIX (points) and ΔDXY (%). Current = 1 from January 2018. Panel B drops θ Post_t and keeps everything else.
* 10%, ** 5%, *** 1%.</p>

<div class="col"><h3>Table 1b · Robustness (with Post)</h3></div>
{q2_table1(["C"], {"C": "EM in local currency (minus World in USD), and Current = 1 from January 2022"})}

<div class="col">
<h3>What γ<sub>4</sub> says</h3>
<ul class="tight">
<li><b>No significant difference.</b> In the main version, γ<sub>4</sub> = −0.08 (t = −0.07). With the VIX and dollar controls it is 0.19
(t = 0.15). In a war or trade crisis month today, EM's relative return is within a fraction of a percentage point of what it was in past
episodes, and the estimate is far from significant. γ<sub>3</sub> for financial crises is just as close to zero (|t| &lt; 0.4 in every version).</li>
<li><b>Crisis months themselves are not special.</b> γ<sub>1</sub> and γ<sub>2</sub> are close to zero too. The one strong term is θ Post
(−1.3 to −1.8, t between −2.9 and −3.6): after the 2009 break, EM's return relative to World is about 1.5 percentage points a month lower than
before, in crisis and normal months alike.</li>
<li><b>Without Post the answer is the same.</b> Dropping θ Post moves EM's stronger pre-2009 returns into the crisis terms:
γ<sub>1</sub> and γ<sub>2</sub> turn positive (0.54 and 0.85) and γ<sub>4</sub> becomes more negative (−0.93, t = −0.86; −1.43, t = −1.62 for EM ex-IT).
None of the crisis terms is significant (all |t| &lt; 1.7), and R² falls from 0.04 to 0.01, so Post carries most of what the model explains.
The dollar effect is unchanged (φ<sub>ΔDXY</sub> = −0.45, t = −4.2).</li>
<li><b>From 2022 the sign is positive but still not significant.</b> With Current starting in 2022, γ<sub>4</sub> rises to 1.12 (t = 1.27)
and to 1.28 with controls (t = 1.44). For EM ex-IT it is only 0.47–0.63. Most of the better showing since 2022 comes through the tech sector.</li>
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

<h3 id="placebo">Placebo test · crisis vs non-crisis dates (Module E)</h3>
<p>The mean CAR[−5,+20] of the named crises is compared with 5,000 placebo sets of non-crisis dates. Each set draws as many past and
current dates as there are events, without replacement, from months with no Fin, GPR or TPU flag and outside ±25 trading days of every
named event (1,802 past and 1,003 current dates).</p>
</div>
{placebo_table()}
<p class="note">Mean cumulative EM − World log return from Day −5 to Day +20, in percentage points. Two-sided empirical p-value centred on
the placebo mean. Core 12 leaves out China devaluation, Crimea and COVID; excluding Crimea leaves 14. A descriptive randomization benchmark,
not a causal test and not a test of the difference between past and current crises.</p>
<div class="col">
<ul class="tight">
<li><b>All 15 crises:</b> EM lost 2.11 points against World, against −0.24 on placebo dates (p = 0.044). For the 12 core events the
loss is 2.85 (p = 0.011), and 2.45 without Crimea (p = 0.023).</li>
<li><b>Past crises:</b> −3.36 against +0.08 on placebo dates (p = 0.026). <b>Current crises:</b> −1.27 against −0.50 (p = 0.49).</li>
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
(mostly 2001–2005) came with a falling dollar (−0.78% a month). Current Geo months came with a rising one (+0.21%). Current Fin months also saw the
dollar rise (+0.34% vs −0.15% in the past). Holding the dollar fixed, γ<sub>4</sub> rises from −0.08 to 0.19, though it stays insignificant.</li>
<li><b>It works through EM currencies.</b> In local currency the dollar coefficient disappears (0.01, t = 0.09). The currency part of EM's
return is −0.34 points a month in current Geo months, against +0.18 in past ones. In the event study, almost all of EM's past
financial-crisis underperformance is currency: −4.9 of the −4.4 total, while the local-currency result is +0.6. In local currency, γ<sub>4</sub> is
0.38 instead of −0.08: in local terms the gain over past crises is larger, and weaker currencies take part of it away.</li>
<li><b>Tech partly offsets the dollar.</b> EM ex-IT has lower γ<sub>4</sub> than EM in every version (−0.39 vs −0.08; 0.47 vs 1.12 from
2022). In current Geo months the tech part adds +0.61 points a month (IT weight 26% on average), against −0.04 in past ones. In the event study,
tech added +1.0 points in current wars but −0.6 in trade conflicts, where tariffs hit EM's tech exporters (Section 301 −1.0, Liberation Day −1.3).</li>
</ul>

<h3 id="limits">Limitations</h3>
<ul class="tight">
<li><b>Few current crises.</b> The current period has only 18 Fin months and 36 Geo months, and the event study has 2 current financial, 3 current war
and 4 trade events. Single events (COVID, Russia–Ukraine) swing the group averages. Standard errors are wide: the 95% interval for γ<sub>4</sub> runs from about −2.3 to +2.2 points a month.</li>
<li><b>Crisis type overlaps with time period.</b> 16 of the 17 past Geo months fall in 2001–2005 (9/11, the Iraq war). They are war shocks in a falling-dollar,
pre-tech-boom era. 30 of the 36 current Geo months are flagged by trade-policy uncertainty (TPU), and only one past month is (January 2017), so there is almost no past trade crisis to compare with.
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
<h2><span class="q">Modules A3, B and C · Questions 1 and 2</span>Mean-variance optimizer: does EM still earn a place?</h2>
<div class="col">
<p>A U.S. investor holds the S&amp;P 500 and the U.S. Aggregate. The optimizer adds MSCI EM or EM ex-IT at fixed shares of 0–100%
(Module A3, rolling 5- and 10-year windows ending each August), lets eight EM countries enter freely (A3, from 2015-04), and runs the same
optimizer out of sample with a 60-month lookback (Module B). Module C tests whether the gains are beyond chance.</p>
</div>
{figure("a3_fig2_sharpe_gain_5y", "Module A3, Fig 2. Sharpe gain over the stock-bond portfolio for each EM share, 5-year windows by window end.")}
<div class="col">
<ul class="tight">
<li><b>EM's gain is significant only in the early 2000s.</b> In 2001–05 the gain is 0.43 (GRS p = 0.005, Holm 0.036), and even a fixed 10%
adds 0.38.</li>
<li><b>Since then the optimizer gives EM 0%</b> in every 5-year window from 2007-09…2012-08 onward, including EM ex-IT today. A forced 10% costs
little: −0.024 for EM and −0.046 for EM ex-IT in 2021–26.</li>
<li><b>Out of sample</b>, adding broad EM does not reliably improve the stock-bond portfolio.</li>
<li><b>Countries.</b> The optimizer's country gains are not distinguishable from chance in any window, and the choice is unstable across
bootstrap samples (Module A tests, O2).</li>
</ul>
</div>
<div class="col"><h3>Is the Sharpe gain beyond chance? A's five-year blocks (Module C)</h3></div>
{q1_sharpe_table()}
<div class="col"><h3>Out-of-sample backtest (Module B)</h3></div>
{q1_oos_table()}
</section>

<section id="robust" class="col">
<h2>Robustness</h2>
<ul class="tight">
<li><b>Local-currency EM.</b> See Table 1b, columns 9–10: γ<sub>3</sub> and γ<sub>4</sub> stay insignificant, and the dollar effect disappears.</li>
<li><b>Current period from 2022.</b> See Table 1b, columns 11–14: γ<sub>4</sub> becomes positive (1.12, t = 1.27) but is not significant at 10%.</li>
</ul>
</section>

<footer>
Produced from <span class="mono">python run_all.py</span> and <span class="mono">python make_report.py</span> in the project repository.
Question 1 results are in <span class="mono">output/q1/</span> (one folder per module); Question 2 tables are in <span class="mono">output/tables/</span>.
</footer>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"report written to {OUT / 'index.html'}")


if __name__ == "__main__":
    build()
