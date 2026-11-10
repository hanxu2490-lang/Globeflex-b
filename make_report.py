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
T = ROOT / "output" / "q2"
F = ROOT / "output" / "q2"
OUT = ROOT / "report"
Q1 = ROOT / "output" / "q1"
FIGS = {"q2_event_paths": F / "q2_event_paths.png",
        "a_fig1_tech_share": Q1 / "module_A" / "fig1_tech_share.png",
        "b_weekly_dcc": Q1 / "module_B" / "B_07_Weekly_DCC.png",
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
    f = Q1 / "module_B" / "B_Module_Results.xlsx"
    d = pd.read_excel(f, sheet_name="OOS_All_Periods")
    d = d[d["Block"] == "All available OOS"]
    order = [f"{e} | {k} | {o}" for e, ks in [("Benchmark", [None]), ("MSCI EM", ["Free", "Fixed 10%", "Fixed 20%"]),
                                               ("EM ex-IT", ["Free", "Fixed 10%", "Fixed 20%"])] for k in ks for o in ["sharpe", "minvar"]]
    order = [s.replace(" | None", "") for s in order]
    cells = lambda r: [int(r["N months"]), num(r["Annual Return %"]), num(r["Annual Volatility %"]), num(r["Sharpe"]),
                       num(r["Max Drawdown %"])]
    rows = []
    for lb in (60, 120):
        g = d[d["Lookback"] == lb].set_index("Strategy")
        rows.append(f"{lb}-month estimation window, first OOS month {g['First month'].min():%Y-%m}")
        rows += [[s.replace("sharpe", "max Sharpe").replace("minvar", "min variance"), *cells(g.loc[s])] for s in order]
    c = pd.read_excel(f, sheet_name="OOS_Country_Joint").set_index("Objective")
    rows.append("S&amp;P 500 + U.S. Aggregate + eight countries, 60-month window, 2020-04 to 2026-08")
    rows += [[f"Countries | {'max Sharpe' if o == 'sharpe' else 'min variance'}", *cells(c.loc[o])] for o in ["sharpe", "minvar"]]
    return table(["Portfolio", "Months", "Return %/yr", "Volatility %/yr", "Sharpe", "Max drawdown %"], rows, "num")


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


SHORT_STATUS = {"Core event (rule met)": "Core", "Retained - shock-triggered": "Shock-triggered",
                "Additional case - rule not met": "Additional case"}
GROUPS = {"Past Fin": "Past financial crises (before 2018)", "Current Fin": "Current financial crises (from 2018)",
          "Past Geo": "Past war crises (before 2018)", "Current Geo": "Current war crises (from 2018)",
          "Current Trade": "Current trade conflicts (from 2018)"}


def selection_table():
    d = pd.read_csv(T / "step3_crisis_selection.csv", index_col=0, encoding="utf-8-sig")
    rows = []
    for e, r in d.iterrows():
        fin = r["type"] == "Fin"
        val = f"{100 * r['test value (event month)']:.1f}%" if fin else f"{r['test value (event month)']:.1f}"
        thr = f"{100 * r['threshold']:.0f}%" if fin else f"{r['threshold']:.1f}"
        rows.append([html.escape(e), r["event date"], r["type"], r["period"], r["test"], val.replace("-", "−"),
                     thr.replace("-", "−"), r["result"], SHORT_STATUS[r["status"]]])
    return table(["Crisis", "Date", "Type", "Period", "Test (event month)", "Value", "Threshold", "Result", "Status"], rows, "num")


def event_table():
    d = pd.read_csv(T / "step3_event_study.csv", index_col=0, encoding="utf-8-sig")
    g = pd.read_csv(T / "step3_event_study_by_group.csv", index_col=0, encoding="utf-8-sig")
    em, wd = "EM cumulative [-5,+20]", "World cumulative [-5,+20]"
    rows = []
    for grp, title in GROUPS.items():
        rows.append(title)
        for e, r in d[(d["period"] + " " + d["type"]) == grp].iterrows():
            rows.append([html.escape(e), SHORT_STATUS[r["status"]], r["day 0"], num(r["CAR[-5,+0]"], sign=True),
                         num(r["CAR[-5,+5]"], sign=True), num(r[em], sign=True), num(r[wd], sign=True),
                         f"<b>{num(r['CAR[-5,+20]'], sign=True)}</b>"])
        a = g.loc[grp]
        rows.append([f"<i>Average of {int(a['n events'])}</i>", "", "", num(a["CAR[-5,+0]"], sign=True),
                     num(a["CAR[-5,+5]"], sign=True), num(a[em], sign=True), num(a[wd], sign=True),
                     f"<b>{num(a['CAR[-5,+20]'], sign=True)}</b>"])
    t = table(["Crisis", "Status", "Day 0", "EM − World [−5, 0]", "EM − World [−5, +5]", "EM [−5, +20]", "World [−5, +20]",
               "EM − World [−5, +20]"], rows, "num t2")
    return t.replace("<tr><td><i>Average", '<tr class="avg"><td><i>Average')


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
  <p class="lede">Team results on MSCI data from January 2001 to August 2026. Question 1: has EM become more like the U.S., and is
  technology the cause? Question 2: does EM do better relative to developed markets in today's crises than in past ones?</p>
  <div class="meta">Jun Wang, Yujia Chen, Xu Han, Yaqi Xiao, Luqi Yang · monthly data 2001-01 to 2026-08, daily data to 31 Aug 2026</div>
  <nav class="toc" aria-label="Sections">
    <a href="#answers">Answers</a><a href="#data">Data</a><a href="#modA">Module A</a><a href="#modB">Module B</a>
    <a href="#modC">Module C</a><a href="#portfolio">Optimizer</a><a href="#selection">Crisis selection</a>
    <a href="#events">Event study</a><a href="#placebo">Placebo test</a><a href="#regression">Crisis regression</a>
  </nav>
</header>

<section id="answers">
<div class="answers">
  <div class="answer"><h4>Question 1 · co-movement</h4>
  <p>EM's link to the U.S. <b>changed once</b>, in a post-GFC transition (point estimate Oct 2009). After it, non-tech EM's
  sensitivity to the U.S. <b>fell</b> (β 1.27 → 0.80, Holm p = 0.008), while EM's link to U.S. tech shows no significant change.
  In 2021–26, Taiwan and Korea carry much of EM's link to U.S. tech.</p></div>
  <div class="answer"><h4>Optimizer · Modules A3 and B</h4>
  <p>EM's gain is significant only in the <b>early 2000s</b>. Since then the optimizer gives EM <b>0%</b>, including EM ex-IT today,
  and adding EM does not reliably help out of sample. The optimizer's country gains cannot be told apart from chance.</p></div>
  <div class="answer"><h4>Question 2 · crises</h4>
  <p>Around the 15 named crises EM lost on average <b>2.11 points</b> against World over 26 trading days, more than on non-crisis dates
  (p = 0.044). The loss was larger in past crises (−3.36, p = 0.026) than in current ones (−1.27, p = 0.49). In the monthly regression,
  crisis months today are <b>not significantly different</b> from past ones.</p></div>
</div>
</section>

<section id="data" class="col">
<h2>Data and design</h2>
<ul class="tight">
<li>MSCI USD net total-return indices for EM, USA, World and USA IT; MSCI country indices for the eight largest EM markets; EM Small Cap
(price and net); a local-currency EM index; the S&amp;P 500 total return and the Bloomberg US Aggregate for the optimizer.</li>
<li>Question 1 (Modules A, A3, A tests, B, C): month-end data 2001-01 to 2026-08. Series are built in simple returns with last month's weights
(EM ex-IT, EM ex-Taiwan &amp; Korea) and converted to log returns for correlations, betas and tests. Module B also uses weekly returns.</li>
<li>Question 2: monthly log returns in % for the crisis regression and daily log returns for the event study. Crisis months: <b>Fin</b> =
MSCI World more than 10% below its running peak (134 months); <b>Geo</b> = GPR or TPU in its top decile over 2001–2026, i.e. GPR ≥ 158.0 or
TPU ≥ 165.2 (53 months). <b>Current</b> = from January 2018.</li>
</ul>
</section>

<section id="modA">
<h2><span class="q">Question 1 · Module A and Module A tests</span>EM's composition and its link to the U.S.</h2>
{figure("a_fig1_tech_share", "Module A, Fig 1. IT share of MSCI EM. The Dec 2018 drop is the GICS reclassification that moved internet firms out of IT; "
       "the dashed line rescales the earlier values to the post-2018 definition (approximation).")}
<div class="col">
<ul class="tight">
<li><b>Composition.</b> IT is 41.6% of MSCI EM in Aug 2026, against 20.9% at the end of 2001. Taiwan (27.4%) and Korea (20.8%) are now
the two largest markets, ahead of China (20.6%).</li>
<li><b>IT risk.</b> In 2021–26 IT carries 35.6% of EM's risk with a 23.6% weight, the largest gap of any block.</li>
<li><b>Link to the U.S.</b> EM's correlation with the U.S. market falls from 0.80 in 2001–05 to 0.62 in 2021–26, and its β from 1.12 to 0.72.
In the trend regression, EM's β on the U.S. falls by 0.25 per decade (p = 0.016), and EM ex-IT's by 0.32 (p = 0.005). EM's β on U.S.
tech shows no trend (p = 0.97).</li>
<li><b>Taiwan and Korea (Module A tests).</b> In 2021–26, removing both lowers EM's correlation with U.S. tech by 0.25 (Holm p = 0.004),
against 0.02–0.03 in every earlier block; removing China raises it by 0.10 (Holm p = 0.028). The gap in correlation between EM and EM ex-IT
(0.20) and the excess of IT's risk share over its weight (0.12) are also significant after Holm.</li>
</ul>
</div>
</section>

<section id="modB">
<h2><span class="q">Question 1 · Module B</span>Time-varying correlations (DCC-GARCH)</h2>
{figure("b_weekly_dcc", "Module B. Full-sample weekly DCC-GARCH correlations (log returns; EM Small Cap price index).")}
<div class="col">
<p>Each series gets an AR(2)-GARCH-t or AR(2)-GJR-GARCH-t model (weekly U.S. IT also tries EGARCH, skewed-t and ARCH(10) versions),
chosen by the lowest BIC among the models that pass all three residual diagnostics; the pairwise DCC has its persistence capped at 0.998.
Weekly U.S. IT is the one series where no model passes all three diagnostics, and EM Small Cap vs U.S. IT ends at the cap
(a + b = 0.998), so that pair is boundary-sensitive. The weekly conditional correlation of EM with the U.S. averages 0.62 in 2003,
0.68 in 2020, 0.57 in 2022 and 0.60 in 2026; with U.S. tech it averages 0.58, 0.65, 0.49 and 0.53 in the same years.</p>
</div>
</section>

<section id="modC">
<h2><span class="q">Question 1 · Module C</span>Structural break and primary tests</h2>
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
</ul>
</div>
<div class="col"><h3>Structural breaks</h3></div>
{q1_breaks_table()}
<div class="col"><h3>Primary tests H1–H3</h3></div>
{q1_primary_table()}
<p class="note">Regime regression with Newey–West SEs; δ = β(last regime) − β(first regime); one-sided p-values, Holm over the six tests;
conditional on the break date. H1: EM ex-IT; H2: spread EM IT − EM ex-IT; H3: spread EM − EM ex-Taiwan &amp; Korea.</p>
<div class="col"><h3>β on the U.S. by five-year block (correlation in brackets)</h3></div>
{q1_blocks_table()}
{figure("c_fig4_step_vs_blocks", "Module C, Fig C4. β of EM and EM ex-IT on the U.S. market: step model at the break vs Module A's five-year blocks.")}
</section>

<section id="portfolio">
<h2><span class="q">Question 1 · Modules A3, B and C</span>Mean-variance optimizer: does EM still earn a place?</h2>
<div class="col">
<p>A U.S. investor holds the S&amp;P 500 and the U.S. Aggregate. The optimizer adds MSCI EM or EM ex-IT at fixed shares of 0–100%
(Module A3, rolling 5- and 10-year windows ending each August), lets eight EM countries enter freely (A3, from 2015-04), and runs the same
optimizer out of sample (Module B): each month, weights from the previous 60 or 120 months are applied to the next month, with EM free,
fixed at 10% or 20%, or the eight countries chosen jointly. Module C tests whether the in-sample gains are beyond chance.</p>
</div>
{figure("a3_fig2_sharpe_gain_5y", "Module A3, Fig 2. Sharpe gain over the stock-bond portfolio for each EM share, 5-year windows by window end.")}
<div class="col">
<ul class="tight">
<li><b>EM's gain is significant only in the early 2000s.</b> In 2001–05 the gain is 0.43 (GRS p = 0.005, Holm 0.036), and even a fixed 10%
adds 0.38.</li>
<li><b>Since then the optimizer gives EM 0%</b> in every 5-year window from 2007-09…2012-08 onward, including EM ex-IT today. A forced 10% costs
little: −0.024 for EM and −0.046 for EM ex-IT in 2021–26.</li>
<li><b>Out of sample</b> the result depends on the window. With 60 months, a fixed 10% or 20% MSCI EM raises the Sharpe ratio a little
(max Sharpe 0.59 → 0.62 and 0.61; min variance 0.37 → 0.42 and 0.43). With 120 months it lowers the max-Sharpe portfolio (0.70 → 0.60 and 0.59).
Free-weight EM is at or below the stock-bond portfolio in every case. The country portfolio (2020-04 to 2026-08) has a Sharpe ratio of 0.41
(max Sharpe) and −0.31 (min variance). All optimizations converged.</li>
<li><b>Countries.</b> The optimizer's country gains are not distinguishable from chance in any window, and the choice is unstable across
bootstrap samples (Module A tests, O2).</li>
</ul>
</div>
<div class="col"><h3>Is the Sharpe gain beyond chance? A's five-year blocks (Module C)</h3></div>
{q1_sharpe_table()}
<div class="col"><h3>Out-of-sample tests (Module B)</h3></div>
{q1_oos_table()}
</section>

<section id="selection">
<h2><span class="q">Question 2 · Module E</span>Crisis selection</h2>
<div class="col">
<p>Each of the 15 named crises is checked against the crisis rule in its event month. Financial: MSCI World at least 10% below its
running peak at the month end; if the 10% level is reached only between Day 0 and Day +20 (daily data), the event is kept as
shock-triggered. War: GPR in its top decile. Trade: TPU in its top decile.</p>
</div>
{selection_table()}
<p class="note">Core = rule met in the event month (12 events). Shock-triggered = the 10% drawdown is reached only after the event
(China devaluation, COVID). Additional case = rule not met (Crimea annexation: GPR 119.8 against 158.0); it is kept as a third past
war event and is labelled as such.</p>
</section>

<section id="events">
<h2><span class="q">Question 2 · Module E</span>Event study, Day −5 to Day +20</h2>
<div class="col">
<p>Daily log returns of MSCI EM and MSCI World, summed from 5 trading days before to 20 trading days after each crisis. Day 0 is the
first trading day on or after the event date.</p>
</div>
{event_table()}
<p class="note">Cumulative log returns in % (percentage points). EM − World [−5, k] is the sum of daily EM minus World returns from
Day −5 to Day k.</p>
{figure("q2_event_paths", "Average cumulative EM minus World from day −5 to +20, past (before 2018) vs current (from 2018). Thick lines "
        "are group averages; thin lines are the individual crises. The right panel puts war and trade crises together, as Geo does in equation (2).")}
<div class="col">
<ul class="tight">
<li><b>EM usually lagged World.</b> In 11 of the 15 crises EM − World is negative at Day +20; the average is −2.11 points.</li>
<li><b>By group.</b> Past financial crises −4.35, current financial crises +1.07 (COVID +3.92, SVB −1.77); past war crises −2.38,
current war crises −2.57; current trade conflicts −1.46.</li>
<li><b>The spread across crises is large</b>, from −8.38 (Russia invades Ukraine) to +3.92 (COVID).</li>
</ul>
</div>
</section>

<section id="placebo">
<h2><span class="q">Question 2 · Module E</span>Placebo test: crisis vs non-crisis dates</h2>
<div class="col">
<p>The mean EM − World [−5, +20] of the named crises is compared with 5,000 placebo sets of non-crisis dates. Each set draws as many past and
current dates as there are events, without replacement, from months with no Fin, GPR or TPU flag and outside ±25 trading days of every
named event (1,802 past and 1,003 current dates).</p>
</div>
{placebo_table()}
<p class="note">Mean cumulative EM − World log return from Day −5 to Day +20, in percentage points. Two-sided empirical p-value centred on
the placebo mean. Core 12 = the events that meet the rule in the event month. A descriptive randomization benchmark, not a causal test and
not a test of the difference between past and current crises.</p>
<div class="col">
<ul class="tight">
<li><b>All 15 crises:</b> −2.11 against −0.24 on placebo dates (p = 0.044). Core 12: −2.85 (p = 0.011). Without Crimea: −2.45 (p = 0.023).</li>
<li><b>Past crises:</b> −3.36 against +0.08 (p = 0.026). <b>Current crises:</b> −1.27 against −0.50 (p = 0.49).</li>
</ul>
</div>
</section>

<section id="regression">
<h2><span class="q">Question 2 · crisis regression</span>Monthly EM minus World in crisis months</h2>
<div class="eq">r<sub>EM,t</sub> − r<sub>World,t</sub> = α + γ<sub>1</sub>Fin<sub>t</sub> + γ<sub>2</sub>Geo<sub>t</sub> + γ<sub>3</sub>(Fin<sub>t</sub>×Current<sub>t</sub>) + γ<sub>4</sub>(Geo<sub>t</sub>×Current<sub>t</sub>) + θPost<sub>t</sub> + φ′X<sub>t</sub> + ε<sub>t</sub></div>
<div class="col">
<p>γ<sub>3</sub> and γ<sub>4</sub> measure how EM's monthly return relative to World in a financial or war / trade crisis month today
differs from the past. Post = 1 after Module C's break (Oct 2009). X<sub>t</sub> = monthly changes in the VIX and the dollar index (DXY).
308 months: 116 past and 18 current Fin months, 17 past and 36 current Geo months.</p>
</div>
{q2_table1(["A", "B"], {"A": "Panel A · with Post (break Oct 2009)", "B": "Panel B · without Post"})}
<p class="note">Dependent variable: monthly log return of the named index minus MSCI World (USD net), in %. Coefficients with Newey–West
t-statistics (5 lags) in brackets. "+ X" adds ΔVIX (points) and ΔDXY (%). * 10%, ** 5%, *** 1%.</p>
<div class="col"><h3>Robustness: EM in local currency, and Current from January 2022</h3></div>
{q2_table1(["C"], {"C": "Robustness (with Post)"})}
<div class="col">
<ul class="tight">
<li><b>No crisis term is significant</b> in any version. γ<sub>4</sub> = −0.08 (t = −0.07), and 0.19 (t = 0.15) with ΔVIX and ΔDXY;
γ<sub>3</sub> is close to zero (|t| &lt; 0.4).</li>
<li><b>θ Post</b> is −1.3 to −1.8 (t between −2.9 and −3.6): after the break, EM's monthly return relative to World is lower in crisis
and normal months alike. <b>φ ΔDXY</b> = −0.43 (t = −3.9).</li>
<li><b>Without Post</b>, γ<sub>1</sub> and γ<sub>2</sub> turn positive (0.54 and 0.85) and γ<sub>4</sub> is −0.93 (t = −0.86); none is significant.</li>
<li><b>Robustness.</b> In local currency γ<sub>4</sub> is 0.38 (t = 0.32) and the dollar coefficient is 0.01 (t = 0.09). With Current from
2022, γ<sub>4</sub> is 1.12 (t = 1.27).</li>
</ul>
</div>
</section>

<footer>
Produced from <span class="mono">python run_all.py</span> and <span class="mono">python make_report.py</span> in the project repository.
Question 1 results are in <span class="mono">output/q1/</span> (one folder per module); Question 2 tables and figures are in <span class="mono">output/q2/</span>.
</footer>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"report written to {OUT / 'index.html'}")


if __name__ == "__main__":
    build()
