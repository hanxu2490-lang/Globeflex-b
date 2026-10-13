"""Shared helpers: OLS with Newey-West errors, table output, chart style."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C

# chart colours (fixed order) and ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SHADE = "#f0efec"


def nw_lags(T):
    """Newey-West lag length, floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def ols_nw(y, X):
    """OLS of y on X (a DataFrame, constant added) with Newey-West standard errors."""
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})


def lincom(res, names):
    """Sum of coefficients `names`, with its Newey-West t-stat and p-value."""
    r = np.zeros(len(res.params))
    for n in names:
        r[list(res.params.index).index(n)] = 1
    t = res.t_test(r)
    return float(np.squeeze(t.effect)), float(np.squeeze(t.tvalue)), float(np.squeeze(t.pvalue))


def save_table(df, name, index=True):
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLE_DIR / f"{name}.csv", index=index)


def to_markdown(df, floatfmt="{:.3f}", index=True):
    """Small markdown table writer (no extra dependency)."""
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def new_fig(nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def save_fig(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return lg


def fig_legend(fig, ax, ncol):
    """One shared legend in a row under the figure title."""
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK_2)
