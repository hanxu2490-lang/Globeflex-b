"""Robustness of the Question 2 crisis regression (methods page).

1. Local-currency EM returns: how much of EM's crisis performance comes from currency moves.
2. 2022 as the start of the current period.
(The EM Small Cap net-return check for Question 1 is in Module C, section 7, and Module B.)
"""
import pandas as pd

from . import config as C
from . import utils as U
from . import step3_crisis


def local_currency(m):
    cols, pvals = {}, {}
    for dep in ("EM", "EM_LC"):
        for controls in (False, True):
            name = f"{'EM (USD)' if dep == 'EM' else 'EM (local ccy)'} - World, {'with' if controls else 'no'} X"
            cols[name], res = step3_crisis.crisis_column(m, dep, controls)
            pvals[name] = res.pvalues.rename(step3_crisis.COEF_NAMES)
    tab = pd.DataFrame(cols)
    U.save_table(tab, "robust2_local_currency_crisis")
    U.save_table(pd.DataFrame(pvals), "robust2_local_currency_crisis_pvalues")
    return tab


def current_from_2022(m):
    tab, pvals = step3_crisis.regression_table(m, current_start=C.CURRENT_START_ROBUST)
    U.save_table(tab, "robust3_current_from_2022_crisis")
    U.save_table(pvals, "robust3_current_from_2022_crisis_pvalues")
    return tab


def run(m):
    return {"local_ccy": local_currency(m), "current_2022": current_from_2022(m)}
