"""Rolling-origin backtest: how well would each model have forecast the past?

For each origin year O we train on 1990..O and forecast O+1..min(O+6, 2024), then
compare with what actually happened. Run: python scripts/run_backtest.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy import stats
from src.energy import *
from src.model import *

ej = load_tes()
SERIES = {"renewables": RENEWABLES, "modern_renewables": MODERN_REN}
ORIGINS = [2012, 2014, 2016, 2018]
rows = []

def add(series, model, origin, years, draws_df, actual):
    for yr in years:
        d = draws_df[yr]
        rows.append(dict(series=series, model=model, origin=origin, year=yr, horizon=yr - origin,
                         actual=actual.loc[yr], median=d.median(),
                         lo90=d.quantile(0.05), hi90=d.quantile(0.95)))

for name, cols in SERIES.items():
    s = group_share(ej, cols)
    for O in ORIGINS:
        train = s.loc[:O]; yrs = list(range(O + 1, min(O + 6, 2024) + 1))
        # M0: random walk with drift on the logit scale (simple baseline, no MCMC)
        y = logit(train.values / 100); d = np.diff(y); n = len(d)
        drift, sd = d.mean(), d.std(ddof=1)
        rng = np.random.default_rng(0)
        sims = {yr: inv_logit(y[-1] + drift * (yr - O) +
                              rng.normal(0, sd * np.sqrt(yr - O), 4000)) * 100 for yr in yrs}
        add(name, "M0 random walk + drift", O, yrs, pd.DataFrame(sims), s)
        # M1: logit linear trend + AR(1) fitted on everything up to the origin
        m, idata, t0 = fit_logit_trend(train.index.values, train.values, draws=600, tune=800,
                                       chains=2, target_accept=0.95)
        add(name, "M1 linear trend + AR(1)", O, yrs, forecast(idata, O, train.iloc[-1], yrs, t0), s)
        # M2: Bayesian smooth local trend
        m, idata = fit_smooth_trend(train.values, draws=500, tune=800, chains=2,
                                    target_accept=0.97)
        add(name, "M2 smooth local trend", O, yrs, forecast_smooth(idata, yrs, O), s)
        print(name, O, "done", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "backtest_results.csv", index=False)
print("ALL DONE", flush=True)
