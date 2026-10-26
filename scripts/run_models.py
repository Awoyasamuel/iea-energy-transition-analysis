"""Fit the Bayesian models and save posteriors + projections to results/.

Run:  python scripts/run_models.py        (about 10-15 min on one CPU core)
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import arviz as az, pandas as pd
from src.energy import *
from src.model import *

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
ej = load_tes()
SERIES = {"renewables": RENEWABLES, "modern_renewables": MODERN_REN}
HORIZON = list(range(2025, 2031))
rows, diag = [], []

for name, cols in SERIES.items():
    s = group_share(ej, cols)
    # --- M1: logit linear trend + AR(1), full period and recent-period sensitivity
    for tag, start in [("m1_1990", 1990), ("m1_2010", 2010)]:
        ss = s.loc[start:]
        m, idata, t0 = fit_logit_trend(ss.index.values, ss.values, draws=1500, tune=2500, target_accept=0.99)
        fc = forecast(idata, 2024, ss.iloc[-1], HORIZON, t0)
        fc.to_csv(RES / f"forecast_{name}_{tag}.csv", index=False)
        idata.to_netcdf(RES / f"idata_{name}_{tag}.nc")
        sm = az.summary(idata, var_names=["a", "b", "sigma", "rho"])
        diag.append(dict(series=name, model=tag, max_rhat=float(sm["r_hat"].max()),
                         min_ess_bulk=float(sm["ess_bulk"].min()),
                         divergences=int(idata.sample_stats.to_dataset()["diverging"].sum())))
        print(name, tag, "done", flush=True)
    # --- M2: Bayesian smooth local trend (main model)
    m, idata = fit_smooth_trend(s.values, draws=1000, tune=1000, target_accept=0.95)
    fc = forecast_smooth(idata, HORIZON, 2024)
    fc.to_csv(RES / f"forecast_{name}_m2_smooth.csv", index=False)
    idata.to_netcdf(RES / f"idata_{name}_m2_smooth.nc")
    sm = az.summary(idata, var_names=["s_obs", "s_slp"])
    sm.to_csv(RES / f"param_summary_{name}_m2_smooth.csv")
    diag.append(dict(series=name, model="m2_smooth", max_rhat=float(sm["r_hat"].max()),
                     min_ess_bulk=float(sm["ess_bulk"].min()),
                     divergences=int(idata.sample_stats.to_dataset()["diverging"].sum())))
    print(name, "m2 done", flush=True)
    pd.DataFrame(diag).to_csv(RES / "sampler_diagnostics.csv", index=False)
print("ALL DONE", flush=True)
