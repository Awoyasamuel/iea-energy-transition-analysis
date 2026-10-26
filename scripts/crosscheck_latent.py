"""Cross-check: the Kalman-marginalised smooth trend must agree with the explicit
latent-state version of the same model (slower, ~2 min). Run: python scripts/crosscheck_latent.py"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import arviz as az, pandas as pd
from src.energy import *
from src.model import *

s = group_share(load_tes(), MODERN_REN)
rows = []
for tag, fit, fc in [("Kalman (marginalised)", fit_smooth_trend, forecast_smooth),
                     ("Latent states (explicit)", fit_smooth_trend_latent, forecast_smooth_latent)]:
    m, idata = fit(s.values, draws=500, tune=800, chains=4)
    sm = az.summary(idata, var_names=["s_obs", "s_slp"])
    d = fc(idata, [2030], 2024)[2030]
    rows.append(dict(model=tag, s_obs=sm.loc["s_obs", "mean"], s_slp=sm.loc["s_slp", "mean"],
                     median_2030=d.median(), lo90_2030=d.quantile(0.05), hi90_2030=d.quantile(0.95)))
    print(tag, "done", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "crosscheck_latent_vs_kalman.csv", index=False)
print("ALL DONE", flush=True)
