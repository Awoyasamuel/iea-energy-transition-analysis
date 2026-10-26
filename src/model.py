"""Bayesian trend model for energy-source shares (PyMC)."""
import numpy as np
import pandas as pd
import pymc as pm
from .energy import logit, inv_logit


def fit_logit_trend(years, share_pct, t0=2000, draws=1500, tune=1500, chains=4, seed=42,
                    ar1=True, prior_b_sd=0.1, target_accept=0.99):
    """Bayesian linear trend on the logit of a share, with AR(1) errors.

    logit(p_t) = a + b * (t - t0) + u_t,   u_t = rho * u_{t-1} + e_t
    The logit keeps projected shares inside (0, 1); AR(1) errors stop the model
    from being over-confident when neighbouring years are correlated.
    """
    t = np.asarray(years) - t0
    y = logit(np.asarray(share_pct) / 100)
    with pm.Model() as m:
        a = pm.Normal("a", mu=float(y.mean()), sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=prior_b_sd)     # logit-units per year
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        mu = a + b * t
        if ar1:
            rho = pm.Uniform("rho", lower=-0.95, upper=0.95)
            pm.Normal("y0", mu=mu[0], sigma=sigma / pm.math.sqrt(1 - rho**2), observed=y[0])
            pm.Normal("y_rest", mu=mu[1:] + rho * (y[:-1] - mu[:-1]), sigma=sigma,
                      observed=y[1:])
        else:
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=False, cores=1)
    return m, idata, t0


def forecast(idata, last_year, last_obs_share_pct, horizon_years, t0, seed=1):
    """Posterior predictive for future years, propagating the AR(1) error."""
    rng = np.random.default_rng(seed)
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    a, b, sig = post["a"].values, post["b"].values, post["sigma"].values
    rho = post["rho"].values if "rho" in post else np.zeros_like(a)
    u_prev = logit(last_obs_share_pct / 100) - (a + b * (last_year - t0))
    out = np.empty((a.size, len(horizon_years)))
    for j, yr in enumerate(horizon_years):
        u_prev = rho * u_prev + rng.normal(0, sig)
        out[:, j] = inv_logit(a + b * (yr - t0) + u_prev) * 100
    return pd.DataFrame(out, columns=list(horizon_years))


def summarize(draws: pd.DataFrame, prob=0.90) -> pd.DataFrame:
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    return pd.DataFrame({"median": draws.median(), f"lo{int(prob*100)}": draws.quantile(lo),
                         f"hi{int(prob*100)}": draws.quantile(hi), "mean": draws.mean()})


# ---------------------------------------------------------------------------
# Bayesian smooth local trend (state-space) on the logit of a share
# ---------------------------------------------------------------------------
import pytensor.tensor as pt


def fit_smooth_trend_latent(share_pct, draws=800, tune=1000, chains=4, seed=42, target_accept=0.95):
    """Bayesian smooth-trend (integrated random walk) model, logit scale.

        y_t    = mu_t + e_t,            e_t ~ N(0, s_obs^2)   observation noise
        mu_t   = mu_{t-1} + beta_{t-1}                         level follows slope
        beta_t = beta_{t-1} + z_t,      z_t ~ N(0, s_slp^2)    slope drifts slowly

    The slope is allowed to change over time, so the model can describe a share
    that was flat for years and then accelerated. We deliberately omit a separate
    level-noise term: with ~35 annual points it cannot be told apart from
    observation noise (an earlier version with both had divergences). The slope is
    non-centred (z_raw) so the sampler does not struggle with funnel geometry.
    """
    y = logit(np.asarray(share_pct) / 100)
    T = len(y)
    with pm.Model() as m:
        s_obs = pm.HalfNormal("s_obs", 0.02)
        s_slp = pm.HalfNormal("s_slp", 0.005)
        mu0 = pm.Normal("mu0", mu=float(y[0]), sigma=0.5)
        b0 = pm.Normal("b0", mu=0.0, sigma=0.05)
        z = pm.Normal("z_raw", 0.0, 1.0, shape=T - 1)
        beta = pm.Deterministic("beta", pt.concatenate([b0[None], b0 + pt.cumsum(s_slp * z)]))
        mu = pm.Deterministic("mu", pt.concatenate([mu0[None], mu0 + pt.cumsum(beta[:-1])]))
        pm.Normal("y", mu=mu, sigma=s_obs, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=False, cores=1)
    return m, idata


def forecast_smooth_latent(idata, horizon_years, last_year, seed=1, include_obs_noise=True):
    """Simulate the latent-state smooth-trend model forward from the last posterior state."""
    rng = np.random.default_rng(seed)
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    mu = post["mu"].values[-1]
    beta = post["beta"].values[-1]
    s_obs, s_slp = post["s_obs"].values, post["s_slp"].values
    horizon_years = list(horizon_years)
    out = np.empty((mu.size, len(horizon_years)))
    step_year = last_year
    for j, yr in enumerate(horizon_years):
        while step_year < yr:
            mu = mu + beta
            beta = beta + rng.normal(0, s_slp)
            step_year += 1
        noise = rng.normal(0, s_obs) if include_obs_noise else 0.0
        out[:, j] = inv_logit(mu + noise) * 100
    return pd.DataFrame(out, columns=horizon_years)


# ---------------------------------------------------------------------------
# Same smooth-trend model, but with the latent states marginalised by a Kalman filter
# ---------------------------------------------------------------------------
def _kalman(y, s_obs, s_slp, prior_sd=(0.5, 0.05)):
    """Kalman filter for state x_t=[level, slope]; returns filtered means/covs,
    standardised one-step-ahead errors and the log-likelihood (all PyTensor)."""
    import pytensor
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    y = np.asarray(y, dtype="float64")

    def update(y_t, m_p, P_p, so):
        S = P_p[0, 0] + so**2
        v = y_t - m_p[0]
        K = P_p[:, 0] / S
        m = m_p + K * v
        P = P_p - pt.outer(K, K) * S
        return m, P, v / pt.sqrt(S), -0.5 * (pt.log(2 * np.pi * S) + v**2 / S)

    m0 = pt.as_tensor_variable(np.array([y[0], 0.0]))
    P0 = pt.as_tensor_variable(np.diag([prior_sd[0] ** 2, prior_sd[1] ** 2]))
    m1, P1, e1, ll1 = update(y[0], m0, P0, s_obs)       # update with first observation

    def step(y_t, m, P, so, ss):
        Q = pt.stack([pt.stack([0.0 * ss, 0.0 * ss]), pt.stack([0.0 * ss, ss**2])])
        m_p = pt.dot(F, m)
        P_p = pt.dot(pt.dot(F, P), F.T) + Q
        return update(y_t, m_p, P_p, so)

    (ms, Ps, es, lls), _ = pytensor.scan(step, sequences=[pt.as_tensor_variable(y[1:])],
                                         outputs_info=[m1, P1, None, None],
                                         non_sequences=[s_obs, s_slp])
    m_all = pt.concatenate([m1[None, :], ms], axis=0)
    P_all = pt.concatenate([P1[None, :, :], Ps], axis=0)
    e_all = pt.concatenate([e1[None], es])
    return m_all, P_all, e_all, ll1 + lls.sum()


def fit_smooth_trend(share_pct, draws=1000, tune=1000, chains=4, seed=42, target_accept=0.95):
    """Smooth local trend (see fit_smooth_trend_latent) with the states integrated out.

    Mathematically the same model, but the sampler only has to explore 2 parameters
    (s_obs, s_slp) instead of ~70 latent values, so it is far faster and has no funnel.
    The filtered state at the last year and the standardised one-step-ahead errors are
    stored for forecasting and diagnostics.
    """
    y = logit(np.asarray(share_pct) / 100)
    with pm.Model() as m:
        s_obs = pm.HalfNormal("s_obs", 0.02)
        s_slp = pm.HalfNormal("s_slp", 0.005)
        m_all, P_all, e_all, ll = _kalman(y, s_obs, s_slp)
        pm.Potential("kalman_loglik", ll)
        pm.Deterministic("level_filt", m_all[:, 0])
        pm.Deterministic("slope_filt", m_all[:, 1])
        pm.Deterministic("std_err", e_all)
        pm.Deterministic("P_last", P_all[-1])
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=False, cores=1)
    return m, idata


def forecast_smooth(idata, horizon_years, last_year, seed=1, include_obs_noise=True):
    """Posterior predictive: draw the last state from its filtered distribution, then
    simulate level/slope forward (propagating parameter, state and future-shock uncertainty)."""
    rng = np.random.default_rng(seed)
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    mu = post["level_filt"].values[-1].copy()
    beta = post["slope_filt"].values[-1].copy()
    P = post["P_last"].values                           # (2, 2, n)
    # draw last-state uncertainty (2x2 Gaussian) per posterior draw
    sd_mu = np.sqrt(np.maximum(P[0, 0], 1e-12))
    cov = P[0, 1]
    z1, z2 = rng.normal(size=mu.size), rng.normal(size=mu.size)
    mu = mu + sd_mu * z1
    beta = beta + (cov / sd_mu) * z1 + np.sqrt(np.maximum(P[1, 1] - cov**2 / sd_mu**2, 1e-12)) * z2
    s_obs, s_slp = post["s_obs"].values, post["s_slp"].values
    horizon_years = list(horizon_years)
    out = np.empty((mu.size, len(horizon_years)))
    step_year = last_year
    for j, yr in enumerate(horizon_years):
        while step_year < yr:
            mu = mu + beta
            beta = beta + rng.normal(0, s_slp)
            step_year += 1
        noise = rng.normal(0, s_obs) if include_obs_noise else 0.0
        out[:, j] = inv_logit(mu + noise) * 100
    return pd.DataFrame(out, columns=horizon_years)
