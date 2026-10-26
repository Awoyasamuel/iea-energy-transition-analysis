"""Helper functions for the Global Energy Transition Analysis project."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "iea_tes_by_source_world.csv"

FOSSIL = ["Coal and coal products", "Natural gas", "Oil and oil products"]
MODERN_REN = ["Hydropower", "Solar, wind and other renewables"]
RENEWABLES = MODERN_REN + ["Biofuels and waste"]
LOW_CARBON = RENEWABLES + ["Nuclear"]
TJ_PER_EJ = 1e6


def load_tes(path=RAW) -> pd.DataFrame:
    """Load the IEA TES-by-source extract and return a clean wide table in EJ.

    The file has 2 metadata lines (source + licence) and a blank line before
    the header, and the year column has no name. Values are in TJ.
    """
    df = pd.read_csv(path, skiprows=3).rename(columns={"Unnamed: 0": "year"})
    assert df["Units"].nunique() == 1 and df["Units"].iloc[0] == "TJ", "unexpected units"
    df = df.drop(columns="Units").set_index("year").sort_index()
    df.columns.name = "source"
    return df / TJ_PER_EJ  # TJ -> EJ


def validate(ej: pd.DataFrame) -> dict:
    """Basic data-quality checks; raises if something is wrong."""
    years = ej.index.to_numpy()
    checks = {
        "n_years": len(ej),
        "year_range": (int(years.min()), int(years.max())),
        "contiguous_years": bool((np.diff(years) == 1).all()),
        "n_missing": int(ej.isna().sum().sum()),
        "all_positive": bool((ej > 0).all().all()),
    }
    assert checks["contiguous_years"] and checks["n_missing"] == 0 and checks["all_positive"]
    return checks


def shares(ej: pd.DataFrame) -> pd.DataFrame:
    """Percentage share of each source in total energy supply."""
    return ej.div(ej.sum(axis=1), axis=0) * 100


def group_share(ej: pd.DataFrame, cols) -> pd.Series:
    """Share (%) of a group of sources in total energy supply."""
    return ej[cols].sum(axis=1) / ej.sum(axis=1) * 100


def cagr(start: float, end: float, periods: int) -> float:
    """Compound annual growth rate. `periods` = number of annual INTERVALS
    (e.g. 1990 -> 2024 is 34 intervals, not 35)."""
    return (end / start) ** (1 / periods) - 1


def cagr_table(ej: pd.DataFrame, windows) -> pd.DataFrame:
    out = {}
    for a, b in windows:
        out[f"{a}-{b}"] = {c: cagr(ej.loc[a, c], ej.loc[b, c], b - a) for c in ej.columns}
        out[f"{a}-{b}"]["Total supply"] = cagr(ej.loc[a].sum(), ej.loc[b].sum(), b - a)
    return pd.DataFrame(out) * 100


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))
