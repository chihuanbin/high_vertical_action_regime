from __future__ import annotations

import numpy as np
import pandas as pd


def split_normal_draws(median: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                       n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a two-piece normal whose side scales match the quoted intervals."""
    median = np.asarray(median, dtype=float)[:, None]
    sigma_low = np.maximum(np.asarray(median[:, 0] - lower, dtype=float), 0)[:, None]
    sigma_high = np.maximum(np.asarray(upper - median[:, 0], dtype=float), 0)[:, None]
    z = rng.standard_normal((len(median), n_draws))
    return median + np.where(z < 0, sigma_low, sigma_high) * z


def phase_space_draws(sample: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(sample)
    distance = split_normal_draws(
        sample["distance_kpc"].to_numpy(), sample["dist16"].to_numpy() / 1000,
        sample["dist84"].to_numpy() / 1000, n_draws, rng)
    log_age = split_normal_draws(sample["logAge50"], sample["logAge16"], sample["logAge84"], n_draws, rng)
    pmra = rng.normal(sample["pmRA"].to_numpy()[:, None], sample["e_pmRA"].to_numpy()[:, None], (n, n_draws))
    pmdec = rng.normal(sample["pmDE"].to_numpy()[:, None], sample["e_pmDE"].to_numpy()[:, None], (n, n_draws))
    rv = rng.normal(sample["RV"].to_numpy()[:, None], sample["e_RV"].to_numpy()[:, None], (n, n_draws))
    rows = pd.DataFrame({
        "ID": np.repeat(sample["ID"].to_numpy(dtype=int), n_draws),
        "draw": np.tile(np.arange(1, n_draws + 1), n),
        "RAdeg": np.repeat(sample["RAdeg"].to_numpy(), n_draws),
        "DEdeg": np.repeat(sample["DEdeg"].to_numpy(), n_draws),
        "distance_kpc": distance.ravel(), "pmRA": pmra.ravel(), "pmDE": pmdec.ravel(),
        "RV": rv.ravel(), "logAge": log_age.ravel(),
    })
    rows["input_valid"] = (
        np.isfinite(rows[["distance_kpc", "pmRA", "pmDE", "RV"]]).all(axis=1)
        & (rows["distance_kpc"] > 0)
    )
    return rows


def nominal_phase_space(sample: pd.DataFrame) -> pd.DataFrame:
    columns = ["ID", "RAdeg", "DEdeg", "distance_kpc", "pmRA", "pmDE", "RV", "logAge50"]
    out = sample[columns].copy().rename(columns={"logAge50": "logAge"})
    out["draw"] = 0
    out["input_valid"] = True
    return out

