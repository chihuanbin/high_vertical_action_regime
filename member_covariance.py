from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import resolve_path


def load_member_astrometry(path, wanted_ids: set[int]) -> dict[int, np.ndarray]:
    """Stream member rows and retain probability-weighted (parallax, pmRA, pmDE)."""
    values: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                cluster_id = int(line[29:33])
            except ValueError:
                continue
            if cluster_id not in wanted_ids:
                continue
            try:
                probability = float(line[58:78])
                pmra = float(line[220:243])
                pmdec = float(line[266:289])
                parallax = float(line[311:334])
            except ValueError:
                continue
            if np.isfinite([probability, parallax, pmra, pmdec]).all() and probability > 0:
                values[cluster_id].append((probability, parallax, pmra, pmdec))
    return {key: np.asarray(rows, dtype=float) for key, rows in values.items()}


def bootstrap_centroid_covariance(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    weights = values[:, 0]
    weights = weights / weights.sum()
    astrometry = values[:, 1:]
    estimates = np.empty((draws, 3))
    for draw in range(draws):
        indices = rng.choice(len(values), size=len(values), replace=True, p=weights)
        estimates[draw] = np.average(astrometry[indices], axis=0, weights=weights[indices])
    return estimates.mean(axis=0), np.cov(estimates, rowvar=False, ddof=1)


def run_member_covariance(config: dict) -> dict:
    tables = resolve_path(config, "results") / "tables"
    sample = pd.read_csv(tables / "master_sample.csv")
    member_values = load_member_astrometry(resolve_path(config, "members"), set(sample["ID"].astype(int)))
    rng = np.random.default_rng(config["project"]["seed"] + 101)
    rows = []
    for cluster_id in sample["ID"].astype(int):
        values = member_values.get(cluster_id)
        if values is None or len(values) < 3:
            rows.append({"ID": cluster_id, "member_count_used": 0, "status": "insufficient_members"})
            continue
        mean, covariance = bootstrap_centroid_covariance(values, config["uncertainty"]["member_bootstrap_draws"], rng)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        rows.append({"ID": cluster_id, "member_count_used": len(values), "status": "ok",
            "bootstrap_Plx": mean[0], "bootstrap_pmRA": mean[1], "bootstrap_pmDE": mean[2],
            "bootstrap_e_Plx": scale[0], "bootstrap_e_pmRA": scale[1], "bootstrap_e_pmDE": scale[2],
            "bootstrap_corr_Plx_pmRA": correlation[0, 1], "bootstrap_corr_Plx_pmDE": correlation[0, 2],
            "bootstrap_corr_pmRA_pmDE": correlation[1, 2]})
    output = pd.DataFrame(rows)
    output.to_csv(tables / "member_bootstrap_covariance.csv", index=False)
    return {"clusters": len(output), "estimated": int((output["status"] == "ok").sum())}

