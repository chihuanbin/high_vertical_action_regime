from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import resolve_path


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else np.nan


def run_gmm_bootstrap(config: dict, bootstrap_count: int | None = None) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = pd.read_csv(tables / "analysis_catalogue.csv")
    data = table[table["potential"] == config["potentials"]["primary"]].reset_index(drop=True)
    raw = np.log10(data[["JR_kpc_kms", "Jz_kpc_kms", "Lz_kpc_kms"]].abs().to_numpy())
    valid = np.isfinite(raw).all(axis=1)
    data, raw = data.loc[valid].reset_index(drop=True), raw[valid]
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    criteria = pd.read_csv(tables / "gmm_model_selection.csv")
    best = criteria.loc[criteria["bic"].idxmin()]
    covariance, k = str(best.covariance_type), int(best.k)
    reference = GaussianMixture(k, covariance_type=covariance, n_init=config["gmm"]["initializations"],
        random_state=config["project"]["seed"]).fit(x)
    reference_labels = reference.predict(x)
    reference_medians = data["log10_Jz_over_abs_Lz"].groupby(reference_labels).median()
    high_reference = int(reference_medians.idxmax())
    bootstrap_count = int(bootstrap_count or config["gmm"]["bootstrap_count"])
    rng = np.random.default_rng(config["project"]["seed"] + 727)
    summary, membership_count = [], np.zeros(len(data))
    for iteration in range(bootstrap_count):
        sampled_indices = rng.integers(0, len(data), len(data))
        sampled_raw = raw[sampled_indices]
        boot_scaler = StandardScaler().fit(sampled_raw)
        boot_x = boot_scaler.transform(sampled_raw)
        model = GaussianMixture(k, covariance_type=covariance, n_init=config["gmm"]["bootstrap_initializations"],
            random_state=config["project"]["seed"] + iteration + 1).fit(boot_x)
        sampled_labels = model.predict(boot_x)
        # Duplicate occurrences retain their row identities in overlap accounting.
        reference_sampled = reference_labels[sampled_indices]
        cost = np.array([[1 - _jaccard(reference_sampled == i, sampled_labels == j) for j in range(k)] for i in range(k)])
        ref_index, candidate_index = linear_sum_assignment(cost)
        mapping = dict(zip(ref_index, candidate_index))
        high_candidate = mapping[high_reference]
        full_labels = model.predict(boot_scaler.transform(raw))
        high_mask = full_labels == high_candidate
        membership_count += high_mask
        summary.append({"bootstrap": iteration, "high_component_jaccard_with_multiplicity": 1 - cost[high_reference, high_candidate],
            "high_component_size": int(high_mask.sum()), "scaler_refit": True,
            "unique_sampled_clusters": int(np.unique(sampled_indices).size)})
    output = data[["ID", "Name"]].copy()
    output["gmm_bootstrap_high_action_probability"] = membership_count / bootstrap_count
    pd.DataFrame(summary).to_csv(tables / "gmm_bootstrap_summary.csv", index=False)
    output.to_csv(tables / "gmm_bootstrap_membership.csv", index=False)
    return {"bootstrap_count": bootstrap_count, "component_count": k, "covariance_type": covariance}
