from __future__ import annotations

import numpy as np
import pandas as pd

from .config import resolve_path
from .secondary import standardized_mean_difference


MATCH_COLUMNS = ["logAge50", "Rg_kpc", "log_abs_Lz", "eccentricity"]
OUTCOMES = ["log10_Jz_over_abs_Lz", "eta_z", "zmax_over_Rg"]


def build_matches(data: pd.DataFrame, treated_ids: set[int], config: dict, strict: bool) -> pd.DataFrame:
    work = data.copy()
    work["log_abs_Lz"] = np.log10(work["Lz_kpc_kms"].abs().where(work["Lz_kpc_kms"].abs() > 0))
    treated = work[work["ID"].isin(treated_ids)]
    controls = work[~work["ID"].isin(treated_ids)]
    cfg = config["matching"]
    rows = []
    for _, item in treated.iterrows():
        eligible = controls[(controls["logAge50"] - item.logAge50).abs() <= cfg["age_caliper_dex"]]
        eligible = eligible[(eligible["Rg_kpc"] - item.Rg_kpc).abs() <= cfg["radius_caliper_kpc"]]
        if strict:
            eligible = eligible[(eligible["log_abs_Lz"] - item.log_abs_Lz).abs() <= cfg["log_lz_caliper_dex"]]
            eligible = eligible[(eligible["eccentricity"] - item.eccentricity).abs() <= cfg["eccentricity_caliper"]]
        if eligible.empty:
            continue
        scale = np.array([cfg["age_caliper_dex"], cfg["radius_caliper_kpc"],
                          cfg["log_lz_caliper_dex"], cfg["eccentricity_caliper"]])
        columns = MATCH_COLUMNS if strict else MATCH_COLUMNS[:2]
        eligible_values = eligible[columns].to_numpy(dtype=float)
        treated_values = item[columns].to_numpy(dtype=float)
        distance = np.sqrt(np.sum(((eligible_values - treated_values) / scale[:len(columns)]) ** 2, axis=1))
        selected = eligible.iloc[int(np.argmin(distance))]
        row = {"match_type": "strict" if strict else "radius_age", "treated_ID": int(item.ID),
               "control_ID": int(selected.ID), "distance": float(np.min(distance))}
        for column in MATCH_COLUMNS:
            row[f"treated_{column}"] = item[column]
            row[f"control_{column}"] = selected[column]
        for outcome in OUTCOMES:
            row[f"treated_{outcome}"] = item[outcome]
            row[f"control_{outcome}"] = selected[outcome]
            row[f"difference_{outcome}"] = item[outcome] - selected[outcome]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_matches(pairs: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance = []
    for match_type, group in pairs.groupby("match_type"):
        for column in MATCH_COLUMNS:
            balance.append({"match_type": match_type, "covariate": column,
                "standardized_mean_difference": standardized_mean_difference(group[f"treated_{column}"], group[f"control_{column}"])})
    rng = np.random.default_rng(config["project"]["seed"] + 313)
    results = []
    for match_type, group in pairs.groupby("match_type"):
        for outcome in OUTCOMES:
            values = group[f"difference_{outcome}"].dropna().to_numpy()
            bootstrap = np.array([np.mean(rng.choice(values, len(values), replace=True))
                                  for _ in range(config["models"]["bootstrap_count"])])
            results.append({"match_type": match_type, "outcome": outcome, "treated_clusters": len(values),
                "unique_controls": group.loc[group[f"difference_{outcome}"].notna(), "control_ID"].nunique(),
                "maximum_control_reuse": int(group["control_ID"].value_counts().max()),
                "mean_difference": np.mean(values), "median_difference": np.median(values),
                "bootstrap_ci_low": np.quantile(bootstrap, 0.025), "bootstrap_ci_high": np.quantile(bootstrap, 0.975)})
    return pd.DataFrame(balance), pd.DataFrame(results)


def run_matching(config: dict) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = pd.read_csv(tables / "analysis_catalogue.csv")
    primary = table[table["potential"] == config["potentials"]["primary"]].copy()
    membership = pd.read_csv(tables / "gmm_membership.csv")
    treated_ids = set(membership.loc[membership["descriptive_high_action_component"].astype(bool), "ID"].astype(int))
    pairs = pd.concat([build_matches(primary, treated_ids, config, False), build_matches(primary, treated_ids, config, True)], ignore_index=True)
    balance, results = summarize_matches(pairs, config)
    pairs.to_csv(tables / "matched_pairs.csv", index=False)
    balance.to_csv(tables / "matching_balance.csv", index=False)
    results.to_csv(tables / "matched_results.csv", index=False)
    return {"treated": len(treated_ids), "pairs": len(pairs), "strict_pairs": int((pairs["match_type"] == "strict").sum())}
