from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path
from .dynamics import compute_actions, potential_table
from .uncertainty import nominal_phase_space, phase_space_draws


ACTION_COLUMNS = ["JR_kpc_kms", "Lz_kpc_kms", "Jz_kpc_kms"]


def summarize_actions(actions: pd.DataFrame, total_draws: int) -> pd.DataFrame:
    valid = actions[actions["action_valid"]].copy()
    rows = []
    for (cluster_id, potential), group in valid.groupby(["ID", "potential"], sort=False):
        row = {"ID": cluster_id, "potential": potential, "valid_draws": len(group),
               "invalid_draw_fraction": 1 - len(group) / total_draws}
        for column in ACTION_COLUMNS:
            values = group[column].to_numpy()
            p16, median, p84 = np.quantile(values, [0.16, 0.5, 0.84])
            row[f"{column}_p16"] = p16
            row[f"{column}_median"] = median
            row[f"{column}_p84"] = p84
            row[f"{column}_relative_interval"] = (p84 - p16) / (2 * abs(median)) if median != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_actions(config: dict, draws: int | None = None, max_clusters: int | None = None,
                potential_name: str | None = None) -> dict:
    results = resolve_path(config, "results")
    production = max_clusters is None and (draws is None or draws == config["uncertainty"]["monte_carlo_draws"])
    tables = results / "tables" if production else results / "smoke" / "tables"
    cache = results / "cache" if production else results / "smoke" / "cache"
    tables.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(results / "tables" / "master_sample.csv")
    if max_clusters is not None:
        sample = sample.head(max_clusters).copy()
    draws = int(draws or config["uncertainty"]["monte_carlo_draws"])
    nominal = nominal_phase_space(sample)
    mc = phase_space_draws(sample, draws, config["project"]["seed"])
    nominal_rows = []
    summaries = []
    definitions = config["potentials"]["models"]
    if potential_name:
        definitions = [item for item in definitions if item["name"] == potential_name]
        if not definitions:
            raise ValueError(f"Unknown potential: {potential_name}")
    for definition in definitions:
        name = definition["name"]
        nominal_action = compute_actions(nominal, config, name)
        nominal_rows.append(nominal_action)
        checkpoint = cache / f"action_draws_{name}.parquet"
        if checkpoint.exists():
            cached = pd.read_parquet(checkpoint)
            complete = len(cached) == len(sample) * draws and cached["ID"].nunique() == len(sample)
        else:
            complete = False
        if complete:
            mc_action = cached
        else:
            mc_action = compute_actions(mc, config, name)
            mc_action.to_parquet(checkpoint, index=False)
        summaries.append(summarize_actions(mc_action, draws))
    nominal_table = pd.concat(nominal_rows, ignore_index=True)
    summary_table = pd.concat(summaries, ignore_index=True)
    suffix = f"_{potential_name}" if potential_name else ""
    nominal_table.to_csv(tables / f"action_catalogue_nominal{suffix}.csv", index=False)
    summary_table.to_csv(tables / f"action_mc_summary{suffix}.csv", index=False)
    summary_table.to_csv(tables / f"action_uncertainties{suffix}.csv", index=False)
    potential_table(config).to_csv(tables / "potential_definitions.csv", index=False)
    return {"profile": "production" if production else "smoke", "clusters": len(sample), "draws_per_cluster": draws, "potentials": len(definitions),
            "nominal_rows": len(nominal_table), "summary_rows": len(summary_table)}


def merge_action_checkpoints(config: dict, required: list[str] | None = None) -> dict:
    results = resolve_path(config, "results")
    tables, cache = results / "tables", results / "cache"
    required = required or [item["name"] for item in config["potentials"]["models"]]
    expected_clusters = int(config["project"]["expected_sample_size"])
    expected_draws = int(config["uncertainty"]["monte_carlo_draws"])
    nominals, summaries = [], []
    for name in required:
        nominal_path = tables / f"action_catalogue_nominal_{name}.csv"
        draw_path = cache / f"action_draws_{name}.parquet"
        if not nominal_path.exists() or not draw_path.exists():
            raise RuntimeError(f"Missing production action checkpoint for {name}")
        nominal = pd.read_csv(nominal_path)
        draws = pd.read_parquet(draw_path)
        if len(nominal) != expected_clusters or len(draws) != expected_clusters * expected_draws:
            raise RuntimeError(f"Incomplete checkpoint for {name}: nominal={len(nominal)}, draws={len(draws)}")
        nominals.append(nominal)
        summaries.append(summarize_actions(draws, expected_draws))
    nominal_table = pd.concat(nominals, ignore_index=True)
    summary_table = pd.concat(summaries, ignore_index=True)
    nominal_table.to_csv(tables / "action_catalogue_nominal.csv", index=False)
    summary_table.to_csv(tables / "action_mc_summary.csv", index=False)
    summary_table.to_csv(tables / "action_uncertainties.csv", index=False)
    return {"potentials": required, "nominal_rows": len(nominal_table), "summary_rows": len(summary_table)}
