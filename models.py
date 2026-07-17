from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

from .config import resolve_path


def assemble_analysis_table(config: dict) -> pd.DataFrame:
    tables = resolve_path(config, "results") / "tables"
    master = pd.read_csv(tables / "master_sample.csv")
    actions = pd.read_csv(tables / "action_catalogue_nominal.csv")
    orbits = pd.read_csv(tables / "orbit_catalogue_nominal.csv")
    uncertainty = pd.read_csv(tables / "action_mc_summary.csv")
    uncertainty_columns = ["ID", "potential", "invalid_draw_fraction", "Jz_kpc_kms_relative_interval"]
    table = (actions.merge(orbits, on=["ID", "potential"], how="left")
             .merge(uncertainty[uncertainty_columns], on=["ID", "potential"], how="left")
             .merge(master, on="ID", how="left"))
    table["Jz_over_abs_Lz"] = table["Jz_kpc_kms"] / table["Lz_kpc_kms"].abs()
    table["log10_Jz_over_abs_Lz"] = np.log10(table["Jz_over_abs_Lz"].where(table["Jz_over_abs_Lz"] > 0))
    table["log10_Jz"] = np.log10(table["Jz_kpc_kms"].where(table["Jz_kpc_kms"] > 0))
    table["eta_z"] = table["Jz_kpc_kms"] / (table["JR_kpc_kms"] + table["Jz_kpc_kms"])
    table["zmax_over_Rg"] = table["zmax_kpc"] / table["Rg_kpc"]
    table["log_age_centered"] = table["logAge50"] - 8.0
    table["radius_centered"] = table["Rg_kpc"] - table["Rg_kpc"].median()
    table["sin_phi"] = np.sin(table["phi_rad"])
    table["cos_phi"] = np.cos(table["phi_rad"])
    table["response"] = table[config["models"]["primary_response"]]
    precision = config["sample"]["action_precision"]
    table["action_precision_sample"] = (
        (table["invalid_draw_fraction"] <= precision["max_invalid_fraction"])
        & (table["Jz_kpc_kms_relative_interval"] <= precision["max_relative_jz_interval"])
    )
    table.to_csv(tables / "analysis_catalogue.csv", index=False)
    return table


def _cv_rmse(formula: str, data: pd.DataFrame, seed: int, folds: int) -> float:
    if len(data) < max(2, folds):
        return float("nan")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    errors = []
    for train, test in splitter.split(data):
        fit = smf.rlm(formula, data=data.iloc[train], M=sm.robust.norms.HuberT()).fit()
        pred = fit.predict(data.iloc[test])
        errors.extend((data.iloc[test]["response"].to_numpy() - pred.to_numpy()) ** 2)
    return float(np.sqrt(np.mean(errors)))


def _sample_subsets(group: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return the two publication samples, tolerating legacy/test tables.

    July result tables and small unit-test fixtures predate the explicit
    ``action_precision_sample`` column.  Such rows are already quality-selected
    inputs, so they are treated as belonging to both named samples.
    """
    if "action_precision_sample" not in group.columns:
        mask = pd.Series(True, index=group.index)
    else:
        mask = group["action_precision_sample"].fillna(False).astype(bool)
    return [("full", group), ("action_precision", group.loc[mask])]


def fit_continuous_models(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows, comparison_rows = [], []
    needed = ["response", "log_age_centered", "radius_centered", "eccentricity", "sin_phi", "cos_phi"]
    for potential, group in table.groupby("potential"):
        for sample_name, subset in _sample_subsets(group):
            data = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
            for name, rhs in config["models"]["formulas"].items():
                formula = rhs.replace("response", "response", 1)
                if len(data) < 2:
                    continue
                fit = smf.rlm(formula, data=data, M=sm.robust.norms.HuberT()).fit()
                for term in fit.params.index:
                    coefficient_rows.append({"potential": potential, "sample": sample_name, "model": name, "term": term,
                        "coefficient": fit.params[term], "standard_error": fit.bse[term],
                        "ci_low": fit.conf_int().loc[term, 0], "ci_high": fit.conf_int().loc[term, 1], "n": len(data)})
                comparison_rows.append({"potential": potential, "sample": sample_name, "model": name, "n": len(data),
                    "robust_scale": fit.scale, "cv_rmse": _cv_rmse(formula, data, config["project"]["seed"], config["models"]["cv_folds"])})
    return pd.DataFrame(coefficient_rows), pd.DataFrame(comparison_rows)


def fit_quantiles(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    formula = "response ~ log_age_centered * radius_centered + eccentricity"
    for potential, group in table.groupby("potential"):
        for sample_name, subset in _sample_subsets(group):
            data = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["response", "log_age_centered", "radius_centered", "eccentricity"])
            for quantile in config["models"]["quantiles"]:
                if len(data) < 2:
                    continue
                fit = smf.quantreg(formula, data).fit(q=quantile, max_iter=5000)
                for term in fit.params.index:
                    rows.append({"potential": potential, "sample": sample_name, "quantile": quantile, "term": term,
                        "coefficient": fit.params[term], "standard_error": fit.bse[term],
                        "ci_low": fit.conf_int().loc[term, 0], "ci_high": fit.conf_int().loc[term, 1], "n": len(data)})
    return pd.DataFrame(rows)


def run_models(config: dict) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = assemble_analysis_table(config)
    tested_potentials = table["potential"].nunique()
    expected = config["project"]["expected_sample_size"] * tested_potentials
    if len(table) != expected:
        raise RuntimeError(f"Refusing production models: expected {expected} nominal potential rows, found {len(table)}")
    coefficients, comparisons = fit_continuous_models(table, config)
    quantiles = fit_quantiles(table, config)
    coefficients.to_csv(tables / "continuous_model_coefficients.csv", index=False)
    comparisons.to_csv(tables / "continuous_model_comparison.csv", index=False)
    quantiles.to_csv(tables / "quantile_regression.csv", index=False)
    radial = coefficients[coefficients["term"].str.contains("radius_centered", regex=False)].copy()
    radial.to_csv(tables / "radial_derivatives.csv", index=False)
    robustness_rows = []
    primary = table[table["potential"] == config["potentials"]["primary"]].set_index("ID")
    for potential, group in table.groupby("potential"):
        current = group.set_index("ID")
        joined = primary[["Jz_kpc_kms", "Jz_over_abs_Lz", "zmax_kpc"]].join(
            current[["Jz_kpc_kms", "Jz_over_abs_Lz", "zmax_kpc"]], lsuffix="_primary", rsuffix="_test")
        robustness_rows.append({"potential": potential,
            "spearman_Jz": joined["Jz_kpc_kms_primary"].corr(joined["Jz_kpc_kms_test"], method="spearman"),
            "spearman_Jz_over_abs_Lz": joined["Jz_over_abs_Lz_primary"].corr(joined["Jz_over_abs_Lz_test"], method="spearman"),
            "spearman_zmax": joined["zmax_kpc_primary"].corr(joined["zmax_kpc_test"], method="spearman")})
    pd.DataFrame(robustness_rows).to_csv(tables / "potential_robustness.csv", index=False)
    return {"analysis_rows": len(table), "coefficient_rows": len(coefficients), "quantile_rows": len(quantiles)}
