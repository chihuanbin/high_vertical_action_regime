from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import resolve_path


def simulate_catalogue(data: pd.DataFrame, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    output = data[["ID", "Rg_kpc", "logAge50"]].copy()
    age = output["logAge50"].to_numpy() - 8.0
    radius = output["Rg_kpc"].to_numpy() - np.nanmedian(output["Rg_kpc"])
    observed = data["log10_Jz_over_abs_Lz"].to_numpy()
    base = np.nanmedian(observed) + 0.25 * age
    if kind == "smooth_flare":
        base = base + 0.08 * radius
    scatter = np.nanstd(observed - np.nanmedian(observed))
    output["response"] = base + rng.normal(0, scatter, len(output))
    return output


def simulation_statistics(data: pd.DataFrame) -> dict:
    fit = smf.quantreg("response ~ logAge50 + Rg_kpc", data).fit(q=0.9, max_iter=2000)
    median = smf.quantreg("response ~ logAge50 + Rg_kpc", data).fit(q=0.5, max_iter=2000)
    return {"q90_radius_slope": fit.params["Rg_kpc"], "q50_radius_slope": median.params["Rg_kpc"],
            "q90_minus_q50_slope": fit.params["Rg_kpc"] - median.params["Rg_kpc"],
            "q90_iterations": int(fit.iterations), "q50_iterations": int(median.iterations),
            "converged": bool(fit.iterations < 2000 and median.iterations < 2000)}


def run_null_simulations(config: dict, catalogues_per_class: int | None = None) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = pd.read_csv(tables / "analysis_catalogue.csv")
    data = table[table["potential"] == config["potentials"]["primary"]].dropna(subset=["Rg_kpc", "logAge50", "log10_Jz_over_abs_Lz"])
    if len(data) != config["project"]["expected_sample_size"]:
        raise RuntimeError("Null simulations require the complete primary-potential sample")
    count = int(catalogues_per_class or config["null_simulations"]["catalogues_per_class"])
    rng = np.random.default_rng(config["project"]["seed"] + 919)
    rows = []
    for kind in config["null_simulations"]["classes"]:
        for iteration in range(count):
            rows.append({"null_class": kind, "simulation": iteration, **simulation_statistics(simulate_catalogue(data, kind, rng))})
    output = pd.DataFrame(rows)
    output.to_csv(tables / "null_simulation_summary.csv", index=False)
    return {"catalogues_per_class": count, "rows": len(output)}
