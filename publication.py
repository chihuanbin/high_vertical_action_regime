from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve_path, sha256


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures(config: dict) -> list[Path]:
    results = resolve_path(config, "results")
    figure_dir = results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(results / "tables" / "analysis_catalogue.csv")
    primary = table[table["potential"] == config["potentials"]["primary"]]
    palette = config["plotting"]["palette"]
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    color = primary["log10_Jz_over_abs_Lz"]
    axes[0].scatter(primary["R_kpc"], primary["z_kpc"], c=color, s=8, cmap="viridis", alpha=.7)
    axes[0].set(xlabel="R [kpc]", ylabel="z [kpc]")
    axes[1].scatter(primary["R_kpc"] * np.cos(primary["phi_rad"]), primary["R_kpc"] * np.sin(primary["phi_rad"]), c=color, s=8, cmap="viridis", alpha=.7)
    axes[1].set(xlabel="x [kpc]", ylabel="y [kpc]")
    paths.append(figure_dir / "sample_coverage.pdf"); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    age_bins = pd.qcut(primary["logAge50"], 3, duplicates="drop")
    for index, (_, group) in enumerate(primary.groupby(age_bins, observed=True)):
        bins = pd.qcut(group["Rg_kpc"], 8, duplicates="drop")
        summary = group.groupby(bins, observed=True).agg(radius=("Rg_kpc", "median"), median=("log10_Jz_over_abs_Lz", "median"), upper=("log10_Jz_over_abs_Lz", lambda x: x.quantile(.9)))
        ax.plot(summary.radius, summary["median"], color=palette[index], marker="o")
        ax.plot(summary.radius, summary.upper, color=palette[index], linestyle="--")
    ax.set(xlabel="Guiding radius [kpc]", ylabel=r"$\log_{10}(J_z/|L_z|)$")
    paths.append(figure_dir / "age_radius_vertical_action.pdf"); _save(fig, paths[-1])

    coefficients = pd.read_csv(results / "tables" / "continuous_model_coefficients.csv")
    m2 = coefficients[(coefficients.potential == config["potentials"]["primary"])
                      & (coefficients["sample"] == "full") & (coefficients.model == "M2")].set_index("term")
    age_grid = np.linspace(primary.logAge50.quantile(.02), primary.logAge50.quantile(.98), 120)
    radius_grid = np.linspace(primary.Rg_kpc.quantile(.02), primary.Rg_kpc.quantile(.98), 140)
    radius_reference = primary.Rg_kpc.median()
    age_centered, radius_centered = np.meshgrid(age_grid - 8.0, radius_grid - radius_reference, indexing="ij")
    predicted = (m2.loc["Intercept", "coefficient"]
                 + m2.loc["log_age_centered", "coefficient"] * age_centered
                 + m2.loc["radius_centered", "coefficient"] * radius_centered
                 + m2.loc["log_age_centered:radius_centered", "coefficient"] * age_centered * radius_centered
                 + m2.loc["eccentricity", "coefficient"] * primary.eccentricity.median())
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    surface = ax.pcolormesh(radius_grid, age_grid, predicted, cmap="viridis", shading="auto")
    contours = ax.contour(radius_grid, age_grid, predicted, levels=7, colors="white", linewidths=.7, alpha=.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.set(xlabel="Guiding radius [kpc]", ylabel=r"$\log_{10}(\mathrm{age/yr})$")
    fig.colorbar(surface, ax=ax, label=r"Predicted $\log_{10}(J_z/|L_z|)$")
    paths.append(figure_dir / "predicted_age_radius_surface.pdf"); _save(fig, paths[-1])

    slopes = coefficients[(coefficients.model == "M2") & (coefficients.term == "radius_centered") & (coefficients["sample"] == "full")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(slopes)); y = slopes.coefficient.to_numpy(); lo = y - slopes.ci_low; hi = slopes.ci_high - y
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=palette[0], capsize=3)
    potential_labels = {
        "mwpotential2014": "Fiducial",
        "mwpotential2014_disk_high": "Disk +20%",
        "mwpotential2014_disk_low": "Disk -20%",
    }
    labels = [potential_labels.get(name, name) for name in slopes.potential]
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(x, labels)
    ax.set_ylabel("M2 radial coefficient")
    paths.append(figure_dir / "potential_comparison.pdf"); _save(fig, paths[-1])

    terms = coefficients[(coefficients.model == "M2") & (coefficients["sample"] == "full")
                         & coefficients.term.isin(["radius_centered", "log_age_centered:radius_centered"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for index, (term, label) in enumerate([
        ("radius_centered", "Radius"),
        ("log_age_centered:radius_centered", "Age x radius"),
    ]):
        current = terms[terms.term == term]
        y = current.coefficient.to_numpy()
        positions = np.arange(len(current)) + (index - .5) * .18
        ax.errorbar(positions, y, yerr=[y-current.ci_low.to_numpy(), current.ci_high.to_numpy()-y],
                    fmt="o", color=palette[index], capsize=3, label=label)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(np.arange(3), ["Fiducial", "Disk +20%", "Disk -20%"])
    ax.set_ylabel("M2 coefficient"); ax.legend(frameon=False)
    paths.append(figure_dir / "interaction_coefficients.pdf"); _save(fig, paths[-1])

    bootstrap = pd.read_csv(results / "tables" / "gmm_bootstrap_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].hist(bootstrap.high_component_jaccard_with_multiplicity, bins=30, color=palette[0], alpha=.8)
    axes[0].set(xlabel="Jaccard overlap", ylabel="Bootstrap count")
    axes[1].hist(bootstrap.high_component_size, bins=30, color=palette[2], alpha=.8)
    axes[1].set(xlabel="High-action partition size", ylabel="Bootstrap count")
    paths.append(figure_dir / "bootstrap_stability.pdf"); _save(fig, paths[-1])

    matched = pd.read_csv(results / "orbital_family_analysis" / "matched_orbit_family_summary.csv")
    metrics = ["Omega_z_over_Omega_phi_abs", "Omega_R_over_Omega_phi_abs", "Jz_over_JR_plus_Jz", "Jz_over_abs_Lz", "zmax_over_Rg"]
    labels = [r"$\Omega_z/|\Omega_\phi|$", r"$\Omega_R/|\Omega_\phi|$", r"$J_z/(J_R+J_z)$", r"$J_z/|L_z|$", r"$z_{\max}/R_g$"]
    diffs = [matched[f"diff_mean_{metric}"].dropna().to_numpy() for metric in metrics]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.boxplot(diffs, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("VEAR minus matched-control mean")
    ax.tick_params(axis="x", rotation=20)
    paths.append(figure_dir / "matched_control_validation.pdf"); _save(fig, paths[-1])

    membership = pd.read_csv(results / "tables" / "gmm_membership.csv")
    spatial = primary.merge(membership[["ID", "gmm_high_action_probability"]], on="ID", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    scatter = ax.scatter(spatial.R_kpc * np.cos(spatial.phi_rad), spatial.R_kpc * np.sin(spatial.phi_rad),
                         c=spatial.gmm_high_action_probability, s=10, cmap="viridis", vmin=0, vmax=1)
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", aspect="equal")
    fig.colorbar(scatter, ax=ax, label="VEAR probability")
    paths.append(figure_dir / "vear_probability_map.pdf"); _save(fig, paths[-1])

    nulls = pd.read_csv(results / "tables" / "null_simulation_summary.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for index, (kind, group) in enumerate(nulls.groupby("null_class")):
        ax.hist(group.q90_minus_q50_slope, bins=30, density=True, alpha=.55, color=palette[index], label=kind)
    ax.set(xlabel="Simulated q90-q50 radial-slope contrast", ylabel="Density"); ax.legend(frameon=False)
    paths.append(figure_dir / "null_model_checks.pdf"); _save(fig, paths[-1])
    return paths


def build_manifest(config: dict) -> dict:
    results = resolve_path(config, "results")
    root = Path(config["_config_path"]).parent.parent
    packages = {}
    for name in ["numpy", "pandas", "scipy", "matplotlib", "astropy", "galpy", "scikit-learn", "statsmodels"]:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = "unavailable"
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception: commit = "unavailable"
    output_files = sorted(path for path in results.rglob("*") if path.is_file() and path.name != "analysis_manifest.json")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "python": platform.python_version(), "packages": packages, "configuration_sha256": config["_config_sha256"],
        "inputs": {key: {"path": str(resolve_path(config, key)), "sha256": sha256(resolve_path(config, key))}
                   for key in ["clusters", "members", "crossmatches"] if resolve_path(config, key).exists()},
        "outputs": {str(path.relative_to(root)): sha256(path) for path in output_files}}
    (results / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def consolidate_tables(config: dict) -> None:
    tables = resolve_path(config, "results") / "tables"
    analysis = pd.read_csv(tables / "analysis_catalogue.csv")
    primary = analysis[analysis["potential"] == config["potentials"]["primary"]]
    gmm = pd.read_csv(tables / "gmm_membership.csv")
    catalogue = primary.merge(gmm[["ID", "gmm_high_action_probability"]], on="ID", how="left", validate="one_to_one")
    if "action_precision_sample" not in catalogue.columns:
        catalogue["action_precision_sample"] = True
    catalogue.to_csv(tables / "machine_readable_cluster_catalogue.csv", index=False)
    coefficients = pd.read_csv(tables / "continuous_model_coefficients.csv")
    quantiles = pd.read_csv(tables / "quantile_regression.csv")
    if "sample" not in coefficients.columns or "sample" not in quantiles.columns:
        raise ValueError("Model outputs must include the publication sample column")
    pd.concat([coefficients.assign(result_type="continuous"), quantiles.assign(result_type="quantile")], ignore_index=True).to_csv(tables / "all_primary_results.csv", index=False)
