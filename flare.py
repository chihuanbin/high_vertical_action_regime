from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import resolve_path


def _design(data: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    radius = data["R_kpc"].to_numpy() - 8.0
    age = data["logAge50"].to_numpy() - 8.0
    phi = data["phi_rad"].to_numpy()
    scale_columns = [np.ones(len(data))]
    if model in {"F1", "F2", "F3"}:
        scale_columns.append(radius)
    if model in {"F2", "F3"}:
        scale_columns.append(age)
    warp_columns = [] if model != "F3" else [radius * np.sin(phi), radius * np.cos(phi)]
    return np.column_stack(scale_columns), np.column_stack(warp_columns) if warp_columns else np.empty((len(data), 0))


def fit_vertical_profile(data: pd.DataFrame, model: str, profile: str) -> dict:
    scale_x, warp_x = _design(data, model)
    z = data["z_kpc"].to_numpy()
    n_scale = scale_x.shape[1]

    def objective(parameters):
        h = np.exp(scale_x @ parameters[:n_scale])
        warp = warp_x @ parameters[n_scale:] if warp_x.shape[1] else 0.0
        residual = z - warp
        if profile == "exponential":
            log_density = -np.log(2 * h) - np.abs(residual) / h
        elif profile == "sech2":
            log_density = -np.log(4 * h) + 2 * np.log(1 / np.cosh(np.clip(residual / (2 * h), -50, 50)))
        else:
            raise ValueError(profile)
        return -float(np.sum(log_density))

    initial = np.zeros(n_scale + warp_x.shape[1])
    initial[0] = np.log(max(np.median(np.abs(z)) / np.log(2), 0.01))
    fit = minimize(objective, initial, method="BFGS")
    k = len(initial)
    return {"model": model, "profile": profile, "converged": bool(fit.success), "n": len(data),
        "negative_log_likelihood": fit.fun, "aic": 2 * k + 2 * fit.fun,
        "bic": k * np.log(len(data)) + 2 * fit.fun,
        **{f"parameter_{i}": value for i, value in enumerate(fit.x)}}


def run_flare(config: dict) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = pd.read_csv(tables / "analysis_catalogue.csv")
    data = table[table["potential"] == config["potentials"]["primary"]].dropna(subset=["R_kpc", "z_kpc", "logAge50", "phi_rad"])
    if len(data) != config["project"]["expected_sample_size"]:
        raise RuntimeError("Flare models require the complete primary-potential sample")
    rows = [fit_vertical_profile(data, model, profile) for profile in ["exponential", "sech2"] for model in ["F0", "F1", "F2", "F3"]]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "flare_model_comparison.csv", index=False)
    comparison.to_csv(tables / "flare_model_parameters.csv", index=False)
    return {"models": len(comparison), "converged": int(comparison["converged"].sum())}
