#!/usr/bin/env python3
"""
Multimodality and topology tests for the VDC.

The tests use the same 1079-cluster action-space sample as the manuscript:
(log J_R, log J_z, log |L_z|), standardized on the full sample.  The null model
is a single continuous multivariate normal distribution with the observed mean,
covariance, and sample size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import diptest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


FEATURE_COLS = ["log_J_R", "log_J_z", "log_abs_L_z"]


def finite_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def add_action_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    jr = out["J_R_kpc_kms"].where(out["J_R_kpc_kms"] > 0)
    jz = out["J_z_kpc_kms"].where(out["J_z_kpc_kms"] > 0)
    lz = out["L_z_kpc_kms"].abs().where(out["L_z_kpc_kms"].abs() > 0)
    rg = out["R_gc_kpc"].where(out["R_gc_kpc"] > 0)
    out["eta_z"] = jz / (jr + jz)
    out["eta_z_Lz"] = jz / lz
    out["zmax_over_Rg"] = out["zmax_kpc"] / rg
    out["log_J_R"] = np.log10(jr)
    out["log_J_z"] = np.log10(jz)
    out["log_abs_L_z"] = np.log10(lz)
    if "is_vertical_action_outlier" in out:
        out["is_vertical_action_outlier"] = finite_bool(out["is_vertical_action_outlier"])
    return out


def identify_vertical_component(df: pd.DataFrame, labels: np.ndarray) -> int:
    work = df.copy()
    work["_label"] = labels
    rows = []
    for label, grp in work.groupby("_label"):
        rows.append(
            {
                "label": int(label),
                "n": int(len(grp)),
                "median_eta_z": float(grp["eta_z"].median()),
                "median_zmax_over_Rg": float(grp["zmax_over_Rg"].median()),
                "median_Delta_logJz": float(grp["Delta_logJz"].median()),
                "fraction_delta_outlier": float(grp["is_vertical_action_outlier"].mean()),
            }
        )
    keys = ["median_eta_z", "median_zmax_over_Rg", "median_Delta_logJz", "fraction_delta_outlier"]
    for row in rows:
        row["score"] = 0.0
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=float)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        if spread == 0:
            continue
        for idx, row in enumerate(rows):
            row["score"] += float((vals[idx] - np.nanmin(vals)) / spread)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return int(rows[0]["label"])


def empirical_pvalue(observed: float, null_values: np.ndarray, alternative: str = "greater") -> tuple[float, float, float, float]:
    null_values = np.asarray(null_values, dtype=float)
    if alternative == "greater":
        p = (1.0 + np.sum(null_values >= observed)) / (len(null_values) + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(null_values <= observed)) / (len(null_values) + 1.0)
    else:
        center = np.nanmean(null_values)
        p = (1.0 + np.sum(np.abs(null_values - center) >= abs(observed - center))) / (len(null_values) + 1.0)
    lo, hi = np.quantile(null_values, [0.025, 0.975])
    return float(p), float(lo), float(hi), float(np.nanmean(null_values))


def standardize_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / np.std(values, ddof=1)


def binned_kde(values: np.ndarray, bandwidth: float, n_grid: int = 512) -> tuple[np.ndarray, np.ndarray]:
    z = standardize_1d(values)
    lo = min(-4.5, float(np.min(z)) - 0.5)
    hi = max(4.5, float(np.max(z)) + 0.5)
    counts, edges = np.histogram(z, bins=n_grid, range=(lo, hi), density=False)
    dx = edges[1] - edges[0]
    grid = 0.5 * (edges[:-1] + edges[1:])
    density = counts.astype(float) / (len(z) * dx)
    sigma = max(float(bandwidth) / dx, 0.0)
    if sigma > 0:
        density = gaussian_filter1d(density, sigma=sigma, mode="nearest")
    return grid, density


def kde_mode_stats(values: np.ndarray, bandwidth: float, n_grid: int = 512) -> dict[str, float | int]:
    grid, density = binned_kde(values, bandwidth, n_grid=n_grid)
    max_density = float(np.max(density))
    prominence_floor = max(1e-12, 0.005 * max_density)
    peaks, props = find_peaks(density, prominence=prominence_floor)
    if len(peaks) == 0:
        return {
            "n_modes": 0,
            "total_prominence": 0.0,
            "max_prominence": 0.0,
            "density_ridge_separation": 0.0,
        }
    prominences = peak_prominences(density, peaks)[0]
    if len(peaks) >= 2:
        top = peaks[np.argsort(density[peaks])[-2:]]
        ridge_sep = abs(float(grid[top[1]] - grid[top[0]]))
    else:
        ridge_sep = 0.0
    return {
        "n_modes": int(len(peaks)),
        "total_prominence": float(np.sum(prominences)),
        "max_prominence": float(np.max(prominences)),
        "density_ridge_separation": ridge_sep,
    }


def critical_bandwidth(values: np.ndarray, n_grid: int = 512) -> float:
    if kde_mode_stats(values, 0.0, n_grid=n_grid)["n_modes"] <= 1:
        return 0.0
    lo, hi = 0.0, 2.0
    while kde_mode_stats(values, hi, n_grid=n_grid)["n_modes"] > 1 and hi < 8.0:
        hi *= 2.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if kde_mode_stats(values, mid, n_grid=n_grid)["n_modes"] > 1:
            lo = mid
        else:
            hi = mid
    return float(hi)


def projection_table(x: np.ndarray, df: pd.DataFrame) -> dict[str, np.ndarray]:
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=10, reg_covar=1e-6)
    labels = gmm.fit_predict(x)
    vdc_label = identify_vertical_component(df, labels)
    vdc_center = x[labels == vdc_label].mean(axis=0)
    other_center = x[labels != vdc_label].mean(axis=0)
    vdc_axis = vdc_center - other_center
    vdc_axis /= np.linalg.norm(vdc_axis)
    _, _, vh = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    axes = {
        "vdc_contrast_axis": vdc_axis,
        "pc1": vh[0],
        "pc2": vh[1],
        "pc3": vh[2],
        "log_J_R": np.array([1.0, 0.0, 0.0]),
        "log_J_z": np.array([0.0, 1.0, 0.0]),
        "log_abs_L_z": np.array([0.0, 0.0, 1.0]),
    }
    return axes


def zero_dim_persistence(x: np.ndarray, k: int = 20, density_k: int = 25) -> dict[str, float | int | list[tuple[float, float]]]:
    nbrs = NearestNeighbors(n_neighbors=max(k + 1, density_k + 1)).fit(x)
    distances, indices = nbrs.kneighbors(x)
    r_k = np.maximum(distances[:, density_k], 1e-12)
    density = -3.0 * np.log(r_k)
    n = len(x)
    graph = [set() for _ in range(n)]
    for i in range(n):
        for j in indices[i, 1 : k + 1]:
            j = int(j)
            graph[i].add(j)
            graph[j].add(i)

    order = np.argsort(-density)
    active = np.zeros(n, dtype=bool)
    parent = np.arange(n)
    birth = density.copy()
    pairs: list[tuple[float, float]] = []
    local_maxima = 0

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int, level: float) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if birth[ra] >= birth[rb]:
            parent[rb] = ra
            pairs.append((float(birth[rb]), float(level)))
        else:
            parent[ra] = rb
            pairs.append((float(birth[ra]), float(level)))

    for idx in order:
        idx = int(idx)
        active[idx] = True
        active_neighbors = [j for j in graph[idx] if active[j]]
        if not active_neighbors:
            local_maxima += 1
        for j in active_neighbors:
            union(idx, int(j), float(density[idx]))

    pers = np.array([b - d for b, d in pairs], dtype=float)
    positive = pers[pers > 0]
    return {
        "local_maxima": int(local_maxima),
        "persistence_pairs": pairs,
        "n_persistent_gt_0p5": int(np.sum(positive > 0.5)),
        "n_persistent_gt_1p0": int(np.sum(positive > 1.0)),
        "max_persistence": float(np.max(positive)) if len(positive) else 0.0,
        "second_persistence": float(np.partition(positive, -2)[-2]) if len(positive) >= 2 else 0.0,
        "total_persistence_gt_0p5": float(np.sum(positive[positive > 0.5])) if len(positive) else 0.0,
    }


def kde2d(xy: np.ndarray, grid_n: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin = xy.min(axis=0) - 0.35
    xmax, ymax = xy.max(axis=0) + 0.35
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    bandwidth = float(len(xy) ** (-1.0 / 6.0))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(xy)
    zz = np.exp(kde.score_samples(grid)).reshape(xx.shape)
    return xx, yy, zz


def morse_smale_proxy(zz: np.ndarray) -> dict[str, float | int | bool]:
    if not HAS_SKIMAGE:
        return {"available": False, "n_density_peaks": math.nan, "n_watershed_basins": math.nan}
    smooth = gaussian_filter(zz, sigma=1.2)
    coords = peak_local_max(smooth, min_distance=8, threshold_rel=0.08)
    markers = np.zeros_like(smooth, dtype=int)
    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx
    labels = watershed(-smooth, markers=markers, mask=smooth > np.quantile(smooth, 0.25))
    return {
        "available": True,
        "n_density_peaks": int(len(coords)),
        "n_watershed_basins": int(len(np.unique(labels[labels > 0]))),
    }


def save_figures(df: pd.DataFrame, x: np.ndarray, outdir: Path, topology: dict[str, float | int | list[tuple[float, float]]]) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    xy = x[:, :2]
    xx, yy, zz = kde2d(xy)

    plt.figure(figsize=(7.1, 5.5))
    plt.contourf(xx, yy, zz, levels=28, cmap="magma")
    plt.scatter(x[:, 0], x[:, 1], s=7, c="white", alpha=0.45, linewidths=0)
    plt.xlabel("standardized log J_R")
    plt.ylabel("standardized log J_z")
    cb = plt.colorbar()
    cb.set_label("2D KDE density")
    plt.tight_layout()
    plt.savefig(figdir / "figure1_kde_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.1, 5.5))
    levels = np.quantile(zz[zz > 0], [0.60, 0.72, 0.82, 0.90, 0.96, 0.985])
    plt.contour(xx, yy, zz, levels=levels, colors=["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b279a2", "#333333"], linewidths=1.4)
    plt.scatter(x[:, 0], x[:, 1], s=8, c="0.35", alpha=0.30, linewidths=0)
    plt.xlabel("standardized log J_R")
    plt.ylabel("standardized log J_z")
    plt.tight_layout()
    plt.savefig(figdir / "figure2_density_ridges.png", dpi=220)
    plt.close()

    pairs = np.array(topology.get("persistence_pairs", []), dtype=float)
    plt.figure(figsize=(5.8, 5.4))
    if len(pairs):
        plt.scatter(pairs[:, 0], pairs[:, 1], s=18, alpha=0.65, color="#4c78a8")
        lo = float(np.min(pairs))
        hi = float(np.max(pairs))
        plt.plot([lo, hi], [lo, hi], color="0.25", lw=1)
    plt.xlabel("birth log-density")
    plt.ylabel("death log-density")
    plt.tight_layout()
    plt.savefig(figdir / "figure3_persistence_diagram.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.6))
    if len(pairs):
        pers = np.sort(pairs[:, 0] - pairs[:, 1])[::-1]
        plt.plot(np.arange(1, len(pers) + 1), pers, marker="o", ms=3, lw=1.2, color="#e45756")
        plt.axhline(0.5, color="0.45", ls="--", lw=1)
        plt.axhline(1.0, color="0.2", ls=":", lw=1)
    plt.xlabel("persistence rank")
    plt.ylabel("log-density persistence")
    plt.tight_layout()
    plt.savefig(figdir / "figure4_topological_persistence.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/vertical_action_outliers/vertical_action_catalog.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/multimodality_vdc_test"))
    parser.add_argument("--n-mc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = add_action_metrics(pd.read_csv(args.catalog))
    features = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[features.index].reset_index(drop=True)
    features = features.reset_index(drop=True)
    if len(df) != 1079:
        raise ValueError(f"Expected 1079 clusters, found {len(df)}")

    scaler = StandardScaler()
    x = scaler.fit_transform(features.to_numpy())
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    axes = projection_table(x, df)
    rng = np.random.default_rng(args.seed)

    observed_rows = []
    null_stats: dict[str, dict[str, list[float]]] = {
        name: {"dip": [], "critical_bandwidth": [], "n_modes_scott": [], "total_prominence": [], "max_prominence": []}
        for name in axes
    }

    scott_h = len(x) ** (-1.0 / 5.0)
    for name, axis in axes.items():
        proj = x @ axis
        dip = float(diptest.dipstat(proj))
        cbw = critical_bandwidth(proj)
        mode_stats = kde_mode_stats(proj, scott_h)
        observed_rows.append(
            {
                "projection": name,
                "dip_statistic": dip,
                "critical_bandwidth": cbw,
                "scott_bandwidth": scott_h,
                "n_modes_scott": mode_stats["n_modes"],
                "total_prominence": mode_stats["total_prominence"],
                "max_prominence": mode_stats["max_prominence"],
                "density_ridge_separation": mode_stats["density_ridge_separation"],
            }
        )

    observed_topology = zero_dim_persistence(x)
    xx, yy, zz = kde2d(x[:, :2])
    morse = morse_smale_proxy(zz)

    topology_null = {
        "local_maxima": [],
        "n_persistent_gt_0p5": [],
        "n_persistent_gt_1p0": [],
        "max_persistence": [],
        "second_persistence": [],
        "total_persistence_gt_0p5": [],
    }
    morse_null = {"n_density_peaks": [], "n_watershed_basins": []}

    for i in range(args.n_mc):
        sample = rng.multivariate_normal(mean, cov, size=len(x))
        for name, axis in axes.items():
            proj = sample @ axis
            null_stats[name]["dip"].append(float(diptest.dipstat(proj)))
            null_stats[name]["critical_bandwidth"].append(critical_bandwidth(proj))
            mode_stats = kde_mode_stats(proj, scott_h)
            null_stats[name]["n_modes_scott"].append(float(mode_stats["n_modes"]))
            null_stats[name]["total_prominence"].append(float(mode_stats["total_prominence"]))
            null_stats[name]["max_prominence"].append(float(mode_stats["max_prominence"]))
        topo = zero_dim_persistence(sample)
        for key in topology_null:
            topology_null[key].append(float(topo[key]))
        if HAS_SKIMAGE and (i < args.n_mc):
            _, _, null_zz = kde2d(sample[:, :2], grid_n=120)
            null_morse = morse_smale_proxy(null_zz)
            morse_null["n_density_peaks"].append(float(null_morse["n_density_peaks"]))
            morse_null["n_watershed_basins"].append(float(null_morse["n_watershed_basins"]))
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"Completed {i + 1}/{args.n_mc} null Monte Carlo samples", flush=True)

    observed = pd.DataFrame(observed_rows)
    dip_rows = []
    silverman_rows = []
    multimodality_rows = []
    for row in observed.to_dict(orient="records"):
        name = row["projection"]
        dip_p, dip_lo, dip_hi, dip_mean = empirical_pvalue(row["dip_statistic"], np.array(null_stats[name]["dip"]))
        dip_rows.append(
            {
                "projection": name,
                "test": "Hartigan dip test",
                "statistic": row["dip_statistic"],
                "null_mean": dip_mean,
                "null_ci95_low": dip_lo,
                "null_ci95_high": dip_hi,
                "mc_p_value": dip_p,
                "n_mc": args.n_mc,
            }
        )
        sil_p, sil_lo, sil_hi, sil_mean = empirical_pvalue(row["critical_bandwidth"], np.array(null_stats[name]["critical_bandwidth"]))
        silverman_rows.append(
            {
                "projection": name,
                "test": "Silverman critical bandwidth",
                "critical_bandwidth": row["critical_bandwidth"],
                "null_mean": sil_mean,
                "null_ci95_low": sil_lo,
                "null_ci95_high": sil_hi,
                "mc_p_value": sil_p,
                "n_mc": args.n_mc,
            }
        )
        for stat, label in [
            ("n_modes_scott", "KDE mode count at Scott bandwidth"),
            ("total_prominence", "Excess-mass proxy: total peak prominence"),
            ("max_prominence", "Excess-mass proxy: max peak prominence"),
        ]:
            p, lo, hi, avg = empirical_pvalue(float(row[stat]), np.array(null_stats[name][stat]))
            multimodality_rows.append(
                {
                    "projection": name,
                    "test": label,
                    "statistic": row[stat],
                    "null_mean": avg,
                    "null_ci95_low": lo,
                    "null_ci95_high": hi,
                    "mc_p_value": p,
                    "n_mc": args.n_mc,
                }
            )

    topology_rows = []
    for stat in topology_null:
        p, lo, hi, avg = empirical_pvalue(float(observed_topology[stat]), np.array(topology_null[stat]))
        topology_rows.append(
            {
                "analysis": "0D density persistent homology on 3D kNN graph",
                "statistic_name": stat,
                "observed": observed_topology[stat],
                "null_mean": avg,
                "null_ci95_low": lo,
                "null_ci95_high": hi,
                "mc_p_value": p,
                "n_mc": args.n_mc,
            }
        )
    for stat in morse_null:
        if len(morse_null[stat]):
            p, lo, hi, avg = empirical_pvalue(float(morse[stat]), np.array(morse_null[stat]))
            topology_rows.append(
                {
                    "analysis": "2D KDE watershed Morse-Smale proxy",
                    "statistic_name": stat,
                    "observed": morse[stat],
                    "null_mean": avg,
                    "null_ci95_low": lo,
                    "null_ci95_high": hi,
                    "mc_p_value": p,
                    "n_mc": args.n_mc,
                }
            )
    topology_rows.append(
        {
            "analysis": "Morse-Smale availability",
            "statistic_name": "skimage_watershed_proxy_available",
            "observed": bool(HAS_SKIMAGE),
            "null_mean": math.nan,
            "null_ci95_low": math.nan,
            "null_ci95_high": math.nan,
            "mc_p_value": math.nan,
            "n_mc": args.n_mc,
        }
    )

    pd.DataFrame(dip_rows).to_csv(args.outdir / "dip_test.csv", index=False)
    pd.DataFrame(silverman_rows).to_csv(args.outdir / "silverman_test.csv", index=False)
    pd.DataFrame(multimodality_rows).to_csv(args.outdir / "multimodality_results.csv", index=False)
    pd.DataFrame(topology_rows).to_csv(args.outdir / "topology_summary.csv", index=False)

    report = {
        "input_catalog": str(args.catalog),
        "n_clusters": int(len(x)),
        "n_mc": int(args.n_mc),
        "null_model": "single multivariate normal preserving observed standardized action-space mean, covariance, and sample size",
        "primary_projection": "vdc_contrast_axis",
        "morse_smale_available": bool(HAS_SKIMAGE),
        "note": "Monte Carlo p-values use (1 + number of null statistics at least as extreme as observed)/(1 + n_mc).",
    }
    (args.outdir / "multimodality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(df, x, args.outdir, observed_topology)
    print(json.dumps(report, indent=2))
    print(f"Wrote multimodality and topology results to {args.outdir}")


if __name__ == "__main__":
    main()
