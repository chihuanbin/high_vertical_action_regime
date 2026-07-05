#!/usr/bin/env python3
"""
Publication-grade v2 pipeline for Vertical-Action Outliers in HUNT24 open clusters.

This script intentionally drops the earlier high-latitude/Lambda/DSI framing.
It defines outliers by excess vertical action relative to an age-R_GC disk
baseline, then tests whether the excess grows with age.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from scipy import stats

from run_high_latitude_dynamics import (
    RO_KPC,
    VO_KMS,
    SOLAR_MOTION,
    SampleCuts,
    base_quality_mask,
    load_clusters,
)


@dataclass(frozen=True)
class OutlierConfig:
    min_members: int = 30
    max_age_width_dex: float = 0.4
    max_frac_parallax_error: float = 0.1
    min_rv_members: int = 3
    min_prob_j: float = 0.5
    outlier_delta_dex: float = 0.3
    outlier_top_fraction: float = 0.05
    outlier_top_n: int = 50
    control_delta_dex: float = 0.15
    sigma_threshold: float = 1.0
    clip_delta_dex: float = 0.3
    n_bootstrap: int = 10000
    seed: int = 42


def load_galpy():
    try:
        from galpy.actionAngle import actionAngleStaeckel
        from galpy.orbit import Orbit
        from galpy.potential import MWPotential2014
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install galpy from requirements.txt before running v2.") from exc
    return Orbit, actionAngleStaeckel, MWPotential2014


def add_phase_space(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coords = SkyCoord(
        ra=out["RAdeg"].to_numpy() * u.deg,
        dec=out["DEdeg"].to_numpy() * u.deg,
        distance=out["distance_kpc"].to_numpy() * u.kpc,
        pm_ra_cosdec=out["pmRA"].to_numpy() * u.mas / u.yr,
        pm_dec=out["pmDE"].to_numpy() * u.mas / u.yr,
        radial_velocity=out["RV"].to_numpy() * u.km / u.s,
        frame="icrs",
    )
    galcen = coords.transform_to(Galactocentric(galcen_distance=RO_KPC * u.kpc))
    out["galcen_x_kpc"] = galcen.x.to_value(u.kpc)
    out["galcen_y_kpc"] = galcen.y.to_value(u.kpc)
    out["galcen_z_kpc"] = galcen.z.to_value(u.kpc)
    out["R_gc_kpc"] = np.hypot(out["galcen_x_kpc"], out["galcen_y_kpc"])
    out["abs_Z_kpc"] = out["galcen_z_kpc"].abs()
    return out


def compute_actions(df: pd.DataFrame) -> pd.DataFrame:
    Orbit, actionAngleStaeckel, MWPotential2014 = load_galpy()
    action_finder = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    records = []
    for _, row in df.iterrows():
        orbit = Orbit(
            vxvv=[
                row["RAdeg"],
                row["DEdeg"],
                row["distance_kpc"],
                row["pmRA"],
                row["pmDE"],
                row["RV"],
            ],
            radec=True,
            ro=RO_KPC,
            vo=VO_KMS,
            solarmotion=SOLAR_MOTION,
        )
        jr, lz, jz = action_finder(orbit)
        try:
            zmax = orbit.zmax(pot=MWPotential2014, analytic=True, use_physical=True)
        except Exception:
            zmax = np.nan
        records.append(
            {
                "J_R_kpc_kms": float(np.atleast_1d(jr)[0] * RO_KPC * VO_KMS),
                "L_z_kpc_kms": float(np.atleast_1d(lz)[0] * RO_KPC * VO_KMS),
                "J_z_kpc_kms": float(np.atleast_1d(jz)[0] * RO_KPC * VO_KMS),
                "zmax_kpc": float(np.atleast_1d(zmax)[0]) if np.isfinite(zmax) else np.nan,
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def design_matrix(df: pd.DataFrame, r0: float | None = None) -> tuple[np.ndarray, float]:
    if r0 is None:
        r0 = float(np.nanmedian(df["R_gc_kpc"]))
    x = np.column_stack(
        [
            np.ones(len(df)),
            df["logAge50"].to_numpy() - 8.0,
            df["R_gc_kpc"].to_numpy() - r0,
        ]
    )
    return x, r0


def fit_baseline(df: pd.DataFrame, clip_delta: float, iterations: int = 2) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    valid = (
        (out["J_z_kpc_kms"] > 0)
        & out["logAge50"].notna()
        & out["R_gc_kpc"].notna()
        & np.isfinite(out["J_z_kpc_kms"])
    )
    train_mask = valid.copy()
    r0 = None
    coeff = np.full(3, np.nan)
    for _ in range(iterations):
        train = out[train_mask].copy()
        x, r0 = design_matrix(train, r0=r0)
        y = np.log10(train["J_z_kpc_kms"].to_numpy())
        coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
        x_all, _ = design_matrix(out, r0=r0)
        pred = x_all @ coeff
        delta = np.log10(out["J_z_kpc_kms"].where(out["J_z_kpc_kms"] > 0)) - pred
        train_mask = valid & (delta <= clip_delta)

    x_all, _ = design_matrix(out, r0=r0)
    out["logJz_baseline"] = x_all @ coeff
    out["Delta_logJz"] = np.log10(out["J_z_kpc_kms"].where(out["J_z_kpc_kms"] > 0)) - out["logJz_baseline"]
    residual = out.loc[train_mask, "Delta_logJz"].dropna()
    scatter = float(np.nanstd(residual, ddof=3))
    out["outlier_sigma"] = out["Delta_logJz"] / scatter if scatter > 0 else np.nan
    return out, {
        "status": "ok",
        "n_total_valid": int(valid.sum()),
        "n_train_after_clip": int(train_mask.sum()),
        "r0_kpc": float(r0),
        "intercept": float(coeff[0]),
        "slope_logAge": float(coeff[1]),
        "slope_R_gc": float(coeff[2]),
        "scatter_dex": scatter,
        "clip_delta_dex": clip_delta,
    }


def classify_outliers(df: pd.DataFrame, cfg: OutlierConfig) -> pd.DataFrame:
    out = df.copy()
    out["is_vertical_action_outlier"] = out["Delta_logJz"] > cfg.outlier_delta_dex
    out["is_vertical_action_outlier_sigma"] = out["outlier_sigma"] > cfg.sigma_threshold
    finite_delta = out["Delta_logJz"].replace([np.inf, -np.inf], np.nan)
    top_fraction_threshold = float(finite_delta.quantile(1.0 - cfg.outlier_top_fraction))
    top_n_count = min(cfg.outlier_top_n, int(finite_delta.notna().sum()))
    top_n_threshold = float(finite_delta.nlargest(top_n_count).min())
    out["is_vertical_action_outlier_top_fraction"] = out["Delta_logJz"] >= top_fraction_threshold
    out["is_vertical_action_outlier_top_n"] = out["Delta_logJz"] >= top_n_threshold
    out["is_control_like"] = out["Delta_logJz"].abs() <= cfg.control_delta_dex
    out.attrs["top_fraction_threshold"] = top_fraction_threshold
    out.attrs["top_n_threshold"] = top_n_threshold
    return out


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        ok &= np.isfinite(yerr) & (yerr > 0)
    x = x[ok]
    y = y[ok]
    if yerr is None:
        w = np.ones_like(y)
    else:
        w = 1.0 / np.square(yerr[ok])
    if len(y) < 4:
        return {"status": "insufficient", "n": int(len(y))}
    x0 = x - np.average(x, weights=w)
    design = np.column_stack([np.ones_like(x0), x0])
    sw = np.sqrt(w)
    coeff, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    resid = y - design @ coeff
    rss = float(np.sum(w * resid**2))
    sigma2 = rss / max(len(y) - 2, 1)
    cov = sigma2 * np.linalg.inv((design * w[:, None]).T @ design)
    bic = len(y) * np.log(max(rss / len(y), 1e-12)) + 2 * np.log(len(y))
    return {
        "status": "ok",
        "n": int(len(y)),
        "intercept": float(coeff[0]),
        "slope": float(coeff[1]),
        "slope_stderr": float(math.sqrt(cov[1, 1])),
        "rss": rss,
        "bic": float(bic),
        "x_center": float(np.average(x, weights=w)),
    }


def constant_fit(y: np.ndarray, yerr: np.ndarray | None = None) -> dict:
    ok = np.isfinite(y)
    if yerr is not None:
        ok &= np.isfinite(yerr) & (yerr > 0)
    y = y[ok]
    if yerr is None:
        w = np.ones_like(y)
    else:
        w = 1.0 / np.square(yerr[ok])
    if len(y) < 4:
        return {"status": "insufficient", "n": int(len(y))}
    mu = float(np.average(y, weights=w))
    resid = y - mu
    rss = float(np.sum(w * resid**2))
    bic = len(y) * np.log(max(rss / len(y), 1e-12)) + np.log(len(y))
    return {"status": "ok", "n": int(len(y)), "mu": mu, "rss": rss, "bic": float(bic)}


def bootstrap_age_slope(outliers: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    data = outliers[["logAge50", "Delta_logJz"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4:
        return {"status": "insufficient", "n": int(len(data))}
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_bootstrap)
    x = data["logAge50"].to_numpy()
    y = data["Delta_logJz"].to_numpy()
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(data), len(data))
        slopes[i] = stats.theilslopes(y[idx], x[idx]).slope
    slope_obs = stats.theilslopes(y, x).slope
    spearman = stats.spearmanr(x, y)
    return {
        "status": "ok",
        "n": int(len(data)),
        "n_bootstrap": int(n_bootstrap),
        "theil_sen_slope": float(slope_obs),
        "slope_p16": float(np.nanpercentile(slopes, 16)),
        "slope_p50": float(np.nanpercentile(slopes, 50)),
        "slope_p84": float(np.nanpercentile(slopes, 84)),
        "fraction_slope_positive": float(np.nanmean(slopes > 0)),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def model_comparison(outliers: pd.DataFrame) -> dict:
    y = outliers["Delta_logJz"].to_numpy()
    x = outliers["logAge50"].to_numpy()
    m0 = constant_fit(y)
    m1 = weighted_linear_fit(x, y)
    if m0.get("status") != "ok" or m1.get("status") != "ok":
        return {"M0": m0, "M1": m1, "status": "insufficient"}
    delta_bic = m1["bic"] - m0["bic"]
    bf_m0_m1 = math.exp(0.5 * delta_bic)
    return {
        "status": "ok",
        "M0_constant": m0,
        "M1_age_slope": m1,
        "delta_BIC_M1_minus_M0": float(delta_bic),
        "bayes_factor_M0_over_M1_BIC_approx": float(bf_m0_m1),
    }


def representative_orbits(df: pd.DataFrame, outdir: Path, n_outliers: int = 3, n_controls: int = 3) -> None:
    Orbit, _, MWPotential2014 = load_galpy()
    selected = pd.concat(
        [
            df[df["is_vertical_action_outlier_top_n"]].sort_values("Delta_logJz", ascending=False).head(n_outliers),
            df[df["is_control_like"]].sort_values("Delta_logJz").head(n_controls),
        ]
    )
    if selected.empty:
        return
    times = np.linspace(0, 1.0, 401) * u.Gyr
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for _, row in selected.iterrows():
        orbit = Orbit(
            vxvv=[row.RAdeg, row.DEdeg, row.distance_kpc, row.pmRA, row.pmDE, row.RV],
            radec=True,
            ro=RO_KPC,
            vo=VO_KMS,
            solarmotion=SOLAR_MOTION,
        )
        orbit.integrate(times, MWPotential2014)
        label = str(row.Name).strip()
        color = "#c43c39" if row.is_vertical_action_outlier_top_n else "#2878b5"
        alpha = 0.85 if row.is_vertical_action_outlier_top_n else 0.55
        axes[0].plot(orbit.R(times, use_physical=True), orbit.z(times, use_physical=True), color=color, alpha=alpha, lw=1.1)
        axes[1].plot(times.to_value(u.Gyr), orbit.z(times, use_physical=True), color=color, alpha=alpha, lw=1.1, label=label)
    axes[0].set_xlabel(r"$R$ [kpc]")
    axes[0].set_ylabel(r"$z$ [kpc]")
    axes[1].set_xlabel("Look-forward time [Gyr]")
    axes[1].set_ylabel(r"$z$ [kpc]")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_representative_orbits.png", dpi=220)
    plt.close(fig)


def save_figures(df: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = df[df["is_vertical_action_outlier_top_n"]]
    ctrl = df[df["is_control_like"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    sc = axes[0].scatter(df["logAge50"], np.log10(df["J_z_kpc_kms"]), c=df["R_gc_kpc"], s=13, cmap="viridis", alpha=0.72)
    axes[0].set_xlabel(r"$\log_{10}({\rm age/yr})$")
    axes[0].set_ylabel(r"$\log_{10} J_z$ [kpc km s$^{-1}$]")
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label(r"$R_{\rm GC}$ [kpc]")
    axes[1].scatter(df["logAge50"], df["Delta_logJz"], s=12, c="0.65", alpha=0.65, label="Open clusters")
    axes[1].scatter(out["logAge50"], out["Delta_logJz"], s=28, c="#c43c39", label="Vertical-action outliers")
    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[1].axhline(0.3, color="#c43c39", ls="--", lw=1.0)
    axes[1].set_xlabel(r"$\log_{10}({\rm age/yr})$")
    axes[1].set_ylabel(r"$\Delta \log J_z$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_jz_age_corrected.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(7, 4.8))
    plt.hist(df["Delta_logJz"].dropna(), bins=36, color="0.72", alpha=0.85, label="All quality OCs")
    plt.hist(out["Delta_logJz"].dropna(), bins=12, color="#c43c39", alpha=0.85, label="Outliers")
    plt.axvline(0.3, color="#c43c39", ls="--", lw=1.2)
    plt.axvline(0, color="0.25", lw=0.8)
    plt.xlabel(r"$\Delta \log J_z$")
    plt.ylabel("Number of clusters")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig2_delta_logjz_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.scatter(ctrl["logAge50"], ctrl["Delta_logJz"], s=18, color="#2878b5", alpha=0.45, label="Control-like")
    plt.scatter(out["logAge50"], out["Delta_logJz"], s=38, color="#c43c39", alpha=0.9, label="Outliers")
    if len(out) >= 4:
        fit = weighted_linear_fit(out["logAge50"].to_numpy(), out["Delta_logJz"].to_numpy())
        xs = np.linspace(out["logAge50"].min(), out["logAge50"].max(), 100)
        ys = fit["intercept"] + fit["slope"] * (xs - fit["x_center"])
        plt.plot(xs, ys, color="#c43c39", lw=1.6, label="Outlier age slope")
        plt.axhline(out["Delta_logJz"].mean(), color="0.2", ls="--", lw=1.2, label="Constant model")
    plt.axhline(0.3, color="#c43c39", ls=":", lw=1.0)
    plt.xlabel(r"$\log_{10}({\rm age/yr})$")
    plt.ylabel(r"$\Delta \log J_z$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig3_outlier_delta_logjz_vs_age.png", dpi=220)
    plt.close()

    representative_orbits(df, figdir)


def write_outputs(df: pd.DataFrame, cfg: OutlierConfig, baseline: dict, outdir: Path) -> None:
    outliers = df[df["is_vertical_action_outlier_top_n"]].copy()
    threshold_outliers = df[df["is_vertical_action_outlier"]].copy()
    controls = df[df["is_control_like"]].copy()
    boot = bootstrap_age_slope(outliers, cfg.n_bootstrap, cfg.seed)
    comparison = model_comparison(outliers)
    summary = {
        "config": cfg.__dict__,
        "counts": {
            "quality_open_clusters": int(len(df)),
            "vertical_action_outliers_delta": int(df["is_vertical_action_outlier"].sum()),
            "vertical_action_outliers_sigma": int(df["is_vertical_action_outlier_sigma"].sum()),
            "vertical_action_outliers_top_fraction": int(df["is_vertical_action_outlier_top_fraction"].sum()),
            "vertical_action_outliers_top_n": int(df["is_vertical_action_outlier_top_n"].sum()),
            "control_like": int(df["is_control_like"].sum()),
        },
        "outlier_thresholds": {
            "delta_fixed_dex": cfg.outlier_delta_dex,
            "top_fraction": cfg.outlier_top_fraction,
            "top_fraction_delta_threshold": df.attrs.get("top_fraction_threshold"),
            "top_n": cfg.outlier_top_n,
            "top_n_delta_threshold": df.attrs.get("top_n_threshold"),
        },
        "baseline": baseline,
        "outlier_age_slope_bootstrap": boot,
        "model_comparison": comparison,
        "notes": [
            "Main publication-grade age-slope test uses top-N Vertical-Action Outliers.",
            "The fixed Delta_logJz > outlier_delta_dex population is reported separately as a broad excess sample.",
            "Bayes factor is a BIC approximation for M0 constant excess versus M1 age slope.",
            "Lambda and DSI are intentionally not computed in the v2 pipeline.",
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "vertical_action_catalog.csv", index=False)
    outliers.to_csv(outdir / "vertical_action_outliers.csv", index=False)
    threshold_outliers.to_csv(outdir / "vertical_action_outliers_delta_threshold.csv", index=False)
    controls.to_csv(outdir / "control_like_clusters.csv", index=False)
    (outdir / "summary_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figures(df, outdir)
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps({"baseline": baseline, "model_comparison": comparison}, indent=2))
    print(f"Wrote v2 results to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=Path("hunt24/clusters.dat"))
    parser.add_argument("--outdir", type=Path, default=Path("results/vertical_action_outliers"))
    parser.add_argument("--min-members", type=int, default=30)
    parser.add_argument("--max-age-width-dex", type=float, default=0.4)
    parser.add_argument("--max-frac-parallax-error", type=float, default=0.1)
    parser.add_argument("--min-rv-members", type=int, default=3)
    parser.add_argument("--min-prob-j", type=float, default=0.5)
    parser.add_argument("--outlier-delta-dex", type=float, default=0.3)
    parser.add_argument("--outlier-top-fraction", type=float, default=0.05)
    parser.add_argument("--outlier-top-n", type=int, default=50)
    parser.add_argument("--control-delta-dex", type=float, default=0.15)
    parser.add_argument("--sigma-threshold", type=float, default=1.0)
    parser.add_argument("--clip-delta-dex", type=float, default=0.3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-moving-groups", action="store_true")
    parser.add_argument("--max-clusters", type=int, default=None, help="Smoke-test limit after quality cuts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OutlierConfig(
        min_members=args.min_members,
        max_age_width_dex=args.max_age_width_dex,
        max_frac_parallax_error=args.max_frac_parallax_error,
        min_rv_members=args.min_rv_members,
        min_prob_j=args.min_prob_j,
        outlier_delta_dex=args.outlier_delta_dex,
        outlier_top_fraction=args.outlier_top_fraction,
        outlier_top_n=args.outlier_top_n,
        control_delta_dex=args.control_delta_dex,
        sigma_threshold=args.sigma_threshold,
        clip_delta_dex=args.clip_delta_dex,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    cuts = SampleCuts(
        min_members=cfg.min_members,
        max_age_width_dex=cfg.max_age_width_dex,
        max_frac_parallax_error=cfg.max_frac_parallax_error,
        min_rv_members=cfg.min_rv_members,
        min_prob_j=cfg.min_prob_j,
        include_moving_groups=args.include_moving_groups,
    )
    clusters = load_clusters(args.clusters)
    sample = clusters[base_quality_mask(clusters, cuts, require_rv=True)].copy()
    if args.max_clusters is not None:
        sample = sample.head(args.max_clusters).copy()
    sample = add_phase_space(sample)
    sample = compute_actions(sample)
    sample, baseline = fit_baseline(sample, clip_delta=cfg.clip_delta_dex)
    sample = classify_outliers(sample, cfg)
    write_outputs(sample, cfg, baseline, args.outdir)


if __name__ == "__main__":
    main()
