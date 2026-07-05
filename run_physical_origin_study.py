#!/usr/bin/env python3
"""
Physical-origin association tests for the vertically enhanced action-space regime.

The goal is not to fit a causal perturbation model.  Instead, this script tests
whether VEAR membership and vertical-action diagnostics vary systematically with
Galactocentric position and vertical phase-space coordinates in ways that are
qualitatively relevant to warp, flare, phase-spiral, bar/spiral, or
Sagittarius-like perturbations.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_high_latitude_dynamics import RO_KPC


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def bootstrap_fraction_ci(flags: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float | int]:
    flags = np.asarray(flags, dtype=float)
    flags = flags[np.isfinite(flags)]
    n = len(flags)
    if n == 0:
        return {"n": 0, "fraction": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        vals[i] = np.mean(rng.choice(flags, size=n, replace=True))
    return {
        "n": int(n),
        "fraction": float(np.mean(flags)),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


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
    # Match the earlier project convention for Galactocentric distance.
    galcen = coords.transform_to(Galactocentric(galcen_distance=RO_KPC * u.kpc))
    x = galcen.x.to_value(u.kpc)
    y = galcen.y.to_value(u.kpc)
    z = galcen.z.to_value(u.kpc)
    vx = galcen.v_x.to_value(u.km / u.s)
    vy = galcen.v_y.to_value(u.km / u.s)
    vz = galcen.v_z.to_value(u.km / u.s)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    v_r = (x * vx + y * vy) / r
    v_phi = (x * vy - y * vx) / r
    out["galcen_x_kpc_astropy"] = x
    out["galcen_y_kpc_astropy"] = y
    out["galcen_z_kpc_astropy"] = z
    out["galcen_vx_kms"] = vx
    out["galcen_vy_kms"] = vy
    out["galcen_vz_kms"] = vz
    out["R_gc_kpc_astropy"] = r
    out["phi_rad"] = phi
    out["phi_deg"] = np.degrees(phi)
    # Local azimuth centered on the Sun--Galactic-centre line.  The raw
    # Galactocentric azimuth wraps the nearby sample around +/-180 deg.
    out["phi_local_rad"] = np.arctan2(y, -x)
    out["phi_local_deg"] = np.degrees(out["phi_local_rad"])
    out["v_R_kms"] = v_r
    out["v_phi_kms"] = v_phi
    out["v_z_kms"] = vz
    out["vertical_phase_angle_rad"] = np.arctan2(vz / 20.0, z)
    out["vertical_phase_radius"] = np.sqrt(z**2 + (vz / 20.0) ** 2)
    out["abs_z_kpc"] = np.abs(z)
    out["outer_disk"] = out["R_gc_kpc"] >= np.nanmedian(out["R_gc_kpc"])
    out["above_plane"] = z > 0
    out["moving_up"] = vz > 0
    out["phase_quadrant"] = (
        np.where(z >= 0, "z+", "z-") + np.where(vz >= 0, "_vz+", "_vz-")
    )
    return out


def add_perturbation_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    phi = out["phi_local_rad"].to_numpy()
    r = out["R_gc_kpc"].to_numpy()
    z = out["galcen_z_kpc_astropy"].to_numpy()
    out["warp_basis_sin_phi"] = np.sin(phi)
    out["warp_basis_cos_phi"] = np.cos(phi)
    out["warp_basis_R_sin_phi"] = (r - 8.0) * np.sin(phi)
    out["warp_basis_R_cos_phi"] = (r - 8.0) * np.cos(phi)
    out["bar_m2_cos"] = np.cos(2.0 * phi)
    out["bar_m2_sin"] = np.sin(2.0 * phi)
    out["spiral_m4_cos"] = np.cos(4.0 * phi)
    out["spiral_m4_sin"] = np.sin(4.0 * phi)
    out["phase_spiral_sin"] = np.sin(out["vertical_phase_angle_rad"])
    out["phase_spiral_cos"] = np.cos(out["vertical_phase_angle_rad"])
    out["warp_like_sign"] = np.sign(z * np.sin(phi))
    return out


def logistic_test(df: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    data = df[["is_vear", *feature_cols]].replace([np.inf, -np.inf], np.nan).dropna()
    y = data["is_vear"].astype(int).to_numpy()
    x = data[feature_cols].to_numpy()
    row = {"test": label, "n": int(len(data)), "n_vear": int(y.sum()), "features": "|".join(feature_cols)}
    if len(np.unique(y)) < 2 or len(data) < 20:
        row.update({"status": "insufficient"})
        return row
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
    model.fit(x, y)
    coef = model.named_steps["logisticregression"].coef_[0]
    row.update(
        {
            "status": "ok",
            "cv_auc_mean": float(np.mean(auc)),
            "cv_auc_std": float(np.std(auc, ddof=1)),
            "max_abs_standardized_coef": float(np.max(np.abs(coef))),
            "coef_json": json.dumps({c: float(v) for c, v in zip(feature_cols, coef)}),
        }
    )
    return row


def binned_fraction(df: pd.DataFrame, col: str, bins: np.ndarray, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    values = df[col].to_numpy()
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (values >= bins[i]) & (values <= bins[i + 1])
        else:
            mask = (values >= bins[i]) & (values < bins[i + 1])
        ci = bootstrap_fraction_ci(df.loc[mask, "is_vear"].astype(float).to_numpy(), n_bootstrap, seed + i)
        rows.append(
            {
                "variable": col,
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "bin_center": float(0.5 * (bins[i] + bins[i + 1])),
                **ci,
            }
        )
    return pd.DataFrame(rows)


def quadrant_table(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx, (name, grp) in enumerate(df.groupby("phase_quadrant")):
        ci = bootstrap_fraction_ci(grp["is_vear"].astype(float).to_numpy(), n_bootstrap, seed + idx)
        rows.append({"variable": "phase_quadrant", "bin": name, **ci})
    return pd.DataFrame(rows)


def contingency_test(df: pd.DataFrame, row_col: str) -> dict:
    table = pd.crosstab(df[row_col], df["is_vear"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": row_col, "status": "insufficient"}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"test": row_col, "status": "ok", "chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def trend_test(df: pd.DataFrame, col: str) -> dict:
    data = df[[col, "is_vear"]].replace([np.inf, -np.inf], np.nan).dropna()
    x = data[col].to_numpy()
    y = data["is_vear"].astype(int).to_numpy()
    rho = stats.spearmanr(x, y)
    return {"test": f"spearman_{col}_vs_vear", "rho": float(rho.statistic), "p_value": float(rho.pvalue), "n": int(len(data))}


def compare_vertical_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["galcen_z_kpc_astropy", "abs_z_kpc", "v_z_kms", "vertical_phase_radius", "R_gc_kpc", "phi_deg"]:
        vear = df.loc[df["is_vear"], metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        other = df.loc[~df["is_vear"], metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vear) >= 3 and len(other) >= 3:
            mw = stats.mannwhitneyu(vear, other, alternative="two-sided")
            ks = stats.ks_2samp(vear, other)
        else:
            mw = ks = None
        rows.append(
            {
                "metric": metric,
                "vear_median": float(np.median(vear)) if len(vear) else math.nan,
                "non_vear_median": float(np.median(other)) if len(other) else math.nan,
                "median_difference": float(np.median(vear) - np.median(other)) if len(vear) and len(other) else math.nan,
                "mann_whitney_p": float(mw.pvalue) if mw is not None else math.nan,
                "ks_p": float(ks.pvalue) if ks is not None else math.nan,
            }
        )
    return pd.DataFrame(rows)


def save_figures(df: pd.DataFrame, frac_tables: dict[str, pd.DataFrame], outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    vear = df["is_vear"]

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(df.loc[~vear, "galcen_x_kpc"], df.loc[~vear, "galcen_y_kpc"], s=10, c="0.72", alpha=0.45, label="Other")
    sc = plt.scatter(
        df.loc[vear, "galcen_x_kpc"],
        df.loc[vear, "galcen_y_kpc"],
        c=df.loc[vear, "galcen_z_kpc_astropy"],
        s=24,
        cmap="coolwarm",
        alpha=0.78,
        label="VEAR",
    )
    plt.scatter([0], [0], marker="+", c="black", s=80, label="Galactic center")
    plt.xlabel("Galactocentric X [kpc]")
    plt.ylabel("Galactocentric Y [kpc]")
    plt.axis("equal")
    plt.legend(frameon=False, loc="best")
    cb = plt.colorbar(sc)
    cb.set_label("VEAR z [kpc]")
    plt.tight_layout()
    plt.savefig(figdir / "figure1_xy_vear_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.4, 5.2))
    plt.scatter(df.loc[~vear, "R_gc_kpc"], df.loc[~vear, "galcen_z_kpc_astropy"], s=10, c="0.72", alpha=0.45, label="Other")
    plt.scatter(df.loc[vear, "R_gc_kpc"], df.loc[vear, "galcen_z_kpc_astropy"], s=22, c="#e45756", alpha=0.72, label="VEAR")
    plt.axhline(0, color="0.35", lw=1)
    plt.xlabel(r"$R_{\rm GC}$ [kpc]")
    plt.ylabel("z [kpc]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "figure2_rz_warp_flare.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.4, 5.8))
    plt.scatter(df.loc[~vear, "galcen_z_kpc_astropy"], df.loc[~vear, "v_z_kms"], s=10, c="0.72", alpha=0.45, label="Other")
    plt.scatter(df.loc[vear, "galcen_z_kpc_astropy"], df.loc[vear, "v_z_kms"], s=22, c="#e45756", alpha=0.72, label="VEAR")
    plt.axhline(0, color="0.35", lw=1)
    plt.axvline(0, color="0.35", lw=1)
    plt.xlabel("z [kpc]")
    plt.ylabel(r"$v_z$ [km s$^{-1}$]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "figure3_phase_spiral_proxy.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, key, xlabel in [
        (axes[0], "R_gc_kpc", r"$R_{\rm GC}$ [kpc]"),
        (axes[1], "phi_local_deg", r"local $\phi$ [deg]"),
    ]:
        table = frac_tables[key]
        ax.errorbar(
            table["bin_center"],
            table["fraction"],
            yerr=[table["fraction"] - table["ci95_low"], table["ci95_high"] - table["fraction"]],
            fmt="o-",
            color="#4c78a8",
            capsize=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("VEAR fraction")
        ax.set_ylim(0, min(1, max(0.75, table["ci95_high"].max() + 0.08)))
    fig.tight_layout()
    fig.savefig(figdir / "figure4_fraction_position_trends.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(7.2, 4.8))
    q = frac_tables["phase_quadrant"].copy()
    plt.bar(q["bin"], q["fraction"], color="#59a14f", alpha=0.78)
    plt.errorbar(
        np.arange(len(q)),
        q["fraction"],
        yerr=[q["fraction"] - q["ci95_low"], q["ci95_high"] - q["fraction"]],
        fmt="none",
        ecolor="0.2",
        capsize=4,
    )
    plt.ylabel("VEAR fraction")
    plt.xlabel("Vertical phase-space quadrant")
    plt.tight_layout()
    plt.savefig(figdir / "figure5_phase_quadrant_fraction.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/orbital_family_analysis/orbit_family_catalog.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/physical_origin_study"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.catalog)
    if "is_vear" not in df:
        raise SystemExit("Input catalog must contain is_vear; run orbital family analysis first.")
    df["is_vear"] = as_bool(df["is_vear"])
    df["is_control_like"] = as_bool(df["is_control_like"])
    df = add_perturbation_proxies(add_phase_space(df))

    r_bins = np.quantile(df["R_gc_kpc"], np.linspace(0, 1, 7))
    r_bins[0] -= 1e-6
    r_bins[-1] += 1e-6
    phi_bins = np.linspace(
        float(np.floor(df["phi_local_deg"].min() / 10.0) * 10.0),
        float(np.ceil(df["phi_local_deg"].max() / 10.0) * 10.0),
        9,
    )
    z_bins = np.quantile(df["abs_z_kpc"], np.linspace(0, 1, 7))
    z_bins[0] -= 1e-6
    z_bins[-1] += 1e-6
    frac_tables = {
        "R_gc_kpc": binned_fraction(df, "R_gc_kpc", r_bins, args.bootstrap, args.seed),
        "phi_local_deg": binned_fraction(df, "phi_local_deg", phi_bins, args.bootstrap, args.seed + 100),
        "abs_z_kpc": binned_fraction(df, "abs_z_kpc", z_bins, args.bootstrap, args.seed + 200),
        "phase_quadrant": quadrant_table(df, args.bootstrap, args.seed + 300),
    }
    pd.concat(frac_tables.values(), ignore_index=True, sort=False).to_csv(args.outdir / "vdc_fraction_position_bins.csv", index=False)

    association_rows = []
    association_rows.extend(
        [
            logistic_test(df, ["R_gc_kpc"], "flare_radial_gradient"),
            logistic_test(df, ["phi_local_rad", "warp_basis_sin_phi", "warp_basis_cos_phi", "warp_basis_R_sin_phi", "warp_basis_R_cos_phi"], "warp_R_phi_modulation"),
            logistic_test(df, ["galcen_z_kpc_astropy", "v_z_kms", "phase_spiral_sin", "phase_spiral_cos", "vertical_phase_radius"], "phase_spiral_proxy"),
            logistic_test(df, ["bar_m2_cos", "bar_m2_sin", "Omega_R_over_Omega_phi_abs", "Omega_z_over_Omega_phi_abs"], "bar_m2_frequency_proxy"),
            logistic_test(df, ["spiral_m4_cos", "spiral_m4_sin", "R_gc_kpc", "phi_local_rad"], "spiral_m4_position_proxy"),
            logistic_test(df, ["outer_disk", "phi_local_rad", "galcen_z_kpc_astropy", "v_z_kms"], "sagittarius_outer_disk_proxy"),
        ]
    )
    for col in ["R_gc_kpc", "phi_local_deg", "abs_z_kpc", "galcen_z_kpc_astropy", "v_z_kms", "vertical_phase_radius"]:
        association_rows.append(trend_test(df, col))
    for col in ["phase_quadrant", "above_plane", "moving_up", "outer_disk", "warp_like_sign"]:
        association_rows.append(contingency_test(df, col))
    association = pd.DataFrame(association_rows)
    association.to_csv(args.outdir / "perturbation_association_statistics.csv", index=False)
    compare_vertical_kinematics(df).to_csv(args.outdir / "vertical_phase_space_statistics.csv", index=False)
    df.to_csv(args.outdir / "physical_origin_catalog.csv", index=False)

    save_figures(df, frac_tables, args.outdir)

    report = {
        "input_catalog": str(args.catalog),
        "n_clusters": int(len(df)),
        "n_vear": int(df["is_vear"].sum()),
        "n_non_vear": int((~df["is_vear"]).sum()),
        "tests": "Associational proxies for warp, flare, phase spiral, bar/spiral modulation, and Sagittarius-like outer-disk asymmetry.",
        "important_caveat": "These tests evaluate positional/phase-space associations only and do not establish causality.",
        "strongest_auc_tests": association[association.get("status", "ok").fillna("ok").eq("ok") & association["cv_auc_mean"].notna()].sort_values("cv_auc_mean", ascending=False).head(3).to_dict(orient="records")
        if "cv_auc_mean" in association
        else [],
    }
    (args.outdir / "physical_origin_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote physical-origin association study to {args.outdir}")


if __name__ == "__main__":
    main()
