#!/usr/bin/env python3
"""
Physical validation of the vertically enhanced action-space regime.

This script computes orbital frequencies with the same galpy/MWPotential2014
conventions as the action calculation and tests whether the VEAR occupies a
distinct orbital-family regime relative to strict matched controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))
os.environ.setdefault("GALPY_CONFIG_FILE", str((Path("results") / ".galpyrc").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from run_high_latitude_dynamics import RO_KPC, SOLAR_MOTION, VO_KMS


ORBIT_METRICS = [
    "Omega_R",
    "Omega_phi",
    "Omega_z",
    "Omega_z_over_Omega_phi_abs",
    "Omega_R_over_Omega_phi_abs",
    "Jz_over_JR_plus_Jz",
    "Jz_over_abs_Lz",
    "JR_over_abs_Lz",
    "zmax_over_Rg",
    "orbit_eccentricity",
    "rperi_kpc",
    "rap_kpc",
    "Rg_kpc",
]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def load_high_label(report_path: Path, fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    label = report.get("high_vertical_gmm_label")
    if label is None:
        raise SystemExit("summary_report.json has no high_vertical_gmm_label; pass --high-label explicitly.")
    return int(label)


def compute_frequencies(df: pd.DataFrame, cache_path: Path, overwrite: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not overwrite:
        freq = pd.read_csv(cache_path)
        return df.merge(freq, on=["ID", "Name"], how="left")

    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    action_finder = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    records = []
    for idx, row in df.iterrows():
        orbit = Orbit(
            vxvv=[row.RAdeg, row.DEdeg, row.distance_kpc, row.pmRA, row.pmDE, row.RV],
            radec=True,
            ro=RO_KPC,
            vo=VO_KMS,
            solarmotion=SOLAR_MOTION,
        )
        try:
            jr, lz, jz, omega_r, omega_phi, omega_z = action_finder.actionsFreqs(orbit)
            rec = {
                "ID": row.ID,
                "Name": row.Name,
                "Omega_R": float(np.atleast_1d(omega_r)[0]),
                "Omega_phi": float(np.atleast_1d(omega_phi)[0]),
                "Omega_z": float(np.atleast_1d(omega_z)[0]),
            }
        except Exception:
            rec = {"ID": row.ID, "Name": row.Name, "Omega_R": np.nan, "Omega_phi": np.nan, "Omega_z": np.nan}
        records.append(rec)
        if (idx + 1) % 100 == 0:
            print(f"Computed frequencies for {idx + 1}/{len(df)} clusters", flush=True)

    freq = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    freq.to_csv(cache_path, index=False)
    return df.merge(freq, on=["ID", "Name"], how="left")


def add_orbit_family_metrics(df: pd.DataFrame, high_label: int) -> pd.DataFrame:
    out = df.copy()
    out["is_control_like"] = as_bool(out["is_control_like"])
    out["is_vear"] = out["gmm_label"].astype(int) == int(high_label)
    out["abs_L_z_kpc_kms"] = out["L_z_kpc_kms"].abs()
    out["Rg_kpc"] = out["abs_L_z_kpc_kms"] / VO_KMS
    out["Omega_phi_abs"] = out["Omega_phi"].abs()
    out["Omega_z_over_Omega_phi_abs"] = out["Omega_z"] / out["Omega_phi_abs"]
    out["Omega_R_over_Omega_phi_abs"] = out["Omega_R"] / out["Omega_phi_abs"]
    out["Jz_over_JR_plus_Jz"] = out["J_z_kpc_kms"] / (out["J_R_kpc_kms"] + out["J_z_kpc_kms"])
    out["Jz_over_abs_Lz"] = out["J_z_kpc_kms"] / out["abs_L_z_kpc_kms"]
    out["JR_over_abs_Lz"] = out["J_R_kpc_kms"] / out["abs_L_z_kpc_kms"]
    out["zmax_over_Rg"] = out["zmax_kpc"] / out["R_gc_kpc"]
    out["radial_vertical_frequency_balance"] = out["Omega_z"] / out["Omega_R"]
    out["apo_peri_ratio"] = out["rap_kpc"] / out["rperi_kpc"]
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    greater = 0
    less = 0
    for val in x:
        greater += int(np.sum(val > y))
        less += int(np.sum(val < y))
    return float((greater - less) / (len(x) * len(y)))


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {"status": "insufficient"}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    medians = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
        medians[i] = np.median(sample)
    return {
        "status": "ok",
        "mean_p16": float(np.quantile(means, 0.16)),
        "mean_p50": float(np.quantile(means, 0.50)),
        "mean_p84": float(np.quantile(means, 0.84)),
        "median_p16": float(np.quantile(medians, 0.16)),
        "median_p50": float(np.quantile(medians, 0.50)),
        "median_p84": float(np.quantile(medians, 0.84)),
        "mean_ci95_low": float(np.quantile(means, 0.025)),
        "mean_ci95_high": float(np.quantile(means, 0.975)),
        "median_ci95_low": float(np.quantile(medians, 0.025)),
        "median_ci95_high": float(np.quantile(medians, 0.975)),
    }


def bayesian_normal_mean(values: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {"status": "insufficient"}
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    if se == 0 or not np.isfinite(se):
        return {"status": "degenerate", "posterior_mean": mean, "posterior_sd": se}
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    prob_gt0 = float(1.0 - stats.t.cdf(0.0, df=n - 1, loc=mean, scale=se))
    return {
        "status": "ok",
        "posterior_mean": mean,
        "posterior_sd": se,
        "posterior_ci95_low": float(ci_low),
        "posterior_ci95_high": float(ci_high),
        "posterior_probability_gt0": prob_gt0,
    }


def summarize_distribution(df: pd.DataFrame, metric: str, high_mask: pd.Series, control_mask: pd.Series) -> dict:
    high = df.loc[high_mask, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    ctrl = df.loc[control_mask, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    row = {
        "metric": metric,
        "n_vear": int(len(high)),
        "n_control": int(len(ctrl)),
        "vear_median": float(np.median(high)) if len(high) else math.nan,
        "control_median": float(np.median(ctrl)) if len(ctrl) else math.nan,
        "median_difference": float(np.median(high) - np.median(ctrl)) if len(high) and len(ctrl) else math.nan,
        "vear_mean": float(np.mean(high)) if len(high) else math.nan,
        "control_mean": float(np.mean(ctrl)) if len(ctrl) else math.nan,
        "mean_difference": float(np.mean(high) - np.mean(ctrl)) if len(high) and len(ctrl) else math.nan,
        "cliffs_delta": cliffs_delta(high, ctrl),
    }
    if len(high) >= 3 and len(ctrl) >= 3:
        row["mann_whitney_p"] = float(stats.mannwhitneyu(high, ctrl, alternative="two-sided").pvalue)
        row["ks_p"] = float(stats.ks_2samp(high, ctrl).pvalue)
    else:
        row["mann_whitney_p"] = math.nan
        row["ks_p"] = math.nan
    return row


def build_augmented_matches(df: pd.DataFrame, pairs: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = ["ID", "Name", *metrics]
    high_values = df[keep].rename(columns={c: f"high_{c}" for c in keep if c not in ["ID", "Name"]})
    high_values = high_values.rename(columns={"ID": "high_ID", "Name": "high_Name"})
    ctrl_values = df[keep].rename(columns={c: f"control_{c}" for c in keep if c not in ["ID", "Name"]})
    ctrl_values = ctrl_values.rename(columns={"ID": "control_ID", "Name": "control_Name"})
    aug = pairs[["high_ID", "high_Name", "control_ID", "control_Name"]].merge(high_values, on=["high_ID", "high_Name"], how="left")
    aug = aug.merge(ctrl_values, on=["control_ID", "control_Name"], how="left")
    for metric in metrics:
        aug[f"diff_{metric}"] = aug[f"high_{metric}"] - aug[f"control_{metric}"]

    rows = []
    for (hid, hname), grp in aug.groupby(["high_ID", "high_Name"], sort=False):
        row = {"high_ID": hid, "high_Name": hname, "n_controls": int(len(grp))}
        for metric in metrics:
            row[f"high_{metric}"] = float(grp[f"high_{metric}"].iloc[0])
            row[f"control_mean_{metric}"] = float(grp[f"control_{metric}"].mean())
            row[f"control_median_{metric}"] = float(grp[f"control_{metric}"].median())
            row[f"diff_mean_{metric}"] = row[f"high_{metric}"] - row[f"control_mean_{metric}"]
            row[f"diff_median_{metric}"] = row[f"high_{metric}"] - row[f"control_median_{metric}"]
        rows.append(row)
    return aug, pd.DataFrame(rows)


def matched_metric_stats(summary: pd.DataFrame, metric: str, n_bootstrap: int, seed: int) -> dict:
    diffs = summary[f"diff_mean_{metric}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    high = summary[f"high_{metric}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    control = summary[f"control_mean_{metric}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    row = {
        "metric": metric,
        "n_matched_vear": int(len(diffs)),
        "median_vear": float(np.median(high)) if len(high) else math.nan,
        "median_control_mean": float(np.median(control)) if len(control) else math.nan,
        "median_diff_mean_control": float(np.median(diffs)) if len(diffs) else math.nan,
        "mean_diff_mean_control": float(np.mean(diffs)) if len(diffs) else math.nan,
        "fraction_positive": float(np.mean(diffs > 0)) if len(diffs) else math.nan,
        "cohens_dz": float(np.mean(diffs) / np.std(diffs, ddof=1)) if len(diffs) > 2 and np.std(diffs, ddof=1) > 0 else math.nan,
    }
    if len(diffs) >= 3:
        row["wilcoxon_p"] = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
        positives = int(np.sum(diffs > 0))
        row["sign_test_p"] = float(stats.binomtest(positives, n=len(diffs), p=0.5, alternative="two-sided").pvalue)
    else:
        row["wilcoxon_p"] = math.nan
        row["sign_test_p"] = math.nan
    boot = bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed)
    bayes = bayesian_normal_mean(diffs)
    for key, value in boot.items():
        row[f"bootstrap_{key}"] = value
    for key, value in bayes.items():
        row[f"bayes_{key}"] = value
    return row


def save_figures(df: pd.DataFrame, matched_summary: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    vear = df["is_vear"]
    ctrl = df["is_control_like"]

    plt.figure(figsize=(7.2, 5.4))
    plt.scatter(
        df.loc[ctrl, "Omega_R_over_Omega_phi_abs"],
        df.loc[ctrl, "Omega_z_over_Omega_phi_abs"],
        s=16,
        c="#4c78a8",
        alpha=0.45,
        label="Control-like",
    )
    plt.scatter(
        df.loc[vear, "Omega_R_over_Omega_phi_abs"],
        df.loc[vear, "Omega_z_over_Omega_phi_abs"],
        s=22,
        c="#e45756",
        alpha=0.65,
        label="VEAR",
    )
    plt.xlabel(r"$\Omega_R/|\Omega_\phi|$")
    plt.ylabel(r"$\Omega_z/|\Omega_\phi|$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "figure1_frequency_ratio_plane.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, metric, label in [
        (axes[0], "Omega_z_over_Omega_phi_abs", r"$\Omega_z/|\Omega_\phi|$"),
        (axes[1], "Omega_R_over_Omega_phi_abs", r"$\Omega_R/|\Omega_\phi|$"),
    ]:
        vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        bins = np.linspace(vals.quantile(0.005), vals.quantile(0.995), 36)
        ax.hist(df.loc[ctrl, metric].dropna(), bins=bins, density=True, alpha=0.58, color="#4c78a8", label="Control-like")
        ax.hist(df.loc[vear, metric].dropna(), bins=bins, density=True, alpha=0.58, color="#e45756", label="VEAR")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "figure2_frequency_ratio_distributions.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(df.loc[ctrl, "orbit_eccentricity"], df.loc[ctrl, "zmax_over_Rg"], s=16, c="#4c78a8", alpha=0.45, label="Control-like")
    plt.scatter(df.loc[vear, "orbit_eccentricity"], df.loc[vear, "zmax_over_Rg"], s=22, c="#e45756", alpha=0.65, label="VEAR")
    plt.xlabel("Orbital eccentricity")
    plt.ylabel(r"$z_{\max}/R_{\rm GC}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "figure3_eccentricity_zmax_plane.png", dpi=220)
    plt.close()

    metrics = ["Omega_z_over_Omega_phi_abs", "Omega_R_over_Omega_phi_abs", "Jz_over_JR_plus_Jz", "Jz_over_abs_Lz", "zmax_over_Rg"]
    labels = [r"$\Omega_z/|\Omega_\phi|$", r"$\Omega_R/|\Omega_\phi|$", r"$J_z/(J_R+J_z)$", r"$J_z/|L_z|$", r"$z_{\max}/R_{\rm GC}$"]
    diffs = [matched_summary[f"diff_mean_{m}"].dropna().to_numpy() for m in metrics]
    plt.figure(figsize=(8.0, 4.8))
    plt.boxplot(diffs, tick_labels=labels, showfliers=False)
    plt.axhline(0, color="0.3", lw=1)
    plt.ylabel("VEAR minus matched-control mean")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "figure4_matched_orbital_family_differences.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/v5_lz_eccentricity_matching/v5_catalog_with_eccentricity.csv"))
    parser.add_argument("--v3-report", type=Path, default=Path("results/v3_action_space/summary_report.json"))
    parser.add_argument("--matched-pairs", type=Path, default=Path("results/v5_lz_eccentricity_matching/matched_pairs_strict.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/orbital_family_analysis"))
    parser.add_argument("--high-label", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--overwrite-frequencies", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    high_label = load_high_label(args.v3_report, args.high_label)
    df = pd.read_csv(args.catalog)
    df = compute_frequencies(df, args.outdir / "orbital_frequencies.csv", overwrite=args.overwrite_frequencies)
    df = add_orbit_family_metrics(df, high_label=high_label)
    df.to_csv(args.outdir / "orbit_family_catalog.csv", index=False)

    pairs = pd.read_csv(args.matched_pairs)
    pair_aug, matched_summary = build_augmented_matches(df, pairs, ORBIT_METRICS)
    pair_aug.to_csv(args.outdir / "matched_orbit_family_pairs.csv", index=False)
    matched_summary.to_csv(args.outdir / "matched_orbit_family_summary.csv", index=False)

    full_rows = [
        summarize_distribution(df, metric, df["is_vear"], df["is_control_like"])
        for metric in ORBIT_METRICS
    ]
    pd.DataFrame(full_rows).to_csv(args.outdir / "orbit_family_summary.csv", index=False)

    matched_rows = [
        matched_metric_stats(matched_summary, metric, n_bootstrap=args.bootstrap, seed=args.seed + idx * 101)
        for idx, metric in enumerate(ORBIT_METRICS)
    ]
    pd.DataFrame(matched_rows).to_csv(args.outdir / "frequency_ratio_statistics.csv", index=False)

    key_rows = pd.DataFrame(matched_rows).set_index("metric")
    report = {
        "input_catalog": str(args.catalog),
        "matched_pairs": str(args.matched_pairs),
        "high_label": int(high_label),
        "n_clusters": int(len(df)),
        "n_vear": int(df["is_vear"].sum()),
        "n_control_like": int(df["is_control_like"].sum()),
        "n_strict_matched_vear": int(len(matched_summary)),
        "n_strict_pairs": int(len(pair_aug)),
        "frequency_method": "galpy actionAngleStaeckel.actionsFreqs, MWPotential2014, delta=0.45",
        "primary_results": {
            metric: {
                "median_diff_mean_control": float(key_rows.loc[metric, "median_diff_mean_control"]),
                "wilcoxon_p": float(key_rows.loc[metric, "wilcoxon_p"]),
                "bootstrap_mean_ci95_low": float(key_rows.loc[metric, "bootstrap_mean_ci95_low"]),
                "bootstrap_mean_ci95_high": float(key_rows.loc[metric, "bootstrap_mean_ci95_high"]),
                "bayes_probability_diff_gt0": float(key_rows.loc[metric, "bayes_posterior_probability_gt0"]),
            }
            for metric in ["Omega_z_over_Omega_phi_abs", "Omega_R_over_Omega_phi_abs", "Jz_over_JR_plus_Jz", "Jz_over_abs_Lz", "zmax_over_Rg"]
        },
    }
    (args.outdir / "orbit_family_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(df, matched_summary, args.outdir)
    print(json.dumps(report, indent=2))
    print(f"Wrote orbital-family analysis to {args.outdir}")


if __name__ == "__main__":
    main()
