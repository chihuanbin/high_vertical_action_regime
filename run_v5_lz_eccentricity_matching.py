#!/usr/bin/env python3
"""
v5 matched-control experiment with Lz and eccentricity controls.

This extends v4 by matching high-vertical clusters to control-like clusters in:
- logAge50
- R_GC
- log10(|L_z|)
- orbital eccentricity
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
from scipy import stats

from run_high_latitude_dynamics import RO_KPC, SOLAR_MOTION, VO_KMS


METRICS = ["eta_z", "eta_z_Lz", "zmax_over_Rg", "Delta_logJz", "J_z_kpc_kms"]


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


def compute_eccentricities(df: pd.DataFrame) -> pd.DataFrame:
    if {"orbit_eccentricity", "rperi_kpc", "rap_kpc"}.issubset(df.columns) and df["orbit_eccentricity"].notna().all():
        out = df.copy()
    else:
        from galpy.orbit import Orbit
        from galpy.potential import MWPotential2014

        records = []
        for _, row in df.iterrows():
            orbit = Orbit(
                vxvv=[row.RAdeg, row.DEdeg, row.distance_kpc, row.pmRA, row.pmDE, row.RV],
                radec=True,
                ro=RO_KPC,
                vo=VO_KMS,
                solarmotion=SOLAR_MOTION,
            )
            try:
                ecc = float(np.atleast_1d(orbit.e(analytic=True, pot=MWPotential2014))[0])
                rperi = float(np.atleast_1d(orbit.rperi(analytic=True, pot=MWPotential2014, use_physical=True))[0])
                rap = float(np.atleast_1d(orbit.rap(analytic=True, pot=MWPotential2014, use_physical=True))[0])
            except Exception:
                ecc = np.nan
                rperi = np.nan
                rap = np.nan
            records.append({"orbit_eccentricity": ecc, "rperi_kpc": rperi, "rap_kpc": rap})
        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    out["abs_L_z_kpc_kms"] = out["L_z_kpc_kms"].abs()
    out["log_abs_L_z"] = np.log10(out["abs_L_z_kpc_kms"].where(out["abs_L_z_kpc_kms"] > 0))
    return out


def build_matches(
    df: pd.DataFrame,
    high_label: int,
    age_tol: float,
    rg_tol: float,
    log_lz_tol: float,
    ecc_tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["is_control_like"] = as_bool(work["is_control_like"])
    high = work[work["gmm_label"].astype(int) == high_label].copy()
    controls = work[work["is_control_like"]].copy()
    rows = []
    matched_high_ids = set()
    for _, h in high.iterrows():
        c = controls[
            (controls["ID"] != h["ID"])
            & ((controls["logAge50"] - h["logAge50"]).abs() < age_tol)
            & ((controls["R_gc_kpc"] - h["R_gc_kpc"]).abs() < rg_tol)
            & ((controls["log_abs_L_z"] - h["log_abs_L_z"]).abs() < log_lz_tol)
            & ((controls["orbit_eccentricity"] - h["orbit_eccentricity"]).abs() < ecc_tol)
        ].copy()
        if c.empty:
            continue
        matched_high_ids.add(h["ID"])
        for _, ctrl in c.iterrows():
            row = {
                "high_ID": h["ID"],
                "high_Name": h["Name"],
                "control_ID": ctrl["ID"],
                "control_Name": ctrl["Name"],
                "delta_logAge50": float(h["logAge50"] - ctrl["logAge50"]),
                "delta_R_gc_kpc": float(h["R_gc_kpc"] - ctrl["R_gc_kpc"]),
                "delta_log_abs_L_z": float(h["log_abs_L_z"] - ctrl["log_abs_L_z"]),
                "delta_eccentricity": float(h["orbit_eccentricity"] - ctrl["orbit_eccentricity"]),
                "high_orbit_eccentricity": h["orbit_eccentricity"],
                "control_orbit_eccentricity": ctrl["orbit_eccentricity"],
                "high_log_abs_L_z": h["log_abs_L_z"],
                "control_log_abs_L_z": ctrl["log_abs_L_z"],
            }
            for metric in METRICS:
                row[f"high_{metric}"] = h[metric]
                row[f"control_{metric}"] = ctrl[metric]
                row[f"diff_{metric}"] = h[metric] - ctrl[metric]
            rows.append(row)
    pairs = pd.DataFrame(rows)
    unmatched = high[~high["ID"].isin(matched_high_ids)].copy()
    return pairs, unmatched


def high_summary_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    for high_id, grp in pairs.groupby("high_ID", sort=False):
        row = {
            "high_ID": high_id,
            "high_Name": grp["high_Name"].iloc[0],
            "n_controls": int(len(grp)),
            "mean_abs_delta_logAge50": float(grp["delta_logAge50"].abs().mean()),
            "mean_abs_delta_R_gc_kpc": float(grp["delta_R_gc_kpc"].abs().mean()),
            "mean_abs_delta_log_abs_L_z": float(grp["delta_log_abs_L_z"].abs().mean()),
            "mean_abs_delta_eccentricity": float(grp["delta_eccentricity"].abs().mean()),
            "median_abs_delta_logAge50": float(grp["delta_logAge50"].abs().median()),
            "median_abs_delta_R_gc_kpc": float(grp["delta_R_gc_kpc"].abs().median()),
            "median_abs_delta_log_abs_L_z": float(grp["delta_log_abs_L_z"].abs().median()),
            "median_abs_delta_eccentricity": float(grp["delta_eccentricity"].abs().median()),
        }
        for metric in METRICS:
            row[f"high_{metric}"] = grp[f"high_{metric}"].iloc[0]
            row[f"control_mean_{metric}"] = grp[f"control_{metric}"].mean()
            row[f"control_median_{metric}"] = grp[f"control_{metric}"].median()
            row[f"diff_mean_{metric}"] = row[f"high_{metric}"] - row[f"control_mean_{metric}"]
            row[f"diff_median_{metric}"] = row[f"high_{metric}"] - row[f"control_median_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def sign_test_pvalue(diffs: pd.Series) -> float:
    arr = diffs.replace([np.inf, -np.inf], np.nan).dropna()
    arr = arr[arr != 0]
    if len(arr) == 0:
        return np.nan
    positives = int((arr > 0).sum())
    return float(stats.binomtest(positives, n=len(arr), p=0.5, alternative="two-sided").pvalue)


def bootstrap_ci(values: pd.Series, n_bootstrap: int, seed: int) -> dict:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {"status": "insufficient"}
    rng = np.random.default_rng(seed)
    meds = np.empty(n_bootstrap)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        meds[i] = np.median(sample)
        means[i] = np.mean(sample)
    return {
        "status": "ok",
        "median_p16": float(np.percentile(meds, 16)),
        "median_p50": float(np.percentile(meds, 50)),
        "median_p84": float(np.percentile(meds, 84)),
        "mean_p16": float(np.percentile(means, 16)),
        "mean_p50": float(np.percentile(means, 50)),
        "mean_p84": float(np.percentile(means, 84)),
    }


def matched_summary_stats(summary: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    out = {}
    for metric in METRICS:
        if summary.empty:
            out[metric] = {"status": "insufficient", "n": 0}
            continue
        diffs = summary[f"diff_mean_{metric}"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(diffs) < 3:
            out[metric] = {"status": "insufficient", "n": int(len(diffs))}
            continue
        wilcoxon = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        out[metric] = {
            "status": "ok",
            "n": int(len(diffs)),
            "median_high": float(summary[f"high_{metric}"].median()),
            "median_control_mean": float(summary[f"control_mean_{metric}"].median()),
            "median_diff_mean_control": float(diffs.median()),
            "mean_diff_mean_control": float(diffs.mean()),
            "wilcoxon_p": float(wilcoxon.pvalue),
            "sign_test_p": sign_test_pvalue(diffs),
            "fraction_positive": float((diffs > 0).mean()),
            "bootstrap": bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed),
        }
    return out


def pair_level_stats(pairs: pd.DataFrame) -> dict:
    out = {}
    for metric in METRICS:
        if pairs.empty:
            out[metric] = {"status": "insufficient", "n_pairs": 0}
            continue
        h = pairs[f"high_{metric}"].replace([np.inf, -np.inf], np.nan)
        c = pairs[f"control_{metric}"].replace([np.inf, -np.inf], np.nan)
        diff = h - c
        mask = h.notna() & c.notna()
        h = h[mask]
        c = c[mask]
        diff = diff[mask]
        if len(diff) < 3:
            out[metric] = {"status": "insufficient", "n_pairs": int(len(diff))}
            continue
        out[metric] = {
            "status": "ok",
            "n_pairs": int(len(diff)),
            "median_high": float(h.median()),
            "median_control": float(c.median()),
            "median_pair_difference": float(diff.median()),
            "mann_whitney_p": float(stats.mannwhitneyu(h, c, alternative="two-sided").pvalue),
            "wilcoxon_p": float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue),
            "ks_p": float(stats.ks_2samp(h, c).pvalue),
        }
    return out


def run_matching_case(
    df: pd.DataFrame,
    high_label: int,
    label: str,
    age_tol: float,
    rg_tol: float,
    log_lz_tol: float,
    ecc_tol: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs, unmatched = build_matches(df, high_label, age_tol, rg_tol, log_lz_tol, ecc_tol)
    summary = high_summary_from_pairs(pairs)
    controls_per_high = summary["n_controls"] if not summary.empty else pd.Series(dtype=float)
    report = {
        "label": label,
        "tolerances": {
            "age_tol_dex": age_tol,
            "rg_tol_kpc": rg_tol,
            "log_lz_tol_dex": log_lz_tol,
            "ecc_tol": ecc_tol,
        },
        "matching": {
            "n_pairs": int(len(pairs)),
            "n_matched_high": int(summary["high_ID"].nunique()) if not summary.empty else 0,
            "n_unmatched_high": int(len(unmatched)),
            "controls_per_high_median": float(controls_per_high.median()) if len(controls_per_high) else math.nan,
            "controls_per_high_min": int(controls_per_high.min()) if len(controls_per_high) else 0,
            "controls_per_high_max": int(controls_per_high.max()) if len(controls_per_high) else 0,
        },
        "per_high_main_tests": matched_summary_stats(summary, n_bootstrap=bootstrap, seed=seed),
        "pair_level_sensitivity_tests": pair_level_stats(pairs),
    }
    eta_p = report["per_high_main_tests"].get("eta_z", {}).get("wilcoxon_p", math.nan)
    eta_lz_p = report["per_high_main_tests"].get("eta_z_Lz", {}).get("wilcoxon_p", math.nan)
    z_p = report["per_high_main_tests"].get("zmax_over_Rg", {}).get("wilcoxon_p", math.nan)
    report["success_criteria"] = {
        "eta_z_per_high_p_lt_1e_minus_5": bool(np.isfinite(eta_p) and eta_p < 1e-5),
        "eta_z_per_high_p_lt_1e_minus_10": bool(np.isfinite(eta_p) and eta_p < 1e-10),
        "eta_z_Lz_per_high_p_lt_1e_minus_10": bool(np.isfinite(eta_lz_p) and eta_lz_p < 1e-10),
        "zmax_over_Rg_per_high_p_lt_1e_minus_10": bool(np.isfinite(z_p) and z_p < 1e-10),
    }
    return report, pairs, summary, unmatched


def save_case_outputs(outdir: Path, label: str, pairs: pd.DataFrame, summary: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    pairs.to_csv(outdir / f"matched_pairs_{label}.csv", index=False)
    summary.to_csv(outdir / f"matched_high_summary_{label}.csv", index=False)
    unmatched.to_csv(outdir / f"unmatched_high_{label}.csv", index=False)


def save_figures(strict_summary: pd.DataFrame, relaxed_summary: pd.DataFrame, strict_pairs: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if not strict_pairs.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fields = [
            ("delta_logAge50", r"$|\Delta \log_{10}{\rm Age}|$"),
            ("delta_R_gc_kpc", r"$|\Delta R_{\rm GC}|$ [kpc]"),
            ("delta_log_abs_L_z", r"$|\Delta \log |L_z||$"),
            ("delta_eccentricity", r"$|\Delta e|$"),
        ]
        for ax, (field, label) in zip(axes.flat, fields):
            ax.hist(strict_pairs[field].abs(), bins=28, color="#2878b5", alpha=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel("Strict pairs")
        fig.tight_layout()
        fig.savefig(figdir / "fig1_strict_matching_quality.png", dpi=220)
        plt.close(fig)

    for metric, label, filename in [
        ("eta_z", r"$\eta_z$", "fig2_strict_eta_z.png"),
        ("eta_z_Lz", r"$J_z/|L_z|$", "fig3_strict_eta_z_lz.png"),
        ("zmax_over_Rg", r"$z_{\max}/R_g$", "fig4_strict_zmax_over_rg.png"),
    ]:
        if strict_summary.empty:
            continue
        plt.figure(figsize=(5.6, 5.2))
        x = strict_summary[f"control_mean_{metric}"]
        y = strict_summary[f"high_{metric}"]
        plt.scatter(x, y, s=24, color="#c43c39", alpha=0.75)
        lim_min = float(np.nanmin([x.min(), y.min()]))
        lim_max = float(np.nanmax([x.max(), y.max()]))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color="0.25", ls="--", lw=1.0)
        plt.xlabel(f"Matched-control mean {label}")
        plt.ylabel(f"HVAP {label}")
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=220)
        plt.close()

    if not strict_summary.empty or not relaxed_summary.empty:
        plt.figure(figsize=(7.2, 4.8))
        if not strict_summary.empty:
            plt.hist(strict_summary["diff_mean_zmax_over_Rg"], bins=26, alpha=0.7, color="#c43c39", label="Strict")
        if not relaxed_summary.empty:
            plt.hist(relaxed_summary["diff_mean_zmax_over_Rg"], bins=26, alpha=0.45, color="#2878b5", label="Relaxed")
        plt.axvline(0, color="0.25", ls="--", lw=1.0)
        plt.xlabel(r"$z_{\max}/R_g - \langle z_{\max}/R_g\rangle_{\rm controls}$")
        plt.ylabel("HVAP clusters")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figdir / "fig5_strict_relaxed_zmax_difference.png", dpi=220)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/v3_action_space/v3_action_space_catalog.csv"))
    parser.add_argument("--v3-report", type=Path, default=Path("results/v3_action_space/summary_report.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results/v5_lz_eccentricity_matching"))
    parser.add_argument("--high-label", type=int, default=None)
    parser.add_argument("--age-tol-dex", type=float, default=0.1)
    parser.add_argument("--rg-tol-kpc", type=float, default=0.5)
    parser.add_argument("--strict-log-lz-tol-dex", type=float, default=0.05)
    parser.add_argument("--strict-ecc-tol", type=float, default=0.05)
    parser.add_argument("--relaxed-log-lz-tol-dex", type=float, default=0.10)
    parser.add_argument("--relaxed-ecc-tol", type=float, default=0.10)
    parser.add_argument("--min-strict-matched-high", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    high_label = load_high_label(args.v3_report, args.high_label)
    raw = pd.read_csv(args.catalog)
    catalog = compute_eccentricities(raw)
    catalog.to_csv(args.outdir / "v5_catalog_with_eccentricity.csv", index=False)

    strict_report, strict_pairs, strict_summary, strict_unmatched = run_matching_case(
        catalog,
        high_label,
        "strict",
        args.age_tol_dex,
        args.rg_tol_kpc,
        args.strict_log_lz_tol_dex,
        args.strict_ecc_tol,
        args.bootstrap,
        args.seed,
    )
    relaxed_report, relaxed_pairs, relaxed_summary, relaxed_unmatched = run_matching_case(
        catalog,
        high_label,
        "relaxed",
        args.age_tol_dex,
        args.rg_tol_kpc,
        args.relaxed_log_lz_tol_dex,
        args.relaxed_ecc_tol,
        args.bootstrap,
        args.seed,
    )
    save_case_outputs(args.outdir, "strict", strict_pairs, strict_summary, strict_unmatched)
    save_case_outputs(args.outdir, "relaxed", relaxed_pairs, relaxed_summary, relaxed_unmatched)
    save_figures(strict_summary, relaxed_summary, strict_pairs, args.outdir)

    report = {
        "input_catalog": str(args.catalog),
        "v3_report": str(args.v3_report),
        "high_vertical_gmm_label": high_label,
        "strict": strict_report,
        "relaxed": relaxed_report,
        "notes": [
            "Strict matching controls age, R_GC, log|L_z|, and eccentricity.",
            "Relaxed matching is reported regardless of strict sample size as a sensitivity check.",
        ],
    }
    (args.outdir / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"strict": strict_report["matching"], "relaxed": relaxed_report["matching"]}, indent=2))
    print(json.dumps({"strict": strict_report["per_high_main_tests"], "relaxed": relaxed_report["per_high_main_tests"]}, indent=2))
    print(json.dumps({"strict": strict_report["success_criteria"], "relaxed": relaxed_report["success_criteria"]}, indent=2))
    print(f"Wrote v5 Lz/e matching experiment to {args.outdir}")


if __name__ == "__main__":
    main()
