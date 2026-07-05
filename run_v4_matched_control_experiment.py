#!/usr/bin/env python3
"""
v4 matched-control experiment for the high-vertical GMM population.

For every high-vertical cluster, find control-like clusters satisfying:
- |Delta logAge50| < 0.1 dex
- |Delta R_GC| < 0.5 kpc

The main inference uses one independent row per high-vertical cluster by
comparing the high cluster to the mean/median of its matched controls.
Pair-level tests are reported only as sensitivity checks.
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


def build_matches(df: pd.DataFrame, high_label: int, age_tol: float, rg_tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "high_logAge50": h["logAge50"],
                "control_logAge50": ctrl["logAge50"],
                "high_R_gc_kpc": h["R_gc_kpc"],
                "control_R_gc_kpc": ctrl["R_gc_kpc"],
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
            "median_abs_delta_logAge50": float(grp["delta_logAge50"].abs().median()),
            "median_abs_delta_R_gc_kpc": float(grp["delta_R_gc_kpc"].abs().median()),
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
    result = stats.binomtest(positives, n=len(arr), p=0.5, alternative="two-sided")
    return float(result.pvalue)


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


def save_figures(pairs: pd.DataFrame, high_summary: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(pairs["delta_logAge50"].abs(), bins=30, color="#2878b5", alpha=0.8)
    axes[0].axvline(0.1, color="0.2", ls="--", lw=1.0)
    axes[0].set_xlabel(r"$|\Delta \log_{10}{\rm Age}|$")
    axes[0].set_ylabel("Matched pairs")
    axes[1].hist(pairs["delta_R_gc_kpc"].abs(), bins=30, color="#2878b5", alpha=0.8)
    axes[1].axvline(0.5, color="0.2", ls="--", lw=1.0)
    axes[1].set_xlabel(r"$|\Delta R_{\rm GC}|$ [kpc]")
    axes[1].set_ylabel("Matched pairs")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_matching_quality.png", dpi=220)
    plt.close(fig)

    for metric, label, filename in [
        ("eta_z", r"$\eta_z$", "fig2_matched_eta_z.png"),
        ("zmax_over_Rg", r"$z_{\max}/R_g$", "fig3_matched_zmax_over_rg.png"),
    ]:
        plt.figure(figsize=(5.6, 5.2))
        x = high_summary[f"control_mean_{metric}"]
        y = high_summary[f"high_{metric}"]
        plt.scatter(x, y, s=22, color="#c43c39", alpha=0.75)
        lim_min = float(np.nanmin([x.min(), y.min()]))
        lim_max = float(np.nanmax([x.max(), y.max()]))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color="0.25", ls="--", lw=1.0)
        plt.xlabel(f"Matched-control mean {label}")
        plt.ylabel(f"High-vertical {label}")
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=220)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(high_summary["diff_mean_eta_z"], bins=32, color="#c43c39", alpha=0.8)
    axes[0].axvline(0, color="0.2", ls="--", lw=1.0)
    axes[0].set_xlabel(r"$\eta_z - \langle\eta_z\rangle_{\rm controls}$")
    axes[0].set_ylabel("High-vertical clusters")
    axes[1].hist(high_summary["diff_mean_zmax_over_Rg"], bins=32, color="#c43c39", alpha=0.8)
    axes[1].axvline(0, color="0.2", ls="--", lw=1.0)
    axes[1].set_xlabel(r"$z_{\max}/R_g - \langle z_{\max}/R_g\rangle_{\rm controls}$")
    axes[1].set_ylabel("High-vertical clusters")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_per_high_differences.png", dpi=220)
    plt.close(fig)


def write_outputs(
    pairs: pd.DataFrame,
    high_summary: pd.DataFrame,
    unmatched: pd.DataFrame,
    report: dict,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(outdir / "matched_pairs.csv", index=False)
    high_summary.to_csv(outdir / "matched_high_summary.csv", index=False)
    unmatched.to_csv(outdir / "unmatched_high_population.csv", index=False)
    (outdir / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not pairs.empty and not high_summary.empty:
        save_figures(pairs, high_summary, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/v3_action_space/v3_action_space_catalog.csv"))
    parser.add_argument("--v3-report", type=Path, default=Path("results/v3_action_space/summary_report.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results/v4_matched_experiment"))
    parser.add_argument("--high-label", type=int, default=None)
    parser.add_argument("--age-tol-dex", type=float, default=0.1)
    parser.add_argument("--rg-tol-kpc", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    high_label = load_high_label(args.v3_report, args.high_label)
    df = pd.read_csv(args.catalog)
    pairs, unmatched = build_matches(df, high_label=high_label, age_tol=args.age_tol_dex, rg_tol=args.rg_tol_kpc)
    high_summary = high_summary_from_pairs(pairs)
    controls_per_high = high_summary["n_controls"] if not high_summary.empty else pd.Series(dtype=float)
    report = {
        "input_catalog": str(args.catalog),
        "v3_report": str(args.v3_report),
        "high_vertical_gmm_label": high_label,
        "matching": {
            "age_tol_dex": args.age_tol_dex,
            "rg_tol_kpc": args.rg_tol_kpc,
            "n_pairs": int(len(pairs)),
            "n_matched_high": int(high_summary["high_ID"].nunique()) if not high_summary.empty else 0,
            "n_unmatched_high": int(len(unmatched)),
            "controls_per_high_median": float(controls_per_high.median()) if len(controls_per_high) else math.nan,
            "controls_per_high_min": int(controls_per_high.min()) if len(controls_per_high) else 0,
            "controls_per_high_max": int(controls_per_high.max()) if len(controls_per_high) else 0,
        },
        "per_high_main_tests": matched_summary_stats(high_summary, n_bootstrap=args.bootstrap, seed=args.seed),
        "pair_level_sensitivity_tests": pair_level_stats(pairs),
        "success_criteria": {},
    }
    eta_p = report["per_high_main_tests"].get("eta_z", {}).get("wilcoxon_p", math.nan)
    z_p = report["per_high_main_tests"].get("zmax_over_Rg", {}).get("wilcoxon_p", math.nan)
    report["success_criteria"] = {
        "eta_z_per_high_p_lt_1e_minus_10": bool(np.isfinite(eta_p) and eta_p < 1e-10),
        "zmax_over_Rg_per_high_p_lt_1e_minus_10": bool(np.isfinite(z_p) and z_p < 1e-10),
        "both_primary_metrics_p_lt_1e_minus_10": bool(
            np.isfinite(eta_p) and np.isfinite(z_p) and eta_p < 1e-10 and z_p < 1e-10
        ),
    }
    write_outputs(pairs, high_summary, unmatched, report, args.outdir)
    print(json.dumps(report["matching"], indent=2))
    print(json.dumps(report["per_high_main_tests"], indent=2))
    print(json.dumps(report["success_criteria"], indent=2))
    print(f"Wrote v4 matched-control experiment to {args.outdir}")


if __name__ == "__main__":
    main()
