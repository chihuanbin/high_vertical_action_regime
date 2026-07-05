#!/usr/bin/env python3
"""
v3 action-space analysis for HUNT24 open clusters.

Inputs are the v2 vertical-action catalog. This script adds:
- eta_z = J_z / (J_R + J_z)
- eta_z_Lz = J_z / |L_z|
- zmax/R_g
- Gaussian-mixture clustering in action space
- HDBSCAN clustering in action space
- publication-oriented statistics and figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))

import hdbscan
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ACTION_COLS = ["J_R_kpc_kms", "J_z_kpc_kms", "L_z_kpc_kms"]


def finite_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def add_v3_metrics(df: pd.DataFrame) -> pd.DataFrame:
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
    for col in [
        "is_vertical_action_outlier",
        "is_vertical_action_outlier_top_n",
        "is_vertical_action_outlier_top_fraction",
        "is_control_like",
    ]:
        if col in out:
            out[col] = finite_bool(out[col])
    return out


def action_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    features = df[["log_J_R", "log_J_z", "log_abs_L_z"]].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    return features, x, scaler


def run_gmm(df: pd.DataFrame, max_components: int, seed: int) -> tuple[pd.DataFrame, dict]:
    features, x, _ = action_feature_matrix(df)
    rows = []
    models = {}
    for k in range(1, max_components + 1):
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=10)
        model.fit(x)
        rows.append({"k": k, "bic": float(model.bic(x)), "aic": float(model.aic(x))})
        models[k] = model
    bic_table = pd.DataFrame(rows)
    best_k = int(bic_table.loc[bic_table["bic"].idxmin(), "k"])
    best_model = models[best_k]
    labels = pd.Series(best_model.predict(x), index=features.index, name="gmm_label")
    probs = pd.Series(best_model.predict_proba(x).max(axis=1), index=features.index, name="gmm_prob")
    out = df.copy()
    out["gmm_label"] = -1
    out["gmm_prob"] = np.nan
    out.loc[labels.index, "gmm_label"] = labels
    out.loc[probs.index, "gmm_prob"] = probs
    summary = {
        "best_k": best_k,
        "bic_table": bic_table.to_dict(orient="records"),
        "delta_bic_k1_minus_best": float(bic_table.loc[bic_table["k"] == 1, "bic"].iloc[0] - bic_table["bic"].min()),
    }
    return out, summary


def run_hdbscan(df: pd.DataFrame, min_cluster_size: int, min_samples: int | None) -> tuple[pd.DataFrame, dict]:
    features, x, _ = action_feature_matrix(df)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels_arr = clusterer.fit_predict(x)
    labels = pd.Series(labels_arr, index=features.index, name="hdbscan_label")
    probs = pd.Series(clusterer.probabilities_, index=features.index, name="hdbscan_prob")
    out = df.copy()
    out["hdbscan_label"] = -99
    out["hdbscan_prob"] = np.nan
    out.loc[labels.index, "hdbscan_label"] = labels
    out.loc[probs.index, "hdbscan_prob"] = probs
    n_clusters = int(len(set(labels_arr)) - (1 if -1 in labels_arr else 0))
    summary = {
        "n_clusters": n_clusters,
        "noise_fraction": float(np.mean(labels_arr == -1)),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None if min_samples is None else int(min_samples),
    }
    return out, summary


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.dropna().to_numpy()
    y_arr = y.dropna().to_numpy()
    if len(x_arr) == 0 or len(y_arr) == 0:
        return np.nan
    greater = 0
    less = 0
    for val in x_arr:
        greater += np.sum(val > y_arr)
        less += np.sum(val < y_arr)
    return float((greater - less) / (len(x_arr) * len(y_arr)))


def group_compare(df: pd.DataFrame, group_mask: pd.Series, control_mask: pd.Series, columns: list[str]) -> dict:
    results = {}
    for col in columns:
        a = df.loc[group_mask, col].replace([np.inf, -np.inf], np.nan).dropna()
        b = df.loc[control_mask, col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 3 or len(b) < 3:
            results[col] = {"status": "insufficient", "n_group": int(len(a)), "n_control": int(len(b))}
            continue
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        ks = stats.ks_2samp(a, b)
        results[col] = {
            "status": "ok",
            "n_group": int(len(a)),
            "n_control": int(len(b)),
            "median_group": float(np.median(a)),
            "median_control": float(np.median(b)),
            "median_difference": float(np.median(a) - np.median(b)),
            "mann_whitney_p": float(mw.pvalue),
            "ks_p": float(ks.pvalue),
            "cliffs_delta": cliffs_delta(a, b),
        }
    return results


def cluster_summary(df: pd.DataFrame, label_col: str) -> list[dict]:
    rows = []
    for label, grp in df.groupby(label_col):
        if label in [-99]:
            continue
        rows.append(
            {
                "label": int(label),
                "n": int(len(grp)),
                "median_log_J_R": float(grp["log_J_R"].median()),
                "median_log_J_z": float(grp["log_J_z"].median()),
                "median_log_abs_L_z": float(grp["log_abs_L_z"].median()),
                "median_eta_z": float(grp["eta_z"].median()),
                "median_zmax_over_Rg": float(grp["zmax_over_Rg"].median()),
                "median_Delta_logJz": float(grp["Delta_logJz"].median()),
                "fraction_delta_outlier": float(grp["is_vertical_action_outlier"].mean()),
                "fraction_topn_outlier": float(grp["is_vertical_action_outlier_top_n"].mean()),
            }
        )
    return sorted(rows, key=lambda row: row["median_eta_z"], reverse=True)


def identify_high_vertical_cluster(summary_rows: list[dict]) -> int | None:
    candidates = [row for row in summary_rows if row["n"] >= 10]
    if not candidates:
        return None
    keys = ["median_eta_z", "median_zmax_over_Rg", "median_Delta_logJz", "fraction_delta_outlier"]
    values = {key: np.array([row[key] for row in candidates], dtype=float) for key in keys}
    for row_idx, row in enumerate(candidates):
        score = 0.0
        for key in keys:
            arr = values[key]
            spread = np.nanmax(arr) - np.nanmin(arr)
            score += 0.0 if spread == 0 else (arr[row_idx] - np.nanmin(arr)) / spread
        row["high_vertical_score"] = float(score)
    candidates.sort(key=lambda row: row["high_vertical_score"], reverse=True)
    return int(candidates[0]["label"])


def correlation_summary(df: pd.DataFrame) -> dict:
    pairs = [("eta_z", "zmax_over_Rg"), ("eta_z", "Delta_logJz"), ("zmax_over_Rg", "Delta_logJz")]
    out = {}
    for xcol, ycol in pairs:
        data = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 4:
            out[f"{xcol}_vs_{ycol}"] = {"status": "insufficient", "n": int(len(data))}
            continue
        sp = stats.spearmanr(data[xcol], data[ycol])
        out[f"{xcol}_vs_{ycol}"] = {
            "status": "ok",
            "n": int(len(data)),
            "spearman_r": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
        }
    return out


def save_figures(df: pd.DataFrame, report: dict, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    high_gmm = report["high_vertical_gmm_label"]
    high_mask = df["gmm_label"] == high_gmm if high_gmm is not None else df["is_vertical_action_outlier_top_n"]
    ctrl = df["is_control_like"]

    plt.figure(figsize=(7.2, 5.4))
    sc = plt.scatter(df["log_J_R"], df["log_J_z"], c=df["eta_z"], s=16, cmap="magma", alpha=0.75)
    if high_gmm is not None:
        plt.scatter(df.loc[high_mask, "log_J_R"], df.loc[high_mask, "log_J_z"], facecolors="none", edgecolors="#2b2b2b", s=42, lw=0.8, label="High-eta GMM cluster")
    plt.xlabel(r"$\log_{10} J_R$")
    plt.ylabel(r"$\log_{10} J_z$")
    cb = plt.colorbar(sc)
    cb.set_label(r"$\eta_z = J_z/(J_R+J_z)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig1_action_space_eta_z.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    bins = np.linspace(df["eta_z"].quantile(0.005), df["eta_z"].quantile(0.995), 34)
    plt.hist(df.loc[ctrl, "eta_z"].dropna(), bins=bins, alpha=0.65, density=True, color="#2878b5", label="Control-like")
    plt.hist(df.loc[high_mask, "eta_z"].dropna(), bins=bins, alpha=0.65, density=True, color="#c43c39", label="High-vertical population")
    plt.xlabel(r"$\eta_z$")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig2_eta_z_distribution.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    bins = np.linspace(df["zmax_over_Rg"].quantile(0.005), df["zmax_over_Rg"].quantile(0.995), 34)
    axes[0].hist(df.loc[ctrl, "zmax_over_Rg"].dropna(), bins=bins, alpha=0.65, density=True, color="#2878b5", label="Control-like")
    axes[0].hist(df.loc[high_mask, "zmax_over_Rg"].dropna(), bins=bins, alpha=0.65, density=True, color="#c43c39", label="High-vertical population")
    axes[0].set_xlabel(r"$z_{\max}/R_g$")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[1].scatter(df["eta_z"], df["zmax_over_Rg"], s=14, c="0.68", alpha=0.5)
    axes[1].scatter(df.loc[high_mask, "eta_z"], df.loc[high_mask, "zmax_over_Rg"], s=28, c="#c43c39", alpha=0.85)
    axes[1].set_xlabel(r"$\eta_z$")
    axes[1].set_ylabel(r"$z_{\max}/R_g$")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_zmax_over_rg.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(7.2, 5.4))
    labels = df["gmm_label"].astype(int)
    for label in sorted(labels.unique()):
        grp = df[labels == label]
        plt.scatter(grp["log_J_R"], grp["log_J_z"], s=14, alpha=0.65, label=f"GMM {label}")
    plt.xlabel(r"$\log_{10} J_R$")
    plt.ylabel(r"$\log_{10} J_z$")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(figdir / "fig4_gmm_action_clusters.png", dpi=220)
    plt.close()


def write_outputs(df: pd.DataFrame, report: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "v3_action_space_catalog.csv", index=False)
    gmm_summary = pd.DataFrame(report["gmm_cluster_summary"])
    hdb_summary = pd.DataFrame(report["hdbscan_cluster_summary"])
    gmm_summary.to_csv(outdir / "gmm_cluster_summary.csv", index=False)
    hdb_summary.to_csv(outdir / "hdbscan_cluster_summary.csv", index=False)
    high_label = report["high_vertical_gmm_label"]
    if high_label is not None:
        df[df["gmm_label"] == high_label].to_csv(outdir / "high_vertical_gmm_population.csv", index=False)
    (outdir / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figures(df, report, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/vertical_action_outliers/vertical_action_catalog.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/v3_action_space"))
    parser.add_argument("--max-gmm-components", type=int, default=6)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.catalog)
    df = add_v3_metrics(df)
    df, gmm = run_gmm(df, max_components=args.max_gmm_components, seed=args.seed)
    df, hdb = run_hdbscan(df, min_cluster_size=args.hdbscan_min_cluster_size, min_samples=args.hdbscan_min_samples)
    gmm_clusters = cluster_summary(df, "gmm_label")
    hdb_clusters = cluster_summary(df[df["hdbscan_label"] != -1], "hdbscan_label")
    high_gmm_label = identify_high_vertical_cluster(gmm_clusters)
    high_hdb_label = identify_high_vertical_cluster(hdb_clusters)
    high_mask = df["gmm_label"] == high_gmm_label if high_gmm_label is not None else df["is_vertical_action_outlier_top_n"]
    control_mask = df["is_control_like"]
    report = {
        "input_catalog": str(args.catalog),
        "counts": {
            "n_clusters": int(len(df)),
            "n_control_like": int(control_mask.sum()),
            "n_high_vertical_gmm_population": int(high_mask.sum()),
            "n_delta_outliers": int(df["is_vertical_action_outlier"].sum()),
            "n_topn_outliers": int(df["is_vertical_action_outlier_top_n"].sum()),
        },
        "gmm": gmm,
        "hdbscan": hdb,
        "gmm_cluster_summary": gmm_clusters,
        "hdbscan_cluster_summary": hdb_clusters,
        "high_vertical_gmm_label": high_gmm_label,
        "high_vertical_hdbscan_label": high_hdb_label,
        "high_vertical_vs_control": group_compare(
            df,
            high_mask,
            control_mask,
            ["eta_z", "eta_z_Lz", "zmax_over_Rg", "Delta_logJz", "J_z_kpc_kms"],
        ),
        "topn_vs_control": group_compare(
            df,
            df["is_vertical_action_outlier_top_n"],
            control_mask,
            ["eta_z", "eta_z_Lz", "zmax_over_Rg", "Delta_logJz", "J_z_kpc_kms"],
        ),
        "correlations": correlation_summary(df),
        "success_criteria": {
            "gmm_prefers_multiple_components": bool(gmm["best_k"] > 1),
            "gmm_delta_bic_k1_minus_best": gmm["delta_bic_k1_minus_best"],
            "hdbscan_finds_clusters": bool(hdb["n_clusters"] > 0),
        },
    }
    write_outputs(df, report, args.outdir)
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["success_criteria"], indent=2))
    print(json.dumps(report["high_vertical_vs_control"], indent=2))
    print(f"Wrote v3 results to {args.outdir}")


if __name__ == "__main__":
    main()
