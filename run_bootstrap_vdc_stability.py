#!/usr/bin/env python3
"""
Bootstrap stability test for the vertical dynamical component.

The input and preprocessing mirror the action-space GMM analysis in
run_v3_action_space_analysis.py: the same 1079-cluster catalog is transformed
to (log J_R, log J_z, log |L_z|), standardized once on the full sample, and
then used for all bootstrap fits.
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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["log_J_R", "log_J_z", "log_abs_L_z"]
METRIC_COLS = ["eta_z", "zmax_over_Rg", "Delta_logJz", "is_vertical_action_outlier"]


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


def model_criteria(model: GaussianMixture, x: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-300, 1.0))))
    bic = float(model.bic(x))
    aic = float(model.aic(x))
    return {"bic": bic, "aic": aic, "icl": bic + 2.0 * entropy}


def fit_gmm_grid(x: np.ndarray, max_k: int, seed: int, n_init: int) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    rows: list[dict[str, float | int]] = []
    models: dict[int, GaussianMixture] = {}
    for k in range(1, max_k + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=n_init,
            reg_covar=1e-6,
        )
        model.fit(x)
        crit = model_criteria(model, x)
        rows.append({"k": k, **crit})
        models[k] = model
    return pd.DataFrame(rows), models


def identify_vertical_component(df: pd.DataFrame, labels: np.ndarray) -> int:
    work = df.copy()
    work["_label"] = labels
    rows = []
    for label, grp in work.groupby("_label"):
        if len(grp) < 10:
            continue
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
    if not rows:
        return int(pd.Series(labels).value_counts().idxmax())
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


def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    _, a_inv = np.unique(a, return_inverse=True)
    _, b_inv = np.unique(b, return_inverse=True)
    contingency = np.zeros((a_inv.max() + 1, b_inv.max() + 1), dtype=float)
    np.add.at(contingency, (a_inv, b_inv), 1.0)
    pxy = contingency / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    return hx + hy - 2.0 * mi


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(a, b).sum() / union)


def component_overlap_cost(reference_labels: np.ndarray, candidate_labels: np.ndarray, n_ref: int, n_cand: int) -> np.ndarray:
    cost = np.ones((n_ref, n_cand), dtype=float)
    for i in range(n_ref):
        ref_mask = reference_labels == i
        for j in range(n_cand):
            cost[i, j] = 1.0 - jaccard(ref_mask, candidate_labels == j)
    return cost


def covariance_to_feature_units(cov_std: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scale = scaler.scale_
    return cov_std * scale[:, None] * scale[None, :]


def save_figures(
    df: pd.DataFrame,
    x_features: np.ndarray,
    baseline_centers: np.ndarray,
    center_rows: pd.DataFrame,
    summary: pd.DataFrame,
    membership: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 4.8))
    bins = np.arange(0.5, 8.6, 1.0)
    plt.hist(summary["best_k_bic"], bins=bins, color="#4c78a8", alpha=0.78, label="BIC")
    plt.hist(summary["best_k_icl"], bins=bins, color="#f58518", alpha=0.48, label="ICL")
    plt.xticks(range(1, 9))
    plt.xlabel("Optimal number of GMM components")
    plt.ylabel("Bootstrap realizations")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "figure1_optimal_k_histogram.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.1, 5.4))
    colors = ["#4c78a8", "#e45756", "#54a24b", "#b279a2"]
    for label, grp in center_rows.groupby("matched_baseline_label"):
        if label < 0:
            continue
        plt.scatter(grp["center_log_J_R"], grp["center_log_J_z"], s=10, alpha=0.18, color=colors[int(label) % len(colors)], label=f"Component {int(label)}")
    plt.scatter(baseline_centers[:, 0], baseline_centers[:, 1], s=96, color="black", marker="x", linewidths=2, label="Full-sample centers")
    plt.xlabel(r"$\log_{10} J_R$")
    plt.ylabel(r"$\log_{10} J_z$")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "figure2_bootstrap_center_cloud.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 5.4))
    sc = plt.scatter(
        x_features[:, 0],
        x_features[:, 1],
        c=membership["vdc_membership_probability"],
        s=18,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.82,
    )
    plt.xlabel(r"$\log_{10} J_R$")
    plt.ylabel(r"$\log_{10} J_z$")
    cb = plt.colorbar(sc)
    cb.set_label("Bootstrap VDC membership probability")
    plt.tight_layout()
    plt.savefig(figdir / "figure3_membership_probability_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.hist(summary["ari"], bins=35, color="#4c78a8", alpha=0.82)
    plt.xlabel("Adjusted Rand index")
    plt.ylabel("Bootstrap realizations")
    plt.tight_layout()
    plt.savefig(figdir / "figure4_ari_distribution.png", dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("results/vertical_action_outliers/vertical_action_catalog.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/bootstrap_vdc_stability"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--survival-jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.catalog)
    df = add_action_metrics(df_raw)
    features = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    valid = features.dropna().index
    df = df.loc[valid].reset_index(drop=True)
    features = features.loc[valid].reset_index(drop=True)
    if len(df) != 1079:
        raise ValueError(f"Expected the paper sample of 1079 clusters, found {len(df)}")

    scaler = StandardScaler()
    x_full = scaler.fit_transform(features.to_numpy())
    n = len(df)

    baseline_table, baseline_models = fit_gmm_grid(x_full, args.max_components, args.baseline_seed, args.n_init)
    baseline_best_k = int(baseline_table.loc[baseline_table["bic"].idxmin(), "k"])
    baseline_model = baseline_models[baseline_best_k]
    baseline_labels = baseline_model.predict(x_full)
    baseline_vdc_label = identify_vertical_component(df, baseline_labels)
    baseline_vdc_mask = baseline_labels == baseline_vdc_label
    baseline_centers = scaler.inverse_transform(baseline_model.means_)
    baseline_covariances = np.array([covariance_to_feature_units(c, scaler) for c in baseline_model.covariances_])

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, float | int | bool]] = []
    center_rows: list[dict[str, float | int | bool]] = []
    criteria_rows: list[dict[str, float | int]] = []
    membership_count = np.zeros(n, dtype=float)
    membership_prob_sum = np.zeros(n, dtype=float)
    component_survival_count = np.zeros(baseline_best_k, dtype=float)
    component_match_count = np.zeros(baseline_best_k, dtype=float)
    component_jaccard_values: dict[int, list[float]] = {i: [] for i in range(baseline_best_k)}
    component_center_dist_values: dict[int, list[float]] = {i: [] for i in range(baseline_best_k)}

    for boot in range(args.n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        x_boot = x_full[sample_idx]
        table, models = fit_gmm_grid(x_boot, args.max_components, args.seed + 7919 * (boot + 1), args.n_init)
        for row in table.to_dict(orient="records"):
            criteria_rows.append({"bootstrap": boot, **row})
        best_k_bic = int(table.loc[table["bic"].idxmin(), "k"])
        best_k_aic = int(table.loc[table["aic"].idxmin(), "k"])
        best_k_icl = int(table.loc[table["icl"].idxmin(), "k"])
        model = models[best_k_bic]
        pred = model.predict(x_full)
        probs = model.predict_proba(x_full)

        cost = component_overlap_cost(baseline_labels, pred, baseline_best_k, best_k_bic)
        ref_ind, cand_ind = linear_sum_assignment(cost)
        cand_to_ref = {int(cand): int(ref) for ref, cand in zip(ref_ind, cand_ind)}
        matched_labels = np.array([cand_to_ref.get(int(label), -1) for label in pred], dtype=int)
        vdc_candidates = [cand for cand, ref in cand_to_ref.items() if ref == baseline_vdc_label]
        if vdc_candidates:
            vdc_candidate = int(vdc_candidates[0])
            boot_vdc_mask = pred == vdc_candidate
            vdc_probability = probs[:, vdc_candidate]
            vdc_jaccard = jaccard(baseline_vdc_mask, boot_vdc_mask)
        else:
            vdc_candidate = -1
            boot_vdc_mask = np.zeros(n, dtype=bool)
            vdc_probability = np.zeros(n, dtype=float)
            vdc_jaccard = 0.0

        membership_count += boot_vdc_mask.astype(float)
        membership_prob_sum += vdc_probability

        ari = adjusted_rand_score(baseline_labels, pred)
        vi = variation_of_information(baseline_labels, pred)
        vdc_survived = bool(vdc_jaccard >= args.survival_jaccard_threshold)
        summary_rows.append(
            {
                "bootstrap": boot,
                "best_k_bic": best_k_bic,
                "best_k_aic": best_k_aic,
                "best_k_icl": best_k_icl,
                "ari": float(ari),
                "variation_of_information": float(vi),
                "vdc_jaccard": float(vdc_jaccard),
                "vdc_survived": vdc_survived,
                "vdc_candidate_label": vdc_candidate,
            }
        )

        boot_centers = scaler.inverse_transform(model.means_)
        boot_covariances = np.array([covariance_to_feature_units(c, scaler) for c in model.covariances_])
        for ref, cand in zip(ref_ind, cand_ind):
            ref = int(ref)
            cand = int(cand)
            component_match_count[ref] += 1
            comp_jaccard = 1.0 - float(cost[ref, cand])
            component_jaccard_values[ref].append(comp_jaccard)
            center_distance = float(np.linalg.norm(boot_centers[cand] - baseline_centers[ref]))
            component_center_dist_values[ref].append(center_distance)
            survived = comp_jaccard >= args.survival_jaccard_threshold
            component_survival_count[ref] += float(survived)
            center_rows.append(
                {
                    "bootstrap": boot,
                    "matched_baseline_label": ref,
                    "bootstrap_component_label": cand,
                    "component_jaccard": comp_jaccard,
                    "survived": survived,
                    "center_distance": center_distance,
                    "center_log_J_R": float(boot_centers[cand, 0]),
                    "center_log_J_z": float(boot_centers[cand, 1]),
                    "center_log_abs_L_z": float(boot_centers[cand, 2]),
                    "cov_log_J_R_log_J_R": float(boot_covariances[cand, 0, 0]),
                    "cov_log_J_z_log_J_z": float(boot_covariances[cand, 1, 1]),
                    "cov_log_abs_L_z_log_abs_L_z": float(boot_covariances[cand, 2, 2]),
                    "cov_frobenius_distance": float(np.linalg.norm(boot_covariances[cand] - baseline_covariances[ref])),
                }
            )

        if args.progress_every > 0 and (boot + 1) % args.progress_every == 0:
            print(f"Completed {boot + 1}/{args.n_bootstrap} bootstraps", flush=True)

    summary = pd.DataFrame(summary_rows)
    centers = pd.DataFrame(center_rows)
    criteria = pd.DataFrame(criteria_rows)
    membership = df[["Name", "ID", *FEATURE_COLS]].copy()
    membership["baseline_gmm_label"] = baseline_labels
    membership["baseline_vdc_member"] = baseline_vdc_mask
    membership["vdc_membership_probability"] = membership_count / args.n_bootstrap
    membership["vdc_mean_model_probability"] = membership_prob_sum / args.n_bootstrap

    survival_rows = []
    for label in range(baseline_best_k):
        vals = np.array(component_jaccard_values[label], dtype=float)
        dists = np.array(component_center_dist_values[label], dtype=float)
        survival_rows.append(
            {
                "baseline_label": label,
                "is_vdc": bool(label == baseline_vdc_label),
                "baseline_n": int(np.sum(baseline_labels == label)),
                "match_probability": float(component_match_count[label] / args.n_bootstrap),
                "survival_probability": float(component_survival_count[label] / args.n_bootstrap),
                "median_jaccard": float(np.nanmedian(vals)) if len(vals) else math.nan,
                "p16_jaccard": float(np.nanpercentile(vals, 16)) if len(vals) else math.nan,
                "p84_jaccard": float(np.nanpercentile(vals, 84)) if len(vals) else math.nan,
                "median_center_distance": float(np.nanmedian(dists)) if len(dists) else math.nan,
            }
        )
    survival = pd.DataFrame(survival_rows)

    center_uncertainty_rows = []
    for label in range(baseline_best_k):
        grp = centers[centers["matched_baseline_label"] == label]
        row = {"baseline_label": label, "is_vdc": bool(label == baseline_vdc_label), "n_matches": int(len(grp))}
        for col in ["center_log_J_R", "center_log_J_z", "center_log_abs_L_z"]:
            row[f"{col}_baseline"] = float(baseline_centers[label, ["center_log_J_R", "center_log_J_z", "center_log_abs_L_z"].index(col)])
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_std"] = float(grp[col].std(ddof=1))
            row[f"{col}_p16"] = float(grp[col].quantile(0.16))
            row[f"{col}_p84"] = float(grp[col].quantile(0.84))
        row["cov_frobenius_distance_median"] = float(grp["cov_frobenius_distance"].median())
        row["cov_frobenius_distance_p84"] = float(grp["cov_frobenius_distance"].quantile(0.84))
        center_uncertainty_rows.append(row)
    center_uncertainty = pd.DataFrame(center_uncertainty_rows)

    baseline_table.to_csv(args.outdir / "baseline_model_selection.csv", index=False)
    criteria.to_csv(args.outdir / "bootstrap_model_selection_long.csv", index=False)
    summary.to_csv(args.outdir / "bootstrap_summary.csv", index=False)
    survival.to_csv(args.outdir / "component_survival.csv", index=False)
    center_uncertainty.to_csv(args.outdir / "center_uncertainty.csv", index=False)
    centers.to_csv(args.outdir / "bootstrap_component_centers.csv", index=False)
    membership.to_csv(args.outdir / "membership_stability.csv", index=False)

    report = {
        "input_catalog": str(args.catalog),
        "n_clusters": int(n),
        "n_bootstrap": int(args.n_bootstrap),
        "max_components": int(args.max_components),
        "n_init": int(args.n_init),
        "baseline_best_k_bic": int(baseline_best_k),
        "baseline_vdc_label": int(baseline_vdc_label),
        "baseline_vdc_members": int(baseline_vdc_mask.sum()),
        "bic_optimal_k_counts": {str(k): int(v) for k, v in summary["best_k_bic"].value_counts().sort_index().items()},
        "aic_optimal_k_counts": {str(k): int(v) for k, v in summary["best_k_aic"].value_counts().sort_index().items()},
        "icl_optimal_k_counts": {str(k): int(v) for k, v in summary["best_k_icl"].value_counts().sort_index().items()},
        "ari_median": float(summary["ari"].median()),
        "ari_p16": float(summary["ari"].quantile(0.16)),
        "ari_p84": float(summary["ari"].quantile(0.84)),
        "vi_median": float(summary["variation_of_information"].median()),
        "vdc_survival_probability": float(summary["vdc_survived"].mean()),
        "vdc_jaccard_median": float(summary["vdc_jaccard"].median()),
        "vdc_jaccard_p16": float(summary["vdc_jaccard"].quantile(0.16)),
        "vdc_jaccard_p84": float(summary["vdc_jaccard"].quantile(0.84)),
        "survival_jaccard_threshold": float(args.survival_jaccard_threshold),
    }
    (args.outdir / "bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(df, features.to_numpy(), baseline_centers, centers, summary, membership, args.outdir)
    print(json.dumps(report, indent=2))
    print(f"Wrote bootstrap VDC stability results to {args.outdir}")


if __name__ == "__main__":
    main()
