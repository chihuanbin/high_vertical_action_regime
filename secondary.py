from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .catalogue import load_clusters, sequential_cutflow
from .config import resolve_path


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def selection_diagnostics(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = table[table["potential"] == config["potentials"]["primary"]].copy()
    threshold = primary["log10_Jz_over_abs_Lz"].quantile(0.9)
    primary["high_tail"] = primary["log10_Jz_over_abs_Lz"] >= threshold
    covariates = ["distance_kpc", "GLON", "GLAT", "R_kpc", "phi_rad", "z_kpc", "logAge50",
        "age_width_dex", "fractional_parallax_error", "e_pmRA", "e_pmDE", "e_RV", "n_RV", "N",
        "mass_msun", "AV50", "probJ"]
    balance = pd.DataFrame({"covariate": covariates,
        "standardized_mean_difference": [standardized_mean_difference(primary.loc[primary.high_tail, c], primary.loc[~primary.high_tail, c]) for c in covariates]})

    parent = load_clusters(resolve_path(config, "clusters"))
    _, selected = sequential_cutflow(parent, config)
    features = ["distance_kpc", "GLON", "GLAT", "logAge50", "age_width_dex", "fractional_parallax_error", "N", "probJ", "AV50"]
    work = parent[features].replace([np.inf, -np.inf], np.nan).copy()
    work = work.fillna(work.median(numeric_only=True))
    y = selected.astype(int).to_numpy()
    cv = StratifiedKFold(config["selection"]["cv_folds"], shuffle=True, random_state=config["project"]["seed"])
    selector = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=10, cv=cv, max_iter=2000, class_weight="balanced", solver="liblinear"))
    selector.fit(work, y)
    chosen_c = float(selector[-1].C_[0])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=chosen_c, max_iter=2000, class_weight="balanced", solver="liblinear"))
    probability = cross_val_predict(model, work, y, cv=cv, method="predict_proba")[:, 1]
    selection = pd.DataFrame({"ID": parent["ID"].astype(int), "selected": y, "selection_probability_cv": probability})
    selection["brier_score"] = brier_score_loss(y, probability)
    selection["selected_regularization_C"] = chosen_c
    return balance, selection


def exploratory_gmm(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = table[table["potential"] == config["potentials"]["primary"]].copy()
    features = primary[["JR_kpc_kms", "Jz_kpc_kms", "Lz_kpc_kms"]].copy()
    features = np.log10(features.abs().where(features.abs() > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    rows, models = [], {}
    for covariance in config["gmm"]["covariance_types"]:
        for k in range(1, config["gmm"]["max_components"] + 1):
            model = GaussianMixture(k, covariance_type=covariance, n_init=config["gmm"]["initializations"],
                                    random_state=config["project"]["seed"]).fit(x)
            probabilities = model.predict_proba(x)
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1)))
            rows.append({"covariance_type": covariance, "k": k, "bic": model.bic(x), "aic": model.aic(x),
                         "icl": model.bic(x) + 2 * entropy, "membership_entropy": entropy / len(x)})
            models[(covariance, k)] = model
    criteria = pd.DataFrame(rows)
    best = criteria.loc[criteria["bic"].idxmin()]
    model = models[(best.covariance_type, int(best.k))]
    labels = model.predict(x)
    response = primary.loc[features.index, "log10_Jz_over_abs_Lz"]
    medians = response.groupby(labels).median()
    high_label = int(medians.idxmax())
    membership = primary.loc[features.index, ["ID", "Name", "log10_Jz_over_abs_Lz"]].copy()
    membership["gmm_label"] = labels
    membership["gmm_high_action_probability"] = model.predict_proba(x)[:, high_label]
    membership["descriptive_high_action_component"] = labels == high_label
    return criteria, membership


def run_secondary(config: dict) -> dict:
    tables = resolve_path(config, "results") / "tables"
    table = pd.read_csv(tables / "analysis_catalogue.csv")
    expected = config["project"]["expected_sample_size"] * table["potential"].nunique()
    if len(table) != expected:
        raise RuntimeError("Secondary analyses require the complete production catalogue")
    balance, selection = selection_diagnostics(table, config)
    criteria, membership = exploratory_gmm(table, config)
    balance.to_csv(tables / "selection_balance.csv", index=False)
    selection.to_csv(tables / "selection_model.csv", index=False)
    criteria.to_csv(tables / "gmm_model_selection.csv", index=False)
    membership.to_csv(tables / "gmm_membership.csv", index=False)
    return {"balance_rows": len(balance), "gmm_grid_rows": len(criteria), "gmm_membership_rows": len(membership)}
