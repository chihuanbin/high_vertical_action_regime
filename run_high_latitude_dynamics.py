#!/usr/bin/env python3
"""
Build a reproducible HUNT24 high-latitude open-cluster dynamics catalog.

The pipeline implements the v0/v1 science plan:
- select high-latitude and matched disk control open clusters from HUNT24;
- compute Galactocentric positions, orbit summaries, and vertical actions;
- estimate a projected virial/dissolution proxy from member kinematics;
- compare high-latitude clusters with age/R_GC/mass matched disk controls;
- write tables, a JSON report, and core diagnostic figures.

The virial ratio is intentionally labeled as a projected proxy. HUNT24 does not
provide full 6D phase space for most member stars, so this quantity should be
used as a robust dissolution diagnostic, not as an exact binding energy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from scipy import stats

Orbit = None
MWPotential2014 = None
actionAngleStaeckel = None


RO_KPC = 8.178
VO_KMS = 236.9
SOLAR_MOTION = "schoenrich"
G_PC_MSUN_KMS2 = 4.30091e-3
KM_S_TO_PC_MYR = 1.022712165


CLUSTER_COLS = [
    ("Name", (0, 20)),
    ("ID", (21, 25)),
    ("AllNames", (26, 279)),
    ("Type", (280, 281)),
    ("CST", (282, 293)),
    ("N", (294, 300)),
    ("CSTt", (301, 312)),
    ("Nt", (313, 318)),
    ("RAdeg", (319, 331)),
    ("DEdeg", (332, 344)),
    ("GLON", (345, 357)),
    ("GLAT", (358, 369)),
    ("r50", (370, 381)),
    ("rc", (382, 393)),
    ("rt", (394, 405)),
    ("rtot", (406, 417)),
    ("r50pc", (418, 431)),
    ("rcpc", (432, 445)),
    ("rtpc", (446, 459)),
    ("rtotpc", (460, 473)),
    ("pmRA", (474, 487)),
    ("s_pmRA", (488, 499)),
    ("e_pmRA", (500, 510)),
    ("pmDE", (511, 523)),
    ("s_pmDE", (524, 535)),
    ("e_pmDE", (536, 546)),
    ("Plx", (547, 559)),
    ("s_Plx", (560, 571)),
    ("e_Plx", (572, 582)),
    ("dist16", (583, 598)),
    ("dist50", (599, 614)),
    ("dist84", (615, 631)),
    ("Ndist", (632, 637)),
    ("globalPlx", (638, 639)),
    ("X", (640, 656)),
    ("Y", (657, 673)),
    ("Z", (674, 690)),
    ("RV", (691, 704)),
    ("s_RV", (705, 718)),
    ("e_RV", (719, 732)),
    ("n_RV", (733, 737)),
    ("CMDCl2.5", (738, 747)),
    ("CMDCl16", (748, 757)),
    ("CMDCl50", (758, 768)),
    ("CMDCl84", (769, 779)),
    ("CMDCl97.5", (780, 790)),
    ("CMDClHuman", (791, 794)),
    ("logAge16", (795, 806)),
    ("logAge50", (807, 818)),
    ("logAge84", (819, 831)),
    ("AV16", (832, 841)),
    ("AV50", (842, 853)),
    ("AV84", (854, 865)),
    ("diffAV16", (866, 875)),
    ("diffAV50", (876, 887)),
    ("diffAV84", (888, 899)),
    ("MOD16", (900, 912)),
    ("MOD50", (913, 925)),
    ("MOD84", (926, 938)),
    ("r50J", (939, 950)),
    ("rJ", (951, 963)),
    ("r50Jpc", (964, 976)),
    ("rJpc", (977, 990)),
    ("probJ", (991, 1001)),
    ("NJ", (1002, 1006)),
    ("MassJ", (1007, 1022)),
    ("e_MassJ", (1023, 1038)),
    ("MassTot", (1039, 1054)),
    ("e_MassTot", (1055, 1070)),
    ("minClSize", (1071, 1073)),
    ("isMerged", (1074, 1075)),
    ("isGMMMemb", (1076, 1077)),
    ("NXmatches", (1078, 1080)),
    ("XmatchType", (1081, 1097)),
    ("Note", (1098, 1131)),
]

MEMBER_COLS = [
    ("Seq", (0, 7)),
    ("Name", (8, 28)),
    ("ID", (29, 33)),
    ("GaiaDR3", (34, 53)),
    ("inrj", (54, 55)),
    ("inrt", (56, 57)),
    ("Prob", (58, 78)),
    ("RAdeg", (79, 103)),
    ("e_RAdeg", (104, 125)),
    ("DEdeg", (126, 148)),
    ("e_DEdeg", (149, 170)),
    ("GLON", (171, 195)),
    ("GLAT", (196, 219)),
    ("pmRA", (220, 243)),
    ("e_pmRA", (244, 265)),
    ("pmDE", (266, 289)),
    ("e_pmDE", (290, 310)),
    ("Plx", (311, 334)),
    ("e_Plx", (335, 355)),
    ("RUWE", (588, 607)),
    ("Gmag", (788, 807)),
    ("BPmag", (808, 827)),
    ("RPmag", (828, 847)),
    ("RV", (920, 942)),
    ("e_RV", (943, 963)),
    ("Mass50", (1135, 1155)),
]


NUMERIC_CLUSTER_COLS = {
    "ID",
    "CST",
    "N",
    "CSTt",
    "Nt",
    "RAdeg",
    "DEdeg",
    "GLON",
    "GLAT",
    "r50",
    "rc",
    "rt",
    "rtot",
    "r50pc",
    "rcpc",
    "rtpc",
    "rtotpc",
    "pmRA",
    "s_pmRA",
    "e_pmRA",
    "pmDE",
    "s_pmDE",
    "e_pmDE",
    "Plx",
    "s_Plx",
    "e_Plx",
    "dist16",
    "dist50",
    "dist84",
    "Ndist",
    "globalPlx",
    "X",
    "Y",
    "Z",
    "RV",
    "s_RV",
    "e_RV",
    "n_RV",
    "CMDCl2.5",
    "CMDCl16",
    "CMDCl50",
    "CMDCl84",
    "CMDCl97.5",
    "logAge16",
    "logAge50",
    "logAge84",
    "AV16",
    "AV50",
    "AV84",
    "diffAV16",
    "diffAV50",
    "diffAV84",
    "MOD16",
    "MOD50",
    "MOD84",
    "r50J",
    "rJ",
    "r50Jpc",
    "rJpc",
    "probJ",
    "NJ",
    "MassJ",
    "e_MassJ",
    "MassTot",
    "e_MassTot",
    "minClSize",
    "isMerged",
    "isGMMMemb",
    "NXmatches",
}

NUMERIC_MEMBER_COLS = {
    "Seq",
    "ID",
    "GaiaDR3",
    "inrj",
    "inrt",
    "Prob",
    "RAdeg",
    "e_RAdeg",
    "DEdeg",
    "e_DEdeg",
    "GLON",
    "GLAT",
    "pmRA",
    "e_pmRA",
    "pmDE",
    "e_pmDE",
    "Plx",
    "e_Plx",
    "RUWE",
    "Gmag",
    "BPmag",
    "RPmag",
    "RV",
    "e_RV",
    "Mass50",
}


@dataclass(frozen=True)
class SampleCuts:
    high_latitude_deg: float = 20.0
    disk_latitude_deg: float = 10.0
    min_members: int = 50
    max_age_width_dex: float = 0.4
    max_frac_parallax_error: float = 0.1
    min_rv_members: int = 3
    min_prob_j: float = 0.5
    include_moving_groups: bool = False


def _blank_to_nan(value: str) -> str | float:
    text = value.strip()
    return np.nan if text == "" or text == "?" else text


def read_fixed_width(path: Path, columns: list[tuple[str, tuple[int, int]]], numeric: set[str]) -> pd.DataFrame:
    specs = [spec for _, spec in columns]
    names = [name for name, _ in columns]
    df = pd.read_fwf(path, colspecs=specs, names=names, dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].map(_blank_to_nan)
    for col in numeric & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in sorted(set(df.columns) - numeric):
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA})
    return df


def parse_fixed_width_lines(
    lines: list[str], columns: list[tuple[str, tuple[int, int]]], numeric: set[str]
) -> pd.DataFrame:
    rows = []
    for line in lines:
        rows.append({name: _blank_to_nan(line[start:end]) for name, (start, end) in columns})
    df = pd.DataFrame(rows, columns=[name for name, _ in columns])
    for col in numeric & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in sorted(set(df.columns) - numeric):
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA})
    return df


def load_members_for_ids(path: Path, wanted_ids: set[int]) -> pd.DataFrame:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                cluster_id = int(line[29:33])
            except ValueError:
                continue
            if cluster_id in wanted_ids:
                lines.append(line.rstrip("\n"))
    if not lines:
        return pd.DataFrame(columns=[name for name, _ in MEMBER_COLS])
    return parse_fixed_width_lines(lines, MEMBER_COLS, NUMERIC_MEMBER_COLS)


def load_clusters(path: Path) -> pd.DataFrame:
    df = read_fixed_width(path, CLUSTER_COLS, NUMERIC_CLUSTER_COLS)
    df["age_width_dex"] = df["logAge84"] - df["logAge16"]
    df["frac_parallax_error"] = df["e_Plx"] / df["Plx"].replace(0, np.nan)
    df["age_myr"] = 10 ** (df["logAge50"] - 6.0)
    mass = df["MassJ"].where(df["MassJ"].notna() & (df["MassJ"] > 0), df["MassTot"])
    df["mass_for_matching_msun"] = mass
    df["distance_kpc"] = df["dist50"] / 1000.0
    return df


def base_quality_mask(df: pd.DataFrame, cuts: SampleCuts, require_rv: bool = True) -> pd.Series:
    allowed_types = ["o", "m"] if cuts.include_moving_groups else ["o"]
    mask = (
        (df["Type"].isin(allowed_types))
        & (df["N"] >= cuts.min_members)
        & (df["Plx"] > 0)
        & (df["frac_parallax_error"] <= cuts.max_frac_parallax_error)
        & (df["age_width_dex"] <= cuts.max_age_width_dex)
        & (df["logAge50"].notna())
        & (df["distance_kpc"] > 0)
        & (df["mass_for_matching_msun"] > 0)
        & (df["probJ"].fillna(0) >= cuts.min_prob_j)
    )
    if require_rv:
        mask &= (df["n_RV"] >= cuts.min_rv_members) & df["RV"].notna()
    return mask


def select_samples(df: pd.DataFrame, cuts: SampleCuts, require_rv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = base_quality_mask(df, cuts, require_rv=require_rv)
    high = df[quality & (df["GLAT"].abs() > cuts.high_latitude_deg)].copy()
    disk = df[quality & (df["GLAT"].abs() < cuts.disk_latitude_deg)].copy()
    return high, disk


def add_galactocentric_positions(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_orbit_row(row: pd.Series, integration_gyr: float, n_steps: int, action_finder) -> dict[str, float]:
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
    times = np.linspace(0, integration_gyr, n_steps) * u.Gyr
    orbit.integrate(times, MWPotential2014)
    jr, lz, jz = action_finder(orbit)
    return {
        "orbit_zmax_kpc": float(np.atleast_1d(orbit.zmax(use_physical=True))[0]),
        "orbit_eccentricity": float(np.atleast_1d(orbit.e())[0]),
        "orbit_rperi_kpc": float(np.atleast_1d(orbit.rperi(use_physical=True))[0]),
        "orbit_rap_kpc": float(np.atleast_1d(orbit.rap(use_physical=True))[0]),
        "J_R_kpc_kms": float(np.atleast_1d(jr)[0] * RO_KPC * VO_KMS),
        "L_z_kpc_kms": float(np.atleast_1d(lz)[0] * RO_KPC * VO_KMS),
        "J_z_kpc_kms": float(np.atleast_1d(jz)[0] * RO_KPC * VO_KMS),
    }


def add_orbit_quantities(df: pd.DataFrame, integration_gyr: float, n_steps: int) -> pd.DataFrame:
    global Orbit, MWPotential2014, actionAngleStaeckel
    if Orbit is None or MWPotential2014 is None or actionAngleStaeckel is None:
        try:
            from galpy.actionAngle import actionAngleStaeckel as _actionAngleStaeckel
            from galpy.orbit import Orbit as _Orbit
            from galpy.potential import MWPotential2014 as _MWPotential2014
        except ImportError as exc:  # pragma: no cover - exercised by users without galpy
            raise SystemExit(
                "This pipeline requires galpy for orbit mode. Install requirements.txt or use --skip-orbits."
            ) from exc
        Orbit = _Orbit
        MWPotential2014 = _MWPotential2014
        actionAngleStaeckel = _actionAngleStaeckel
    out = add_galactocentric_positions(df)
    action_finder = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    records: list[dict[str, float]] = []
    for _, row in out.iterrows():
        try:
            records.append(compute_orbit_row(row, integration_gyr, n_steps, action_finder))
        except Exception as exc:
            warnings.warn(f"Orbit failed for {row['Name']} (ID={row['ID']}): {exc}")
            records.append(
                {
                    "orbit_zmax_kpc": np.nan,
                    "orbit_eccentricity": np.nan,
                    "orbit_rperi_kpc": np.nan,
                    "orbit_rap_kpc": np.nan,
                    "J_R_kpc_kms": np.nan,
                    "L_z_kpc_kms": np.nan,
                    "J_z_kpc_kms": np.nan,
                }
            )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def add_orbit_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    out = add_galactocentric_positions(df)
    out["orbit_zmax_kpc"] = out["abs_Z_kpc"]
    out["orbit_eccentricity"] = np.nan
    out["orbit_rperi_kpc"] = np.nan
    out["orbit_rap_kpc"] = np.nan
    out["J_R_kpc_kms"] = np.nan
    out["L_z_kpc_kms"] = np.nan
    out["J_z_kpc_kms"] = np.nan
    return out


def robust_sigma(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return np.nan
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr)))
    sigma = 1.4826 * mad
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.nanstd(arr, ddof=1)
    return float(sigma)


def tangential_velocity_dispersion_kms(members: pd.DataFrame, distance_kpc: float) -> float:
    pmra_sig = robust_sigma(members["pmRA"])
    pmde_sig = robust_sigma(members["pmDE"])
    if not np.isfinite(pmra_sig) or not np.isfinite(pmde_sig):
        return np.nan
    return 4.74047 * distance_kpc * math.sqrt(0.5 * (pmra_sig**2 + pmde_sig**2))


def projected_virial_proxy(members: pd.DataFrame, cluster: pd.Series) -> dict[str, float]:
    quality = members[(members["Prob"] >= 0.5) & (members["Plx"] > 0)].copy()
    if quality.empty:
        quality = members.copy()
    if "inrj" in quality:
        in_radius = quality[quality["inrj"] == 1]
        if len(in_radius) >= 10:
            quality = in_radius

    sigma_pm = tangential_velocity_dispersion_kms(quality, cluster["distance_kpc"])
    sigma_rv = robust_sigma(quality.loc[quality["RV"].notna(), "RV"])
    n_rv_members = int(quality["RV"].notna().sum())
    if np.isfinite(sigma_rv) and n_rv_members >= 5:
        sigma_3d = math.sqrt(max(0.0, 2.0 * sigma_pm**2 + sigma_rv**2))
        velocity_basis = "pm+rv"
    elif np.isfinite(sigma_pm):
        sigma_3d = math.sqrt(3.0) * sigma_pm
        velocity_basis = "pm_only_isotropic"
    else:
        sigma_3d = np.nan
        velocity_basis = "insufficient"

    mass = cluster["MassJ"] if pd.notna(cluster["MassJ"]) and cluster["MassJ"] > 0 else cluster["MassTot"]
    radius = cluster["r50Jpc"] if pd.notna(cluster["r50Jpc"]) and cluster["r50Jpc"] > 0 else cluster["r50pc"]
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(radius) or radius <= 0:
        alpha_vir = np.nan
    elif np.isfinite(sigma_3d):
        sigma_1d = sigma_3d / math.sqrt(3.0)
        alpha_vir = 5.0 * sigma_1d**2 * radius / (G_PC_MSUN_KMS2 * mass)
    else:
        alpha_vir = np.nan

    if np.isfinite(sigma_3d) and sigma_3d > 0 and np.isfinite(radius):
        t_cross_myr = radius / (sigma_3d * KM_S_TO_PC_MYR)
    else:
        t_cross_myr = np.nan
    n_eff = max(int(quality["GaiaDR3"].nunique()), 2)
    if np.isfinite(t_cross_myr) and n_eff > 2:
        t_relax_myr = 0.1 * n_eff / math.log(n_eff) * t_cross_myr
    else:
        t_relax_myr = np.nan
    dsi = cluster["age_myr"] / t_relax_myr if np.isfinite(t_relax_myr) and t_relax_myr > 0 else np.nan

    return {
        "member_count_used": int(len(quality)),
        "member_rv_count_used": n_rv_members,
        "sigma_pm_tan_kms": sigma_pm,
        "sigma_rv_kms": sigma_rv,
        "sigma_3d_proxy_kms": sigma_3d,
        "velocity_basis": velocity_basis,
        "virial_alpha_proxy": alpha_vir,
        "Lambda_proxy": math.log10(alpha_vir) if np.isfinite(alpha_vir) and alpha_vir > 0 else np.nan,
        "t_cross_myr": t_cross_myr,
        "t_relax_myr": t_relax_myr,
        "DSI": dsi,
    }


def add_member_dissolution_metrics(sample: pd.DataFrame, members_path: Path) -> pd.DataFrame:
    wanted_ids = set(sample["ID"].dropna().astype(int).tolist())
    members = load_members_for_ids(members_path, wanted_ids)
    grouped = {cluster_id: group for cluster_id, group in members.groupby("ID", sort=False)}

    records = []
    for _, cluster in sample.iterrows():
        cluster_id = int(cluster["ID"])
        group = grouped.get(cluster_id, pd.DataFrame(columns=members.columns))
        records.append(projected_virial_proxy(group, cluster))
    return pd.concat([sample.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def match_disk_controls(high: pd.DataFrame, disk: pd.DataFrame, controls_per_cluster: int = 3) -> pd.DataFrame:
    controls: list[pd.DataFrame] = []
    if high.empty or disk.empty:
        return disk.iloc[0:0].copy()
    high = high[high["R_gc_kpc"].notna()].copy()
    disk = disk[disk["R_gc_kpc"].notna()].copy()
    if high.empty or disk.empty:
        return disk.iloc[0:0].copy()
    disk_features = disk[["logAge50", "R_gc_kpc", "mass_for_matching_msun"]].copy()
    disk_features["log_mass"] = np.log10(disk_features["mass_for_matching_msun"])
    feature_cols = ["logAge50", "R_gc_kpc", "log_mass"]
    scale = disk_features[feature_cols].std(ddof=0).replace(0, 1.0)

    for _, cluster in high.iterrows():
        target = pd.Series(
            {
                "logAge50": cluster["logAge50"],
                "R_gc_kpc": cluster["R_gc_kpc"],
                "log_mass": math.log10(cluster["mass_for_matching_msun"]),
            }
        )
        distance = (((disk_features[feature_cols] - target) / scale) ** 2).sum(axis=1) ** 0.5
        nearest = disk.loc[distance.nsmallest(controls_per_cluster).index].copy()
        nearest["matched_high_ID"] = cluster["ID"]
        nearest["match_distance"] = distance.loc[nearest.index].to_numpy()
        controls.append(nearest)
    matched = pd.concat(controls, ignore_index=True)
    return matched.drop_duplicates(subset=["ID"]).reset_index(drop=True)


def fit_disk_jz_baseline(disk: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = sample.copy()
    train = disk[(disk["J_z_kpc_kms"] > 0) & disk["logAge50"].notna() & disk["R_gc_kpc"].notna()].copy()
    if len(train) < 8:
        out["logJz_disk_pred"] = np.nan
        out["Delta_logJz"] = np.nan
        return out, {"status": "insufficient_disk_controls", "n_train": int(len(train))}

    y = np.log10(train["J_z_kpc_kms"])
    x1 = train["logAge50"] - 8.0
    x2 = train["R_gc_kpc"] - train["R_gc_kpc"].median()
    design = np.column_stack([np.ones(len(train)), x1, x2])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = coeff[0] + coeff[1] * (out["logAge50"] - 8.0) + coeff[2] * (
        out["R_gc_kpc"] - train["R_gc_kpc"].median()
    )
    out["logJz_disk_pred"] = pred
    out["Delta_logJz"] = np.log10(out["J_z_kpc_kms"].where(out["J_z_kpc_kms"] > 0)) - pred
    residual = y - design @ coeff
    return out, {
        "status": "ok",
        "n_train": int(len(train)),
        "intercept": float(coeff[0]),
        "slope_logAge": float(coeff[1]),
        "slope_R_gc": float(coeff[2]),
        "scatter_dex": float(np.nanstd(residual, ddof=3)),
    }


def compare_groups(high: pd.DataFrame, control: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[str, float | int | None]]:
    results: dict[str, dict[str, float | int | None]] = {}
    for col in columns:
        h = high[col].replace([np.inf, -np.inf], np.nan).dropna()
        c = control[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(h) < 3 or len(c) < 3:
            results[col] = {"n_high": int(len(h)), "n_control": int(len(c)), "status": "insufficient"}
            continue
        ks = stats.ks_2samp(h, c, alternative="two-sided", method="auto")
        ad = stats.anderson_ksamp([h.to_numpy(), c.to_numpy()])
        mw = stats.mannwhitneyu(h, c, alternative="two-sided")
        results[col] = {
            "n_high": int(len(h)),
            "n_control": int(len(c)),
            "median_high": float(np.nanmedian(h)),
            "median_control": float(np.nanmedian(c)),
            "median_difference_high_minus_control": float(np.nanmedian(h) - np.nanmedian(c)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "anderson_darling_statistic": float(ad.statistic),
            "anderson_darling_significance_level": float(ad.significance_level),
            "mann_whitney_u_pvalue": float(mw.pvalue),
            "status": "ok",
        }
    return results


def linear_slope(x: pd.Series, y: pd.Series) -> dict[str, float | int | str]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4:
        return {"status": "insufficient", "n": int(len(data))}
    slope, intercept, r, p, stderr = stats.linregress(data["x"], data["y"])
    return {
        "status": "ok",
        "n": int(len(data)),
        "slope": float(slope),
        "intercept": float(intercept),
        "rvalue": float(r),
        "pvalue": float(p),
        "stderr": float(stderr),
    }


def save_figures(high: pd.DataFrame, control: pd.DataFrame, full: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 5.0))
    plt.scatter(full["R_gc_kpc"], full["galcen_z_kpc"], s=8, c="0.78", label="Quality HUNT24 OCs")
    plt.scatter(control["R_gc_kpc"], control["galcen_z_kpc"], s=18, c="#2878b5", label="Matched disk")
    plt.scatter(high["R_gc_kpc"], high["galcen_z_kpc"], s=28, c="#c43c39", label="High latitude")
    plt.axhline(0, color="0.4", lw=0.8)
    plt.xlabel(r"$R_{\rm GC}$ [kpc]")
    plt.ylabel(r"$Z_{\rm GC}$ [kpc]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig1_rgc_z_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    delta_values = pd.concat([high["Delta_logJz"], control["Delta_logJz"]]).dropna()
    if len(delta_values) > 1 and delta_values.min() != delta_values.max():
        bins = np.linspace(delta_values.min(), delta_values.max(), 18)
        plt.hist(control["Delta_logJz"].dropna(), bins=bins, alpha=0.65, density=True, color="#2878b5", label="Matched disk")
        plt.hist(high["Delta_logJz"].dropna(), bins=bins, alpha=0.65, density=True, color="#c43c39", label="High latitude")
        plt.axvline(0, color="0.25", lw=1.0)
    else:
        plt.text(0.5, 0.5, "Jz unavailable: run without --skip-orbits", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel(r"$\Delta \log J_z$ relative to disk baseline")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig2_delta_logjz_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    lambda_values = pd.concat([high["Lambda_proxy"], control["Lambda_proxy"]]).dropna()
    if len(lambda_values) > 0:
        bins = np.linspace(lambda_values.min(), lambda_values.max(), 18)
    else:
        bins = 10
    plt.hist(control["Lambda_proxy"].dropna(), bins=bins, alpha=0.65, density=True, color="#2878b5", label="Matched disk")
    plt.hist(high["Lambda_proxy"].dropna(), bins=bins, alpha=0.65, density=True, color="#c43c39", label="High latitude")
    plt.axvline(0, color="0.25", lw=1.0)
    plt.xlabel(r"$\Lambda_{\rm proxy}=\log_{10}(\alpha_{\rm vir,proxy})$")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / "fig3_lambda_proxy_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    sc = plt.scatter(
        high["Lambda_proxy"],
        high["Delta_logJz"],
        c=high["logAge50"],
        s=48,
        cmap="viridis",
        edgecolor="0.2",
        linewidth=0.3,
    )
    plt.axhline(0, color="0.5", lw=0.8)
    plt.axvline(0, color="0.5", lw=0.8)
    plt.xlabel(r"$\Lambda_{\rm proxy}$")
    plt.ylabel(r"$\Delta \log J_z$")
    cb = plt.colorbar(sc)
    cb.set_label(r"$\log_{10}({\rm age/yr})$")
    plt.tight_layout()
    plt.savefig(figdir / "fig4_high_latitude_dissolution_sequence.png", dpi=220)
    plt.close()


def write_report(
    high: pd.DataFrame,
    control: pd.DataFrame,
    quality: pd.DataFrame,
    cuts: SampleCuts,
    baseline: dict[str, float],
    outdir: Path,
) -> dict:
    comparisons = compare_groups(high, control, ["Delta_logJz", "Lambda_proxy", "orbit_zmax_kpc", "DSI"])
    slopes = {
        "high_Delta_logJz_vs_logAge": linear_slope(high["logAge50"], high["Delta_logJz"]),
        "high_Lambda_proxy_vs_logAge": linear_slope(high["logAge50"], high["Lambda_proxy"]),
        "high_Lambda_proxy_vs_zmax": linear_slope(high["orbit_zmax_kpc"], high["Lambda_proxy"]),
        "high_Delta_logJz_vs_Lambda_proxy": linear_slope(high["Lambda_proxy"], high["Delta_logJz"]),
    }
    report = {
        "cuts": cuts.__dict__,
        "counts": {
            "quality_clusters_after_cuts": int(len(quality)),
            "high_latitude": int(len(high)),
            "matched_disk_control": int(len(control)),
        },
        "disk_jz_baseline": baseline,
        "group_comparisons": comparisons,
        "high_latitude_slopes": slopes,
        "notes": [
            "Lambda_proxy is log10 of a projected virial alpha proxy, not a full 6D binding-energy ratio.",
            "Main sample requires cluster-level RV support; 5D-only candidates should be handled as sensitivity tests.",
            "Delta_logJz is measured relative to a linear disk baseline in logAge and R_GC fitted from matched disk controls.",
        ],
    }
    (outdir / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=Path("hunt24/clusters.dat"))
    parser.add_argument("--members", type=Path, default=Path("hunt24/members.dat"))
    parser.add_argument("--outdir", type=Path, default=Path("results/high_latitude_dynamics"))
    parser.add_argument("--high-latitude-deg", type=float, default=20.0)
    parser.add_argument("--disk-latitude-deg", type=float, default=10.0)
    parser.add_argument("--min-members", type=int, default=50)
    parser.add_argument("--max-age-width-dex", type=float, default=0.4)
    parser.add_argument("--max-frac-parallax-error", type=float, default=0.1)
    parser.add_argument("--min-rv-members", type=int, default=3)
    parser.add_argument("--min-prob-j", type=float, default=0.5)
    parser.add_argument(
        "--include-moving-groups",
        action="store_true",
        help="Include HUNT24 Type=m objects as a sensitivity sample.",
    )
    parser.add_argument("--controls-per-cluster", type=int, default=3)
    parser.add_argument("--integration-gyr", type=float, default=5.0)
    parser.add_argument("--orbit-steps", type=int, default=401)
    parser.add_argument("--skip-orbits", action="store_true", help="Skip galpy orbit/action calculations for a fast structural run.")
    parser.add_argument(
        "--include-5d",
        action="store_true",
        help="Include clusters without reliable RV. Automatically skips orbit/action calculations.",
    )
    parser.add_argument("--max-high", type=int, default=None, help="Debug/smoke-test limit for high-latitude clusters.")
    parser.add_argument("--max-disk", type=int, default=None, help="Debug/smoke-test limit for disk clusters before matching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cuts = SampleCuts(
        high_latitude_deg=args.high_latitude_deg,
        disk_latitude_deg=args.disk_latitude_deg,
        min_members=args.min_members,
        max_age_width_dex=args.max_age_width_dex,
        max_frac_parallax_error=args.max_frac_parallax_error,
        min_rv_members=args.min_rv_members,
        min_prob_j=args.min_prob_j,
        include_moving_groups=args.include_moving_groups,
    )

    clusters = load_clusters(args.clusters)
    require_rv = not args.include_5d
    if args.include_5d:
        args.skip_orbits = True

    high, disk = select_samples(clusters, cuts, require_rv=require_rv)
    if args.max_high is not None:
        high = high.head(args.max_high).copy()
    if args.max_disk is not None:
        disk = disk.head(args.max_disk).copy()

    quality = clusters[base_quality_mask(clusters, cuts, require_rv=require_rv)].copy()
    quality_positions = add_galactocentric_positions(quality)
    high_positions = quality_positions[quality_positions["ID"].isin(high["ID"])].copy()
    disk_positions = quality_positions[quality_positions["ID"].isin(disk["ID"])].copy()
    matched_disk_positions = match_disk_controls(
        high_positions, disk_positions, controls_per_cluster=args.controls_per_cluster
    )

    orbit_input = pd.concat([high_positions, matched_disk_positions], ignore_index=True).drop_duplicates(subset=["ID"])
    if args.skip_orbits:
        orbit_catalog = add_orbit_placeholders(orbit_input)
    else:
        orbit_catalog = add_orbit_quantities(orbit_input, integration_gyr=args.integration_gyr, n_steps=args.orbit_steps)
    high_orbits = orbit_catalog[orbit_catalog["ID"].isin(high["ID"])].copy()
    matched_disk = orbit_catalog[orbit_catalog["ID"].isin(matched_disk_positions["ID"])].copy()
    combined_ids = pd.concat([high_orbits["ID"], matched_disk["ID"]]).drop_duplicates()
    combined = orbit_catalog.copy()
    combined = add_member_dissolution_metrics(combined, args.members)

    high_final = combined[combined["ID"].isin(high_orbits["ID"])].copy()
    control_final = combined[combined["ID"].isin(matched_disk["ID"])].copy()
    baseline_sample = pd.concat([high_final, control_final], ignore_index=True)
    baseline_sample, baseline = fit_disk_jz_baseline(control_final, baseline_sample)
    high_final = baseline_sample[baseline_sample["ID"].isin(high_orbits["ID"])].copy()
    control_final = baseline_sample[baseline_sample["ID"].isin(matched_disk["ID"])].copy()

    high_final.to_csv(args.outdir / "high_latitude_cluster_catalog.csv", index=False)
    control_final.to_csv(args.outdir / "matched_disk_control_catalog.csv", index=False)
    orbit_catalog.to_csv(args.outdir / "candidate_open_cluster_orbit_catalog.csv", index=False)
    quality_positions.to_csv(args.outdir / "quality_open_cluster_position_catalog.csv", index=False)
    pd.DataFrame({"selected_ID": combined_ids}).to_csv(args.outdir / "selected_cluster_ids.csv", index=False)

    save_figures(high_final, control_final, quality_positions, args.outdir)
    report = write_report(high_final, control_final, quality_positions, cuts, baseline, args.outdir)

    print(json.dumps(report["counts"], indent=2))
    print(f"Wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
