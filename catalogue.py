from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    start: int
    end: int
    numeric: bool = True
    unit: str = ""
    description: str = ""


CLUSTER_COLUMNS = [
    Column("Name", 0, 20, False, "", "Main accepted cluster name"),
    Column("ID", 21, 25, True, "", "HUNT24 internal identifier"),
    Column("AllNames", 26, 279, False, "", "Comma-separated aliases"),
    Column("Type", 280, 281, False, "", "HUNT24 object classification"),
    Column("CST", 282, 293, True, "", "Cluster significance statistic"),
    Column("N", 294, 300, True, "", "Number of member stars"),
    Column("CSTt", 301, 312, True), Column("Nt", 313, 318, True),
    Column("RAdeg", 319, 331, True, "deg"), Column("DEdeg", 332, 344, True, "deg"),
    Column("GLON", 345, 357, True, "deg"), Column("GLAT", 358, 369, True, "deg"),
    Column("r50", 370, 381, True, "deg"), Column("rc", 382, 393, True, "deg"),
    Column("rt", 394, 405, True, "deg"), Column("rtot", 406, 417, True, "deg"),
    Column("r50pc", 418, 431, True, "pc"), Column("rcpc", 432, 445, True, "pc"),
    Column("rtpc", 446, 459, True, "pc"), Column("rtotpc", 460, 473, True, "pc"),
    Column("pmRA", 474, 487, True, "mas/yr"), Column("s_pmRA", 488, 499, True, "mas/yr"),
    Column("e_pmRA", 500, 510, True, "mas/yr"), Column("pmDE", 511, 523, True, "mas/yr"),
    Column("s_pmDE", 524, 535, True, "mas/yr"), Column("e_pmDE", 536, 546, True, "mas/yr"),
    Column("Plx", 547, 559, True, "mas"), Column("s_Plx", 560, 571, True, "mas"),
    Column("e_Plx", 572, 582, True, "mas"), Column("dist16", 583, 598, True, "pc"),
    Column("dist50", 599, 614, True, "pc"), Column("dist84", 615, 631, True, "pc"),
    Column("Ndist", 632, 637, True), Column("globalPlx", 638, 639, True),
    Column("X", 640, 656, True, "pc"), Column("Y", 657, 673, True, "pc"),
    Column("Z", 674, 690, True, "pc"), Column("RV", 691, 704, True, "km/s"),
    Column("s_RV", 705, 718, True, "km/s"), Column("e_RV", 719, 732, True, "km/s"),
    Column("n_RV", 733, 737, True), Column("CMDCl2.5", 738, 747, True),
    Column("CMDCl16", 748, 757, True), Column("CMDCl50", 758, 768, True),
    Column("CMDCl84", 769, 779, True), Column("CMDCl97.5", 780, 790, True),
    Column("CMDClHuman", 791, 794, False), Column("logAge16", 795, 806, True, "log10(yr)"),
    Column("logAge50", 807, 818, True, "log10(yr)"), Column("logAge84", 819, 831, True, "log10(yr)"),
    Column("AV16", 832, 841, True, "mag"), Column("AV50", 842, 853, True, "mag"),
    Column("AV84", 854, 865, True, "mag"), Column("diffAV16", 866, 875, True, "mag"),
    Column("diffAV50", 876, 887, True, "mag"), Column("diffAV84", 888, 899, True, "mag"),
    Column("MOD16", 900, 912, True, "mag"), Column("MOD50", 913, 925, True, "mag"),
    Column("MOD84", 926, 938, True, "mag"), Column("r50J", 939, 950, True, "deg"),
    Column("rJ", 951, 963, True, "deg"), Column("r50Jpc", 964, 976, True, "pc"),
    Column("rJpc", 977, 990, True, "pc"), Column("probJ", 991, 1001, True),
    Column("NJ", 1002, 1006, True), Column("MassJ", 1007, 1022, True, "Msun"),
    Column("e_MassJ", 1023, 1038, True, "Msun"), Column("MassTot", 1039, 1054, True, "Msun"),
    Column("e_MassTot", 1055, 1070, True, "Msun"), Column("minClSize", 1071, 1073, True),
    Column("isMerged", 1074, 1075, True), Column("isGMMMemb", 1076, 1077, True),
    Column("NXmatches", 1078, 1080, True), Column("XmatchType", 1081, 1097, False),
    Column("Note", 1098, 1131, False),
]

CROSSMATCH_COLUMNS = [
    Column("ID", 0, 4, True), Column("NameLit", 5, 26, False),
    Column("SourceCat", 27, 43, False), Column("TypeSourceCat", 44, 72, False),
    Column("Sep", 73, 83, True, "deg"), Column("SepTidal", 84, 95, True),
    Column("SepTidalLit", 96, 107, True), Column("SepTidalData", 108, 119, True),
    Column("pmRASep", 120, 130, True, "mas/yr"), Column("pmRASigma", 131, 142, True),
    Column("pmDESep", 143, 153, True, "mas/yr"), Column("pmDESigma", 154, 164, True),
    Column("PlxSep", 165, 175, True, "mas"), Column("PlxSigma", 176, 186, True),
    Column("maxSigma", 187, 196, True), Column("meanSigma", 197, 207, True),
]


def read_fixed_width(path: str | Path, columns: list[Column]) -> pd.DataFrame:
    specs = [(column.start, column.end) for column in columns]
    names = [column.name for column in columns]
    frame = pd.read_fwf(path, colspecs=specs, names=names, dtype=str, keep_default_na=False)
    for column in columns:
        values = frame[column.name].str.strip().replace({"": pd.NA, "?": pd.NA})
        frame[column.name] = pd.to_numeric(values, errors="coerce") if column.numeric else values.astype("string")
    return frame


def load_clusters(path: str | Path) -> pd.DataFrame:
    frame = read_fixed_width(path, CLUSTER_COLUMNS)
    frame["age_width_dex"] = frame["logAge84"] - frame["logAge16"]
    frame["fractional_parallax_error"] = frame["e_Plx"] / frame["Plx"].replace(0, np.nan)
    frame["distance_kpc"] = frame["dist50"] / 1000.0
    frame["distance_err_minus_kpc"] = (frame["dist50"] - frame["dist16"]) / 1000.0
    frame["distance_err_plus_kpc"] = (frame["dist84"] - frame["dist50"]) / 1000.0
    frame["mass_msun"] = frame["MassJ"].where(frame["MassJ"] > 0, frame["MassTot"])
    frame["mass_error_msun"] = frame["e_MassJ"].where(frame["MassJ"] > 0, frame["e_MassTot"])
    return frame


def sequential_cutflow(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    cuts = config["sample"]
    steps = [
        ("raw_catalogue", pd.Series(True, index=frame.index)),
        ("object_type", frame["Type"].isin(cuts["allowed_types"])),
        ("minimum_members", frame["N"] >= cuts["min_members"]),
        ("positive_parallax", frame["Plx"] > 0),
        ("parallax_precision", frame["fractional_parallax_error"] <= cuts["max_fractional_parallax_error"]),
        ("finite_age", frame["logAge50"].notna()),
        ("age_precision", frame["age_width_dex"] <= cuts["max_age_width_dex"]),
        ("positive_distance", frame["distance_kpc"] > 0),
        ("positive_mass", frame["mass_msun"] > 0),
        ("jacobi_probability", frame["probJ"].fillna(0) >= cuts["min_prob_j"]),
        ("rv_members", frame["n_RV"] >= cuts["min_rv_members"]),
        ("finite_radial_velocity", frame["RV"].notna()),
    ]
    active = pd.Series(True, index=frame.index)
    rows = []
    previous = len(frame)
    for order, (name, condition) in enumerate(steps):
        active &= condition.fillna(False)
        remaining = int(active.sum())
        rows.append({"order": order, "cut": name, "remaining": remaining, "removed_at_step": previous - remaining})
        previous = remaining
    return pd.DataFrame(rows), active


def column_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        {"column": c.name, "start_byte_zero_based": c.start, "end_byte_exclusive": c.end,
         "unit": c.unit, "description": c.description, "numeric": c.numeric}
        for c in CLUSTER_COLUMNS
    )


def duplicate_flags(frame: pd.DataFrame, crossmatches: pd.DataFrame | None = None) -> pd.DataFrame:
    out = frame[["ID", "Name", "AllNames", "NXmatches", "XmatchType", "isMerged"]].copy()
    out["duplicate_id"] = out["ID"].duplicated(keep=False)
    out["duplicate_name"] = out["Name"].str.casefold().duplicated(keep=False)
    out["complex_crossmatch"] = out["XmatchType"].fillna("").str.contains("m")
    if crossmatches is not None:
        counts = crossmatches.groupby("ID", dropna=True).size().rename("crossmatch_rows")
        out = out.join(counts, on="ID")
    else:
        out["crossmatch_rows"] = np.nan
    out["possible_duplicate"] = out[["duplicate_id", "duplicate_name", "complex_crossmatch"]].any(axis=1)
    return out
