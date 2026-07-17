from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))

import numpy as np
import pandas as pd
from galpy.util import conversion

from .config import resolve_path
from .dynamics import potential_from_definition


def _integrate_batch(sample: pd.DataFrame, config: dict, definition: dict, batch_size: int = 64) -> pd.DataFrame:
    from galpy.orbit import Orbit
    from galpy.potential import rl, verticalfreq

    pot = potential_from_definition(definition)
    coords = dict(config["coordinates"])
    coords["r0_kpc"] = definition.get("r0_kpc", coords["r0_kpc"])
    coords["v0_kms"] = definition.get("v0_kms", coords["v0_kms"])
    orbit_cfg = config["orbits"]
    time_unit_gyr = conversion.time_in_Gyr(coords["v0_kms"], coords["r0_kpc"])
    rows = []
    for start in range(0, len(sample), batch_size):
        group = sample.iloc[start:start + batch_size].reset_index(drop=True)
        orbit = Orbit(vxvv=np.column_stack([group.RAdeg, group.DEdeg, group.distance_kpc, group.pmRA, group.pmDE, group.RV]),
            radec=True, ro=coords["r0_kpc"], vo=coords["v0_kms"], zo=coords["z_sun_pc"] / 1000,
            solarmotion=coords["solar_peculiar_kms"])
        R0 = np.asarray(orbit.R(use_physical=False))
        vT0 = np.asarray(orbit.vT(use_physical=False))
        omega_z = np.abs(np.asarray(verticalfreq(pot, R0)))
        periods = 2 * np.pi / omega_z * time_unit_gyr
        targets = np.clip(orbit_cfg["vertical_periods"] * periods, orbit_cfg["min_time_gyr"], orbit_cfg["max_time_gyr"])
        # The common grid resolves the shortest vertical period in the batch.
        dt_gyr = np.nanmin(periods) / orbit_cfg["samples_per_period"]
        n_steps = max(1001, int(np.ceil(np.nanmax(targets) / dt_gyr)) + 1)
        times_gyr = np.linspace(0, np.nanmax(targets), n_steps)
        times = times_gyr / time_unit_gyr
        try:
            orbit.integrate(times, pot, method="dop853_c", progressbar=False, numcores=1)
            radii = np.asarray(orbit.R(times, use_physical=True))
            heights = np.asarray(orbit.z(times, use_physical=True))
            if radii.ndim == 1:
                radii, heights = radii[None, :], heights[None, :]
            for index, item in group.iterrows():
                use = times_gyr <= targets[index]
                radius = radii[index, use]
                height = heights[index, use]
                rperi, rap = float(np.nanmin(radius)), float(np.nanmax(radius))
                lz = abs(R0[index] * vT0[index])
                rows.append({"ID": int(item.ID), "potential": definition["name"],
                    "zmax_kpc": float(np.nanmax(abs(height))), "rperi_kpc": rperi, "rap_kpc": rap,
                    "eccentricity": (rap - rperi) / (rap + rperi),
                    "Rg_kpc": float(rl(pot, lz)) * coords["r0_kpc"],
                    "vertical_period_gyr": periods[index], "integration_time_gyr": targets[index],
                    "integration_steps": int(use.sum()), "orbit_valid": True, "error": ""})
        except Exception as exc:
            for _, item in group.iterrows():
                rows.append({"ID": int(item.ID), "potential": definition["name"], "zmax_kpc": np.nan,
                    "rperi_kpc": np.nan, "rap_kpc": np.nan, "eccentricity": np.nan, "Rg_kpc": np.nan,
                    "vertical_period_gyr": np.nan, "integration_time_gyr": np.nan,
                    "integration_steps": 0, "orbit_valid": False, "error": type(exc).__name__})
    return pd.DataFrame(rows)


def integrate_nominal_orbits(sample: pd.DataFrame, config: dict, potential_name: str,
                             max_clusters: int | None = None) -> pd.DataFrame:
    from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import rl, verticalfreq

    if max_clusters is not None:
        sample = sample.head(max_clusters)
    definition = next(x for x in config["potentials"]["models"] if x["name"] == potential_name)
    if definition["kind"] == "MWPotential2014":
        if max_clusters is not None:
            sample = sample.head(max_clusters)
        return _integrate_batch(sample, config, definition)
    pot = potential_from_definition(definition)
    coords = dict(config["coordinates"])
    coords["r0_kpc"] = definition.get("r0_kpc", coords["r0_kpc"])
    coords["v0_kms"] = definition.get("v0_kms", coords["v0_kms"])
    orbit_cfg = config["orbits"]
    time_unit_gyr = conversion.time_in_Gyr(coords["v0_kms"], coords["r0_kpc"])
    rows = []
    for _, item in sample.iterrows():
        orbit = Orbit(vxvv=[item.RAdeg, item.DEdeg, item.distance_kpc, item.pmRA, item.pmDE, item.RV],
                      radec=True, ro=coords["r0_kpc"], vo=coords["v0_kms"],
                      zo=coords["z_sun_pc"] / 1000, solarmotion=coords["solar_peculiar_kms"])
        try:
            R, z = float(orbit.R(use_physical=False)), float(orbit.z(use_physical=False))
            delta = float(estimateDeltaStaeckel(pot, R, z))
            use_c = definition["kind"] == "MWPotential2014"
            finder = actionAngleStaeckel(pot=pot, delta=delta, c=use_c)
            try:
                _, lz, _, _, _, oz = finder.actionsFreqs(orbit, c=use_c)
                oz = abs(float(np.atleast_1d(oz)[0]))
            except NotImplementedError:
                _, lz, _ = finder(orbit, c=use_c)
                oz = abs(float(verticalfreq(pot, R)))
            vertical_period_gyr = 2 * np.pi / oz * time_unit_gyr if oz > 0 else np.nan
            target_gyr = np.clip(orbit_cfg["vertical_periods"] * vertical_period_gyr,
                                 orbit_cfg["min_time_gyr"], orbit_cfg["max_time_gyr"])
            n_steps = max(1001, int(orbit_cfg["samples_per_period"] * target_gyr / vertical_period_gyr) + 1)
            times = np.linspace(0, target_gyr / time_unit_gyr, n_steps)
            method = "dop853_c" if use_c else "dop853"
            orbit.integrate(times, pot, method=method)
            radii = np.asarray(orbit.R(times, use_physical=True))
            heights = np.asarray(orbit.z(times, use_physical=True))
            rperi, rap = float(np.nanmin(radii)), float(np.nanmax(radii))
            eccentricity = (rap - rperi) / (rap + rperi)
            lz_internal = float(np.atleast_1d(lz)[0])
            rg = float(rl(pot, abs(lz_internal))) * coords["r0_kpc"]
            row = {"ID": int(item.ID), "potential": potential_name, "zmax_kpc": float(np.nanmax(abs(heights))),
                   "rperi_kpc": rperi, "rap_kpc": rap, "eccentricity": eccentricity, "Rg_kpc": rg,
                   "vertical_period_gyr": vertical_period_gyr, "integration_time_gyr": target_gyr,
                   "integration_steps": n_steps, "orbit_valid": True, "error": ""}
        except Exception as exc:
            row = {"ID": int(item.ID), "potential": potential_name, "zmax_kpc": np.nan,
                   "rperi_kpc": np.nan, "rap_kpc": np.nan, "eccentricity": np.nan, "Rg_kpc": np.nan,
                   "vertical_period_gyr": np.nan, "integration_time_gyr": np.nan,
                   "integration_steps": 0, "orbit_valid": False, "error": type(exc).__name__}
        rows.append(row)
    return pd.DataFrame(rows)


def run_orbits(config: dict, max_clusters: int | None = None, potential_name: str | None = None) -> dict:
    results = resolve_path(config, "results")
    tables = results / "tables" if max_clusters is None else results / "smoke" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(results / "tables" / "master_sample.csv")
    all_rows = []
    definitions = config["potentials"]["models"]
    if potential_name:
        definitions = [item for item in definitions if item["name"] == potential_name]
    for definition in definitions:
        all_rows.append(integrate_nominal_orbits(sample, config, definition["name"], max_clusters))
    output = pd.concat(all_rows, ignore_index=True)
    suffix = f"_{potential_name}" if potential_name else ""
    output.to_csv(tables / f"orbit_catalogue_nominal{suffix}.csv", index=False)
    primary = output[output["potential"] == config["potentials"]["primary"]]
    return {"profile": "production" if max_clusters is None else "smoke", "rows": len(output), "valid": int(output["orbit_valid"].sum()),
            "primary_valid": int(primary["orbit_valid"].sum())}


def merge_orbit_checkpoints(config: dict, required: list[str]) -> dict:
    tables = resolve_path(config, "results") / "tables"
    expected = int(config["project"]["expected_sample_size"])
    frames = []
    for name in required:
        path = tables / f"orbit_catalogue_nominal_{name}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing orbit checkpoint for {name}")
        frame = pd.read_csv(path)
        if len(frame) != expected or frame["ID"].nunique() != expected:
            raise RuntimeError(f"Incomplete orbit checkpoint for {name}: {len(frame)} rows")
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(tables / "orbit_catalogue_nominal.csv", index=False)
    return {"potentials": required, "rows": len(output), "valid": int(output["orbit_valid"].sum())}
