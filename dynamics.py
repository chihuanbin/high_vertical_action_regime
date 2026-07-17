from __future__ import annotations

import copy
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".mplconfig").resolve()))
os.environ.setdefault("GALPY_CONFIG", str((Path("results") / ".galpyrc").resolve()))

import numpy as np
import pandas as pd


def potential_from_definition(definition: dict):
    if definition["kind"] == "McMillan17":
        from galpy.potential.mwpotentials import McMillan17
        return copy.deepcopy(McMillan17)
    if definition["kind"] == "MWPotential2014":
        from galpy.potential import MWPotential2014
        potential = copy.deepcopy(MWPotential2014)
        scale = float(definition.get("disk_scale", 1.0))
        if scale != 1.0:
            potential[1] = potential[1] * scale
        return potential
    raise ValueError(f"Unsupported potential kind: {definition['kind']}")


def potential_table(config: dict) -> pd.DataFrame:
    rows = []
    for item in config["potentials"]["models"]:
        pot = potential_from_definition(item)
        rows.append({
            "name": item["name"], "implementation": f"galpy {item['kind']}",
            "disk_scale": item.get("disk_scale", 1.0),
            "components": ";".join(type(component).__name__ for component in pot),
            "r0_kpc": item.get("r0_kpc", config["coordinates"]["r0_kpc"]),
            "v0_kms": item.get("v0_kms", config["coordinates"]["v0_kms"]),
            "z_sun_pc": config["coordinates"]["z_sun_pc"],
            "solar_peculiar_kms": ",".join(map(str, config["coordinates"]["solar_peculiar_kms"])),
        })
    return pd.DataFrame(rows)


def compute_actions(phase_space: pd.DataFrame, config: dict, potential_name: str) -> pd.DataFrame:
    from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
    from galpy.orbit import Orbit

    definition = next(item for item in config["potentials"]["models"] if item["name"] == potential_name)
    pot = potential_from_definition(definition)
    coords = dict(config["coordinates"])
    coords["r0_kpc"] = definition.get("r0_kpc", coords["r0_kpc"])
    coords["v0_kms"] = definition.get("v0_kms", coords["v0_kms"])
    valid = phase_space["input_valid"].to_numpy(bool)
    work = phase_space.loc[valid].reset_index(drop=True)
    orbit = Orbit(
        vxvv=[work["RAdeg"].to_numpy(), work["DEdeg"].to_numpy(), work["distance_kpc"].to_numpy(),
              work["pmRA"].to_numpy(), work["pmDE"].to_numpy(), work["RV"].to_numpy()],
        radec=True, ro=coords["r0_kpc"], vo=coords["v0_kms"], zo=coords["z_sun_pc"] / 1000,
        solarmotion=coords["solar_peculiar_kms"],
    )
    R, vR, vT, z, vz, phi = [np.atleast_1d(value) for value in [orbit.R(use_physical=False), orbit.vR(use_physical=False),
        orbit.vT(use_physical=False), orbit.z(use_physical=False), orbit.vz(use_physical=False), orbit.phi()]]
    delta = estimateDeltaStaeckel(pot, R, z, no_median=True)
    delta = np.where(np.isfinite(delta) & (delta > 1e-6), delta, config["actions"]["legacy_delta"])
    use_c = definition["kind"] == "MWPotential2014"
    finder = actionAngleStaeckel(pot=pot, delta=config["actions"]["legacy_delta"], c=use_c)
    jr, lz, jz = finder(R, vR, vT, z, vz, phi, delta=delta, c=use_c)
    scale = coords["r0_kpc"] * coords["v0_kms"]
    result = work[["ID", "draw", "logAge"]].copy()
    result["potential"] = potential_name
    result["R_kpc"] = R * coords["r0_kpc"]
    result["phi_rad"] = phi
    result["z_kpc"] = z * coords["r0_kpc"]
    result["vR_kms"] = vR * coords["v0_kms"]
    result["vT_kms"] = vT * coords["v0_kms"]
    result["vz_kms"] = vz * coords["v0_kms"]
    result["delta"] = delta
    result["JR_kpc_kms"] = np.asarray(jr) * scale
    result["Lz_kpc_kms"] = np.asarray(lz) * scale
    result["Jz_kpc_kms"] = np.asarray(jz) * scale
    result["action_valid"] = np.isfinite(result[["JR_kpc_kms", "Lz_kpc_kms", "Jz_kpc_kms"]]).all(axis=1) & (result["JR_kpc_kms"] >= 0) & (result["Jz_kpc_kms"] >= 0)
    return result
