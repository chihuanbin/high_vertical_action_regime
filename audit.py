from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .catalogue import CROSSMATCH_COLUMNS, column_dictionary, duplicate_flags, load_clusters, read_fixed_width, sequential_cutflow
from .config import resolve_path, sha256


def _crossmatches(config: dict) -> pd.DataFrame | None:
    path = resolve_path(config, "crossmatches")
    if not path.exists():
        return None
    frame = read_fixed_width(path, CROSSMATCH_COLUMNS)
    if len(frame) != 29956:
        raise RuntimeError(f"Crossmatch table has {len(frame)} records; expected 29956")
    return frame


def build_master_sample(config: dict) -> dict:
    results = resolve_path(config, "results")
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    clusters_path = resolve_path(config, "clusters")
    all_clusters = load_clusters(clusters_path)
    cutflow, mask = sequential_cutflow(all_clusters, config)
    sample = all_clusters.loc[mask].copy().reset_index(drop=True)
    crossmatches = _crossmatches(config)
    duplicate_table = duplicate_flags(all_clusters, crossmatches)
    sample = sample.merge(duplicate_table.drop(columns=["Name", "AllNames", "NXmatches", "XmatchType", "isMerged"]), on="ID", how="left")
    sample.to_csv(tables / "master_sample.csv", index=False)
    cutflow.to_csv(tables / "sample_cutflow.csv", index=False)
    column_dictionary().to_csv(tables / "input_column_dictionary.csv", index=False)
    duplicate_table.to_csv(tables / "duplicate_crossmatch_flags.csv", index=False)
    observed = len(sample)
    expected = int(config["project"]["expected_sample_size"])
    if observed != expected:
        raise RuntimeError(f"Published sample mismatch: expected {expected}, reproduced {observed}. See sample_cutflow.csv")
    return {
        "input": str(clusters_path),
        "input_sha256": sha256(clusters_path),
        "crossmatch_available": crossmatches is not None,
        "raw_count": len(all_clusters),
        "sample_count": observed,
        "possible_duplicate_count": int(duplicate_table["possible_duplicate"].sum()),
    }


def write_audit(config: dict, summary: dict) -> Path:
    path = resolve_path(config, "results") / "audit_report.md"
    crossmatch_note = (
        "The verified CDS crossmatch table was ingested."
        if summary["crossmatch_available"] else
        "`crossma.dat` was unavailable or failed validation; cross-identification is limited to `AllNames`, `NXmatches`, and `XmatchType`."
    )
    text = f"""# Audit of the pre-A&A analysis

## Scope and reproducibility

The raw HUNT24 catalogue contains {summary['raw_count']} rows. The legacy published cuts reproduce {summary['sample_count']} clusters. Input SHA-256: `{summary['input_sha256']}`. {crossmatch_note}

## Input measurements

The cluster table uses `dist50` as distance with `dist16/dist84` intervals; `logAge50` with `logAge16/logAge84`; `RV` with cluster standard error `e_RV`; proper motions `pmRA/pmDE` with standard errors `e_pmRA/e_pmDE`; parallax `Plx` with `e_Plx`; `N`; `n_RV`; `probJ`; `MassJ/e_MassJ` with `MassTot/e_MassTot` fallback; and `AV16/50/84`. Cluster-level astrometric covariance is not published. Member rows contain star-level parallax/proper-motion correlations.

## Legacy coordinate and action conventions

- `R0=8.178 kpc`, `V0=236.9 km/s`, galpy `solarmotion='schoenrich'`.
- Galactocentric positions were also transformed with Astropy using only `galcen_distance`, leaving other frame defaults implicit.
- Actions used galpy `MWPotential2014` and `actionAngleStaeckel(delta=0.45, c=True)`.
- Stored actions are physical `kpc km s^-1`; `Lz` retains galpy's sign and ratios use `abs(Lz)`.
- Legacy `zmax` was analytic (`Orbit.zmax(..., analytic=True)`), not time-integrated.
- The descriptive component score was the sum of min-max normalized component medians in `eta_z`, `zmax/Rg`, `Delta_logJz`, and the vertical-outlier fraction. These are correlated derivatives of the same phase-space measurements.

## Legacy sample and controls

The 1,079-cluster sample applies `Type=o`, `N>=30`, positive parallax, fractional parallax error at most 0.1, age-width at most 0.4 dex, valid positive mass, `probJ>=0.5`, at least three RV members, and finite RV, age, and distance. The v2 control-like set was defined by proximity to an age-radius `log10(Jz)` baseline, not an external control population. Later matching reused eligible controls and summarized one row per high-action cluster.

## Bootstrap and mixture risks

- The legacy GMM scaler was fit once to the complete sample and reused in every bootstrap.
- Resampling was with replacement, but each fitted model was predicted on the unique full catalogue. Duplicate bootstrap occurrences therefore affected fitting without remaining explicit rows in membership comparison.
- Hungarian component matching used overlap on full-catalogue predictions, so it did not directly mishandle duplicate row identities, but it did not quantify their multiplicities either.
- The continuous multimodality null was a fitted multivariate Gaussian preserving the transformed sample mean and covariance, not a physical disk-selection model.
- BIC preference for several overlapping Gaussian terms was sometimes described alongside action, amplitude, and frequency diagnostics as though those were independent validations. They all derive from the same six-dimensional phase space and adopted potential.

## Claims lacking a single stored source of truth

The old manuscript manually copied results from several CSV/JSON files. It had no validator tying every numerical statement to a machine-readable source. Sample counts, GMM stability rates, component centers, matched differences, positional fractions, and p-values must all be regenerated and exported as manuscript macros before reuse.

## Current pipeline flow

```text
HUNT24 clusters.dat
  -> fixed-width parsing
  -> quality/RV cuts (1,079 clusters)
  -> nominal 6D transformation
  -> MWPotential2014 + fixed-delta Staeckel actions
  -> age-radius log(Jz) residuals
  -> standardized action-space GMM
  -> bootstrap component matching
  -> matched controls / orbit-family / origin proxies
  -> manually assembled ApJ manuscript
```

## Principal risks addressed by the revision

1. No propagation of catalogue measurement errors into actions.
2. Dependence on one potential and fixed Staeckel focal parameter.
3. Analytic rather than integrated vertical amplitudes.
4. GMM components treated too prominently despite overlapping-density and unstable-K results.
5. Incomplete selection-function treatment and control reuse.
6. Correlated derived quantities presented as separate support.
7. No table-to-manuscript numerical consistency check.
"""
    path.write_text(text, encoding="utf-8")
    (path.parent / "logs" / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
