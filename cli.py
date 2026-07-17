from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import build_master_sample, write_audit
from .action_pipeline import merge_action_checkpoints, run_actions
from .models import run_models
from .orbits import merge_orbit_checkpoints, run_orbits
from .secondary import run_secondary
from .flare import run_flare
from .gmm_bootstrap import run_gmm_bootstrap
from .matching import run_matching
from .member_covariance import run_member_covariance
from .publication import build_manifest, consolidate_tables, make_figures
from .simulations import run_null_simulations
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A&A open-cluster vertical-action pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Audit legacy work and reproduce the master sample")
    actions = sub.add_parser("actions", help="Propagate phase-space uncertainties into actions")
    actions.add_argument("--draws", type=int, default=None)
    actions.add_argument("--max-clusters", type=int, default=None)
    actions.add_argument("--potential", default=None)
    merge = sub.add_parser("merge-actions", help="Validate and merge complete per-potential action checkpoints")
    merge.add_argument("--potentials", nargs="+", default=None)
    orbits = sub.add_parser("orbits", help="Integrate nominal orbits")
    orbits.add_argument("--max-clusters", type=int, default=None)
    orbits.add_argument("--potential", default=None)
    merge_orbits = sub.add_parser("merge-orbits", help="Validate and merge complete per-potential orbit checkpoints")
    merge_orbits.add_argument("--potentials", nargs="+", required=True)
    sub.add_parser("models", help="Fit continuous and quantile models")
    sub.add_parser("secondary", help="Run selection diagnostics and exploratory GMM")
    sub.add_parser("member-covariance", help="Estimate member-bootstrap centroid covariance")
    sub.add_parser("flare", help="Fit sample-conditional phenomenological flare models")
    matching = sub.add_parser("matching", help="Run secondary matched-control analyses")
    gmm_bootstrap = sub.add_parser("gmm-bootstrap", help="Bootstrap the exploratory GMM")
    gmm_bootstrap.add_argument("--count", type=int, default=None)
    nulls = sub.add_parser("nulls", help="Run continuous null simulations")
    nulls.add_argument("--count", type=int, default=None)
    sub.add_parser("publication", help="Build figures, consolidated tables, and manifest")
    sub.add_parser("all", help="Run all implemented stages in dependency order")
    return parser.parse_args()


def run_audit(config: dict) -> dict:
    summary = build_master_sample(config)
    write_audit(config, summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command in {"audit", "all"}:
        print(json.dumps(run_audit(config), indent=2))
    if args.command in {"actions", "all"}:
        print(json.dumps(run_actions(config, getattr(args, "draws", None), getattr(args, "max_clusters", None), getattr(args, "potential", None)), indent=2))
    if args.command == "merge-actions":
        print(json.dumps(merge_action_checkpoints(config, args.potentials), indent=2))
    if args.command in {"orbits", "all"}:
        print(json.dumps(run_orbits(config, getattr(args, "max_clusters", None), getattr(args, "potential", None)), indent=2))
    if args.command == "merge-orbits":
        print(json.dumps(merge_orbit_checkpoints(config, args.potentials), indent=2))
    if args.command in {"models", "all"}:
        print(json.dumps(run_models(config), indent=2))
    if args.command in {"secondary", "all"}:
        print(json.dumps(run_secondary(config), indent=2))
    if args.command in {"member-covariance", "all"}:
        print(json.dumps(run_member_covariance(config), indent=2))
    if args.command in {"flare", "all"}:
        print(json.dumps(run_flare(config), indent=2))
    if args.command in {"matching", "all"}:
        print(json.dumps(run_matching(config), indent=2))
    if args.command in {"gmm-bootstrap", "all"}:
        print(json.dumps(run_gmm_bootstrap(config, getattr(args, "count", None)), indent=2))
    if args.command in {"nulls", "all"}:
        print(json.dumps(run_null_simulations(config, getattr(args, "count", None)), indent=2))
    if args.command in {"publication", "all"}:
        consolidate_tables(config)
        figures = make_figures(config)
        manifest = build_manifest(config)
        print(json.dumps({"figures": len(figures), "manifest_outputs": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
