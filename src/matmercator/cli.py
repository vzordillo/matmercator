"""Unified command-line interface for matmercator.

A single ``matmercator`` command with subcommands, all driven by one
``PipelineConfig`` (optionally seeded from a JSON config file and then overridden
by flags), so every entry point shares the same configuration path:

    matmercator run         # load -> featurize -> map (single process)
    matmercator features    # build the parallel feature cache
    matmercator map         # build the map from the cache
    matmercator landscapes  # node-based landscapes from the cache
    matmercator hero        # render the README hero banner

Config resolution order: ``PipelineConfig`` defaults < ``--config`` JSON file <
explicit CLI flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from matmercator import __version__
from matmercator.config import PipelineConfig

log = logging.getLogger("matmercator")


def _load_config(config_path: str | None, overrides: dict) -> PipelineConfig:
    """Build a resolved config from an optional JSON file plus flag overrides."""
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    data: dict = {}
    if config_path:
        loaded = json.loads(Path(config_path).read_text())
        data.update({k: v for k, v in loaded.items() if k in fields})
    data.update(
        {k: v for k, v in overrides.items() if v is not None and k in fields}
    )
    return PipelineConfig(**data).resolved()


def _setup_logging(output_dir: Path) -> None:
    """Send INFO logs to the console and to ``<output_dir>/run.log``."""
    logger = logging.getLogger("matmercator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(out / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Attach the shared config-override flags to a subcommand parser."""
    parser.add_argument(
        "--config", default=None, help="JSON config file (flags override it)"
    )
    parser.add_argument(
        "--data-root", default=None, help="folder containing mp_20/, ..."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["mp_20", "carbon_24", "perov_5"],
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="cap rows per split (for quick tests)",
    )
    parser.add_argument("--frame-set-size", type=int, default=None)
    parser.add_argument("--gtm-k", type=int, default=None)
    parser.add_argument("--gtm-niter", type=int, default=None)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--sort-eigenvalues", action="store_true")
    parser.add_argument("--no-diag-elems", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)


def _overrides(args: argparse.Namespace) -> dict:
    """Translate parsed flags into a ``PipelineConfig`` override dict."""
    return {
        "data_root": args.data_root,
        "dataset": args.dataset,
        "splits": tuple(args.splits) if args.splits else None,
        "max_structures": args.max_structures,
        "frame_set_size": args.frame_set_size,
        "gtm_k": args.gtm_k,
        "gtm_niter": args.gtm_niter,
        "n_permutations": args.n_permutations,
        "random_state": args.seed,
        "output_dir": args.output_dir,
        "standardize": False if args.no_standardize else None,
        "sort_eigenvalues": True if args.sort_eigenvalues else None,
        "diag_elems": False if args.no_diag_elems else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="matmercator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"matmercator {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, helptext in [
        ("run", "load -> featurize -> map (single process)"),
        ("features", "build the parallel feature cache"),
        ("map", "build the map from the cache"),
        ("landscapes", "render node-based landscapes from the cache"),
        ("hero", "render the README hero banner"),
        ("select", "Q2-driven GTM hyperparameter selection"),
        ("experiment", "descriptor comparison (SCM vs composition vs union)"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        _add_common(sp)
        if name == "features":
            sp.add_argument(
                "--split",
                default=None,
                help="featurize a single split (default: all configured)",
            )
            sp.add_argument("--procs", type=int, default=4)
        if name == "select":
            sp.add_argument("--folds", type=int, default=5)
    return parser


def _load_features(cfg: PipelineConfig):
    """Load ``(df, X)`` from the feature cache, else featurize in memory."""
    try:
        from matmercator.jobs import load_cache

        return load_cache(cfg)
    except (FileNotFoundError, ValueError, OSError):
        from matmercator.pipeline import build_features

        df, X, _ = build_features(cfg)
        return df, X


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    cfg = _load_config(args.config, _overrides(args))
    _setup_logging(cfg.output_dir)

    if args.command == "run":
        from matmercator.pipeline import _format_report
        from matmercator.pipeline import run

        log.info("\n%s", _format_report(run(cfg)))
    elif args.command == "features":
        from matmercator.featurize_cache import build_cache

        splits = (args.split,) if args.split else None
        build_cache(
            cfg,
            procs=args.procs,
            splits=splits,
            max_structures=cfg.max_structures,
        )
    elif args.command == "map":
        from matmercator.jobs import map_from_cache
        from matmercator.pipeline import _format_report

        log.info("\n%s", _format_report(map_from_cache(cfg)))
    elif args.command == "landscapes":
        from matmercator.jobs import landscapes_from_cache

        landscapes_from_cache(cfg)
    elif args.command == "hero":
        from matmercator.hero import make_hero

        make_hero(cfg)
    elif args.command == "select":
        from matmercator.selection import select_gtm

        df, X = _load_features(cfg)
        report = select_gtm(cfg, df, X, n_folds=args.folds)
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "selection_report.json").write_text(json.dumps(report, indent=2))
        b = report["best"]
        log.info(
            "best: k=%d m=%d s=%.2f regul=%.3f -> mean Q2=%+.3f",
            b["k"],
            b["m"],
            b["s"],
            b["regul"],
            b["mean_q2"],
        )
    elif args.command == "experiment":
        from matmercator.composition import cache_composition
        from matmercator.experiment import _format_md
        from matmercator.experiment import run_experiment
        from matmercator.featurize_cache import cache_dir_for

        cache = cache_dir_for(cfg)
        missing = [s for s in cfg.splits if not (cache / f"X_{s}.npz").exists()]
        if missing:
            log.error(
                "missing SCM feature cache for %s; run `matmercator features` "
                "first",
                missing,
            )
            return 1
        if any(not (cache / f"Xcomp_{s}.npz").exists() for s in cfg.splits):
            log.info("composition cache missing -> building it (serial)")
            cache_composition(cfg)
        log.info("\n%s", _format_md(run_experiment(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
