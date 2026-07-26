"""Command line entry point.

Three commands, in the order they are meant to be run:

    uv run sq4 train      # fit and check the backbone classifier
    uv run sq4 explain    # run every method over the sampled nodes
    uv run sq4 report     # summarise the results per method
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from sq4_explain.config import Settings, load_settings
from sq4_explain.data import EllipticDataModule
from sq4_explain.experiment import ExplanationExperiment, sample_target_nodes, save_results
from sq4_explain.explainers import available_explainers
from sq4_explain.runtime import select_device, setup_logging, write_manifest
from sq4_explain.train import BackboneTrainer, load_checkpoint, save_checkpoint

MANIFEST_TRAIN = "manifest_train.json"
MANIFEST_EXPLAIN = "manifest_explain.json"


def _prepare(args: argparse.Namespace) -> Settings:
    """Load settings and configure logging before any work starts."""
    settings = load_settings(Path(args.config) if args.config else None)
    setup_logging(settings.logging, settings.paths.log_file)
    return settings


def command_train(args: argparse.Namespace) -> None:
    """Fit the backbone on the early time steps and evaluate on the later ones."""
    settings = _prepare(args)
    device = select_device(args.device)
    logger.info("Training on {}", device)

    bundle = EllipticDataModule(settings.data, settings.paths.data_root).load()
    trainer = BackboneTrainer(settings, device)
    model, report = trainer.fit(bundle)
    save_checkpoint(model, settings.paths.artifact_root)

    per_step = trainer.evaluate_per_timestep(model, bundle)
    frame = pd.DataFrame(
        [{"timestep": step, **scores.as_dict()} for step, scores in per_step.items()]
    )
    destination = settings.paths.artifact_root / "classifier_per_timestep.csv"
    frame.to_csv(destination, index=False)
    logger.info("Per-timestep report written to {}", destination)

    write_manifest(
        settings.paths.artifact_root / MANIFEST_TRAIN,
        settings,
        extra={"test_report": report.as_dict(), "device": str(device)},
    )


def command_explain(args: argparse.Namespace) -> None:
    """Run every configured explanation method over the sampled nodes."""
    settings = _prepare(args)
    device = select_device(args.device)

    bundle = EllipticDataModule(settings.data, settings.paths.data_root).load()
    bundle.data.to(device)
    model = load_checkpoint(bundle, settings, device)

    nodes = sample_target_nodes(model, bundle, settings)
    logger.info("Explaining {} predicted-illicit test nodes", nodes.numel())

    experiment = ExplanationExperiment(model, bundle, settings, device)
    explanations, consistency_frame, attributions = experiment.run(nodes)
    save_results(
        explanations, consistency_frame, attributions, settings.paths.artifact_root
    )
    write_manifest(
        settings.paths.artifact_root / MANIFEST_EXPLAIN,
        settings,
        extra={
            "nodes": nodes.tolist(),
            "device": str(device),
            "registered_methods": available_explainers(),
        },
    )


def command_report(args: argparse.Namespace) -> None:
    """Summarise runtime, fidelity and consistency per method."""
    settings = _prepare(args)
    root = settings.paths.artifact_root
    explanations = pd.read_csv(root / "explanations.csv")
    consistency_frame = pd.read_csv(root / "consistency.csv")

    per_method = explanations.groupby("method").agg(
        runtime_s_mean=("runtime_s", "mean"),
        runtime_s_max=("runtime_s", "max"),
        fidelity_plus_mean=("fidelity_plus", "mean"),
        fidelity_minus_mean=("fidelity_minus", "mean"),
        explanations=("node_index", "count"),
    )
    stability = consistency_frame.groupby("method").agg(
        jaccard_mean=("mean_jaccard", "mean"),
        jaccard_worst=("min_jaccard", "min"),
        spearman_mean=("mean_spearman", "mean"),
    )
    summary = per_method.join(stability)
    destination = root / "summary.csv"
    summary.to_csv(destination)
    logger.info("Summary written to {}", destination)
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser and its subcommands."""
    parser = argparse.ArgumentParser(prog="sq4", description=__doc__)
    parser.add_argument("--config", default=None, help="path to a TOML config file")
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | cuda | mps"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("train", command_train),
        ("explain", command_explain),
        ("report", command_report),
    ):
        subparser = subparsers.add_parser(name, help=handler.__doc__)
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
