"""Configuration and run metadata utilities."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess

from .data import file_record, read_records, write_json


def parse_config(parser: argparse.ArgumentParser, argv=None):
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file; command line options override it.",
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args(argv)
    if preliminary.config:
        values = json.loads(preliminary.config.read_text(encoding="utf-8"))
        actions = {action.dest: action for action in parser._actions}
        if not isinstance(values, dict) or set(values) - (
            actions.keys() - {"help", "config"}
        ):
            raise ValueError(
                "Configuration must be an object containing recognized argument names"
            )
        for name, value in values.items():
            action = actions[name]
            if value is None:
                raise ValueError(f"Omit unspecified configuration field {name!r}")
            if (
                isinstance(action, argparse.BooleanOptionalAction)
                or (action.nargs == 0 and isinstance(action.const, bool))
            ) and not isinstance(value, bool):
                raise ValueError(f"Configuration field {name!r} must be boolean")
            if action.type:
                if action.type is int and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    raise ValueError(f"Configuration field {name!r} must be an integer")
                if action.type is float and isinstance(value, bool):
                    raise ValueError(f"Configuration field {name!r} must be numeric")
                value = action.type(value)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Configuration field {name!r} must be finite")
            if action.choices and value not in action.choices:
                raise ValueError(f"Invalid configuration value for {name!r}")
            action.required = False
            parser.set_defaults(**{name: value})
    return parser.parse_args(argv)


def run_provenance(args, files: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = {
        name: value for name, value in vars(args).items() if not isinstance(value, Path)
    }
    artifacts = {}
    for name, path in files.items():
        if path is not None:
            artifacts[name] = file_record(path)
            if Path(path).suffix.lower() in (".jsonl", ".csv"):
                rows = read_records(path)
                artifacts[name]["record_count"] = len(rows)
                artifacts[name]["original_ids"] = (
                    [str(row["id"]) for row in rows]
                    if all("id" in row for row in rows)
                    else None
                )
    packages = {}
    for package in (
        "torch",
        "transformers",
        "Pillow",
        "nltk",
        "numpy",
        "scipy",
        "filelock",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    code = {
        str(path.relative_to(root)): file_record(path)["sha256"]
        for path in sorted(root.rglob("*.py"))
        if not {"__pycache__", ".venv", "venv", ".git"}.intersection(
            path.relative_to(root).parts
        )
    }
    return {
        "configuration": config,
        "inputs": artifacts,
        "package_versions": packages,
        "enclosing_git_commit": commit,
        "code_sha256": code,
        "scope": "new_local_run_not_original_paper_run",
    }


def save_history(
    output: Path, provenance: dict, history: list[dict], selected_epoch: int | None
) -> None:
    write_json(
        {
            "provenance": provenance,
            "selected_epoch": selected_epoch,
            "history": history,
        },
        Path(output).with_suffix(".run.json"),
    )


def validate_training_args(args) -> None:
    for name in (
        "dropout",
        "learning_rate",
        "weight_decay",
        "claim_weight",
        "evidence_weight",
    ):
        if hasattr(args, name) and not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    for name in (
        "epochs",
        "patience",
        "batch_size",
        "hidden_dim",
        "max_length",
        "heads",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.heads < 2 or args.heads % 2 or args.hidden_dim % args.heads:
        raise ValueError("heads must be even and divide hidden_dim")
    if args.encoder_layers < 0 or args.decoder_layers < 0:
        raise ValueError("Layer counts cannot be negative")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid dropout, learning rate, or weight decay")
    if not 1 <= args.max_length <= (77 if hasattr(args, "stance_checkpoint") else 512):
        raise ValueError("max_length exceeds the frozen target model limit")
    if hasattr(args, "max_evidence") and not 1 <= args.max_evidence <= 5:
        raise ValueError("max_evidence must lie between one and five")
    if hasattr(args, "claim_weight") and (
        args.claim_weight < 0
        or args.evidence_weight < 0
        or args.claim_weight + args.evidence_weight <= 0
    ):
        raise ValueError("Alignment weights must be nonnegative with a positive sum")
