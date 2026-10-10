#!/usr/bin/env python3
"""Recompute ViVQA supplementary diagnostics from retained per item predictions.

Run from the research workspace with
python github_release/embedding_translation_adapter/recompute_supplementary.py \
    --workspace-root . --output-dir revision/experiments/vivqa_supplementary_recomputed

This command rescores explicitly supplied predictions and verifies their hash
against the evaluation record. It never selects a checkpoint using test scores.
The default inputs are retained predictions for the checkpoint identified by
the authors for the current controlled protocol. Author confirmation of run
membership is recorded separately from artifact verification through
--checkpoint-provenance. To analyse a fresh evaluation, pass its predictions.csv and
result.json through --adapter-predictions and --adapter-record. The prediction
CSV must retain img_id, type, question, answer_english, prediction and exact.
The record must include predictions_sha256, checkpoint_sha256, test.samples and
test.exact_accuracy. All input paths can be overridden. Image inference and
tokenisation are not repeated by this rescoring command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
from embedding_transfer.metrics import NORMALIZATION, normalize_answer, token_f1
from scipy.stats import spearmanr

TYPES = {0: "Object", 1: "Count", 2: "Colour", 3: "Location"}
EXCLUDED_REPORTED_ACCURACIES = (0.4365,)


def artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align_rows(references: list[dict], rows: list[dict]) -> list[dict]:
    """Retain duplicate questions and verify their original test row identity."""
    if len(rows) != len(references):
        raise ValueError("Prediction and reference row counts differ")
    if not rows:
        raise ValueError("Cannot analyse an empty test split")
    id_column = next(
        (name for name in ("sample_id", "row_index") if name in rows[0]), None
    )
    if id_column:
        indexed = {int(row[id_column]): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(range(len(rows))):
            raise ValueError(
                "Prediction row IDs must be unique and cover the test split"
            )
        rows = [indexed[index] for index in range(len(rows))]
    for index, (reference, row) in enumerate(zip(references, rows)):
        for name in ("img_id", "type", "question", "answer_english"):
            actual_name = (
                "question_vietnamese"
                if name == "question" and name not in row
                else name
            )
            if row.get(actual_name) != reference[name]:
                raise ValueError(f"Test identity mismatch at row {index} in {name}")
    return rows


def excluded_accuracy(value: float) -> bool:
    return any(
        abs(value - rejected) < 0.00005 for rejected in EXCLUDED_REPORTED_ACCURACIES
    )


def verify_record(record: dict, predictions: Path, checkpoint: Path | None = None) -> None:
    """Require exact artifact identity rather than a similar aggregate score."""
    if excluded_accuracy(float(record["test"]["exact_accuracy"])):
        raise ValueError("The rejected 0.4365 run must not enter supplementary diagnostics")
    if artifact(predictions)["sha256"] != record["predictions_sha256"]:
        raise ValueError("Prediction hash differs from the retained evaluation record")
    if checkpoint is not None and artifact(checkpoint)["sha256"] != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the retained evaluation record")


def verify_scores(record: dict, exact: np.ndarray) -> None:
    if len(exact) != record["test"]["samples"] or abs(
        float(exact.mean()) - record["test"]["exact_accuracy"]
    ) > 1e-12:
        raise ValueError("Recomputed exact match or sample count differs from the evaluation record")
    if excluded_accuracy(float(exact.mean())):
        raise ValueError("The rejected 0.4365 run must not enter supplementary diagnostics")


def checkpoint_provenance(path: Path | None, checkpoint_sha256: str) -> dict:
    """Keep an author's run identification separate from measured artifact identity."""
    if path is None:
        return {"basis": "not_supplied", "training_records_independently_audited": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Controlled run provenance refers to a different checkpoint")
    membership = value["controlled_run_membership"]
    if membership.get("basis") != "author_confirmation" or not membership.get("statement"):
        raise ValueError("Run membership needs an explicit author confirmation")
    return {**membership, "record": artifact(path)}


def score_rows(
    references: list[dict],
    rows: list[dict],
    prediction_column: str,
    normalizer=normalize_answer,
) -> dict:
    normalizer = lru_cache(maxsize=None)(normalizer)
    pairs = [
        (normalizer(row[prediction_column]), normalizer(ref["answer_english"]))
        for ref, row in zip(references, rows)
    ]
    return {
        "exact": np.asarray([float(pred == ref) for pred, ref in pairs]),
        "token_f1": np.asarray([token_f1(pred, ref) for pred, ref in pairs]),
    }


def bootstrap(values: np.ndarray, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or samples < 1
    ):
        raise ValueError(
            "Bootstrap needs finite per item values and a positive resample count"
        )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n": len(values),
    }


def summarise(references: list[dict], scores: dict, samples: int, seed: int) -> dict:
    kinds = np.asarray([int(row["type"]) for row in references])
    groups = {
        "Overall": np.ones(len(references), dtype=bool),
        **{name: kinds == index for index, name in TYPES.items()},
    }
    result = {}
    for name, selected in groups.items():
        result[name] = {
            "n": int(selected.sum()),
            "methods": {},
            "adapter_minus_gemini": {},
        }
        for method, metrics in scores.items():
            result[name]["methods"][method] = {
                metric: bootstrap(values[selected], samples, seed)
                for metric, values in metrics.items()
            }
        for metric in ("exact", "token_f1"):
            delta = (
                scores["adapter"][metric][selected] - scores["gemini"][metric][selected]
            )
            result[name]["adapter_minus_gemini"][metric] = bootstrap(
                delta, samples, seed
            )
    return result


def tokenisation_summary(
    references: list[dict],
    cache_path: Path,
    correctness: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    rows = read_csv(cache_path)
    # This cache has no answers. Add answers only after checking every row ID,
    # image, question, type and English question against the reference split.
    if len(rows) != len(references):
        raise ValueError("Tokenisation cache row count differs from test split")
    for index, (row, ref) in enumerate(zip(rows, references)):
        if int(row["sample_id"]) != index or any(
            row[key] != ref[key]
            for key in ("img_id", "type", "question", "question_english")
        ):
            raise ValueError("Tokenisation cache identity mismatch")
    vi = np.asarray([int(row["vi_content_tokens"]) for row in rows])
    en = np.asarray([int(row["en_content_tokens"]) for row in rows])
    absolute = np.abs(vi - en)
    truncated = np.asarray([int(row["either_truncated_at_max_length"]) for row in rows])
    if (vi < 0).any() or (en < 0).any() or not np.isin(truncated, [0, 1]).all():
        raise ValueError("Invalid cached token counts or truncation flags")
    correlation = spearmanr(absolute, correctness)
    return {
        "tokenisation_rerun": False,
        "source": "Retained mBERT and BLIP token counts, verified against all question identities",
        "mean_vi_tokens": float(vi.mean()),
        "mean_en_tokens": float(en.mean()),
        "mean_absolute_difference": float(absolute.mean()),
        "equal_share": float((absolute == 0).mean()),
        "truncated_share": float(truncated.mean()),
        "spearman_correctness": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
        "bins": {
            name: bootstrap(correctness[mask], samples, seed)
            for name, mask in {
                "equal": absolute == 0,
                "one_to_two": (absolute >= 1) & (absolute <= 2),
                "three_to_five": (absolute >= 3) & (absolute <= 5),
                "over_five": absolute > 5,
            }.items()
        },
    }


def baseline_controls(references: list[dict], directory: Path) -> dict:
    """Rescore the fixed baseline runs without importing historical adapters."""
    locations = {
        "input_linear": "vivqa_adapter_baselines_controlled_30epoch/linear_seed",
        "input_mlp": "vivqa_adapter_baselines_controlled_30epoch/mlp_seed",
        "contextual_linear": "vivqa_contextual_source_controlled_30epoch/contextual_linear_seed",
        "contextual_mlp": "vivqa_contextual_source_controlled_30epoch/contextual_mlp_seed",
        "internal_bottleneck": "vivqa_peft_bottleneck_controlled_30epoch/bottleneck_seed",
    }
    result = {}
    selection_hashes = set()
    for method, relative in locations.items():
        runs = []
        for seed in (42, 43, 44):
            path = directory / f"{relative}{seed}"
            record = json.loads((path / "result.json").read_text())
            predictions = path / "test_predictions.csv"
            if artifact(predictions)["sha256"] != record["predictions_sha256"]:
                raise ValueError(
                    f"Baseline prediction hash mismatch for {method} seed {seed}"
                )
            selection_hashes.add(record["validation_selection_sha256"])
            rows = align_rows(references, read_csv(predictions))
            score = score_rows(references, rows, "prediction")
            if not np.array_equal(
                score["exact"], np.asarray([int(row["exact"]) for row in rows])
            ):
                raise ValueError(
                    f"Baseline normalization mismatch for {method} seed {seed}"
                )
            value = float(score["exact"].mean())
            if abs(value - record["test"]["exact_accuracy"]) > 1e-12:
                raise ValueError(
                    f"Baseline aggregate mismatch for {method} seed {seed}"
                )
            runs.append(
                {
                    "seed": seed,
                    "exact": value,
                    "token_f1": float(score["token_f1"].mean()),
                    "selected_epoch": record["best_epoch"],
                    "validation_exact": record["best_validation_exact"],
                    "predictions": artifact(predictions),
                    "record": artifact(path / "result.json"),
                }
            )
        result[method] = {
            "runs": runs,
            "exact_mean": statistics.mean(run["exact"] for run in runs),
            "exact_sample_sd": statistics.stdev(run["exact"] for run in runs),
        }
    if len(selection_hashes) != 1:
        raise ValueError("Baseline validation selection identities differ")
    result["validation_selection_sha256"] = selection_hashes.pop()
    return result


def write_tables(summary: dict, destination: Path) -> None:
    lines = []
    for metric, title, label in (
        ("exact", "Exact match accuracy", "em"),
        ("token_f1", "Token F1", "f1"),
    ):
        lines.extend(
            [
                r"\begin{table}[!htbp]",
                r"\centering",
                rf"\caption{{{title} by ViVQA question type. Differences are adapter minus Gemini in percentage points with paired $95\%$ bootstrap intervals.}}",
                rf"\label{{tab:supp-qtype-{label}}}",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Type & Count & Gemini $\uparrow$ & Qwen $\uparrow$ & Adapter $\uparrow$ & Difference with interval \\",
                r"\midrule",
            ]
        )
        for name, group in summary["groups"].items():
            values = [
                group["methods"][method][metric]["mean"]
                for method in ("gemini", "qwen", "adapter")
            ]
            delta = group["adapter_minus_gemini"][metric]
            lower, upper = [value * 100 for value in delta["ci95"]]
            count = f"{group['n']:,}".replace(",", "{,}")
            lines.append(
                f"{name} & {count} & "
                + " & ".join(f"{value:.4f}" for value in values)
                + rf" & ${delta['mean'] * 100:+.2f}\ [{lower:.2f},{upper:.2f}]$ \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_diagnostics(summary: dict, destination: Path) -> None:
    """Generate the paper fragment from the same metrics as the JSON export."""
    overall = summary["groups"]["Overall"]
    adapter = overall["methods"]["adapter"]
    gemini = overall["methods"]["gemini"]
    delta = overall["adapter_minus_gemini"]["token_f1"]
    drift = summary["tokenisation"]
    equal, over_five = drift["bins"]["equal"], drift["bins"]["over_five"]
    table_path = destination.with_name("question_type_tables.tex")
    write_tables(summary, table_path)
    text = (
        "% Generated by recompute_supplementary.py from verified per item answers.\n"
        "\\subsection{ViVQA results by question type}\n\n"
        "The diagnostic checkpoint gives exact match accuracy "
        rf"${adapter['exact']['mean']:.4f}$ and Token F1 ${adapter['token_f1']['mean']:.4f}$. "
        rf"The corresponding Gemini values are ${gemini['exact']['mean']:.4f}$ and ${gemini['token_f1']['mean']:.4f}$. "
        "Tables~\\ref{tab:supp-qtype-em} and~\\ref{tab:supp-qtype-f1} report these metrics "
        "by question type and include the Qwen translation pipeline. Differences are "
        "paired adapter minus Gemini values in percentage points. Each interval uses "
        "$" + f"{summary['bootstrap']['samples']:,}".replace(",", "{,}") + rf"$ paired bootstrap samples with seed ${summary['bootstrap']['seed']}$. "
        rf"Across all test items, the paired Token F1 difference is ${delta['mean'] * 100:.2f}$ points "
        rf"with a $95\%$ interval of $[{delta['ci95'][0] * 100:.2f},{delta['ci95'][1] * 100:.2f}]$."
        "\n\n"
        + table_path.read_text(encoding="utf-8")
        + "\nThe intervals describe uncertainty from resampling test examples for this "
        "checkpoint and do not include variation across training seeds. They are not "
        "adjusted for comparisons across question types.\n\n"
        "\\subsection{Tokenisation drift}\n\n"
        "We count tokens before truncation and exclude tokenizer special IDs. "
        rf"Vietnamese and English questions average ${drift['mean_vi_tokens']:.2f}$ and ${drift['mean_en_tokens']:.2f}$ content tokens. "
        rf"Their mean absolute count difference is ${drift['mean_absolute_difference']:.2f}$. "
        rf"Token counts are equal for ${drift['equal_share'] * 100:.2f}\%$ of questions, "
        rf"and at least one sequence is truncated in ${drift['truncated_share'] * 100:.2f}\%$ of cases at the maximum length of $32$. "
        rf"Adapter exact match is ${equal['mean']:.4f}$ when the token counts are equal "
        rf"and ${over_five['mean']:.4f}$ when the absolute difference exceeds five. "
        rf"Their bootstrap intervals are $[{equal['ci95'][0]:.4f},{equal['ci95'][1]:.4f}]$ "
        rf"and $[{over_five['ci95'][0]:.4f},{over_five['ci95'][1]:.4f}]$. "
        rf"The Spearman correlation between absolute count difference and correctness is ${drift['spearman_correctness']:.4f}$. "
        "Token counts describe segmentation differences. Section~\\ref{subsec:supp-token-alignment} "
        "examines correspondence between the resulting representations directly. "
        "Correctness and its intervals use the same predictions as the question type analysis.\n"
    )
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--saved-summary", type=Path,
        help="Render a saved summary without rerunning inference, metrics or bootstrap",
    )
    parser.add_argument(
        "--paper-output-dir", type=Path,
        help="Also write the generated ViVQA diagnostics fragment in this paper directory",
    )
    for name in (
        "data",
        "gemini-predictions",
        "qwen-predictions",
        "adapter-predictions",
        "adapter-record",
        "token-counts",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Optionally rescore the five repeated baseline methods",
    )
    parser.add_argument(
        "--adapter-checkpoint", type=Path,
        help="Optional original checkpoint, verified by SHA256 against the record",
    )
    parser.add_argument(
        "--checkpoint-provenance", type=Path,
        help="Author confirmation of controlled run membership tied to checkpoint SHA256",
    )
    parser.add_argument(
        "--token-count-record", type=Path,
        help="Fresh tokenisation record with token_counts_sha256 and data_sha256",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.saved_summary:
        summary = json.loads(args.saved_summary.read_text(encoding="utf-8"))
        if excluded_accuracy(summary["groups"]["Overall"]["methods"]["adapter"]["exact"]["mean"]):
            raise ValueError("The rejected 0.4365 run must not enter supplementary diagnostics")
        for saved in summary["inputs"].values():
            if artifact(Path(saved["path"]))["sha256"] != saved["sha256"]:
                raise ValueError(f"Saved summary input hash changed for {saved['path']}")
        provenance = {
            "saved_summary": artifact(args.saved_summary),
            "renderer": artifact(Path(__file__)),
            "numeric_metrics_recomputed": False,
            "inference_rerun": False,
            "training_rerun": False,
            "input_hashes_verified": True,
            "controlled_run_membership": summary["prediction_provenance"].get(
                "controlled_run_membership", {"basis": "not_supplied"}
            ),
            "note": "Render saved diagnostic metrics only. Run membership follows the separately recorded author confirmation; numerical checks verify artifact identity and scores.",
        }
        for destination in [args.output_dir] + ([args.paper_output_dir] if args.paper_output_dir else []):
            destination.mkdir(parents=True, exist_ok=True)
            write_diagnostics(summary, destination / "vivqa_diagnostics.tex")
            (destination / "render_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance, indent=2))
        return
    root = args.workspace_root.resolve()
    defaults = {
        "data": "data_source/vivqa/translated_test_gpt.csv",
        "gemini_predictions": "revision/experiments/vivqa_token_f1_predictions.csv",
        "qwen_predictions": "src/ckpts/r2_1_qwen3_4b_blip_standard_predictions.csv",
        "adapter_predictions": "revision/experiments/vivqa_adapter_baselines_matched_split/proposed_seed42/test_predictions.csv",
        "adapter_record": "revision/experiments/vivqa_adapter_baselines_matched_split/proposed_seed42/result.json",
        "token_counts": "revision/experiments/vivqa_tokenization_drift_per_sample.csv",
    }
    paths = {
        name: getattr(args, name) or root / relative
        for name, relative in defaults.items()
    }
    references = read_csv(paths["data"])
    if len(references) != 3001 or {int(row["type"]) for row in references} != set(TYPES):
        raise ValueError(
            "Expected all 3001 test items with the four documented question types"
        )
    record = json.loads(paths["adapter_record"].read_text())
    verify_record(record, paths["adapter_predictions"], args.adapter_checkpoint)
    rows = {
        name: align_rows(references, read_csv(paths[name + "_predictions"]))
        for name in ("gemini", "qwen", "adapter")
    }
    scores = {
        name: score_rows(
            references,
            values,
            "translation_prediction" if name == "gemini" else "prediction",
        )
        for name, values in rows.items()
    }
    for name in ("adapter", "qwen"):
        if not np.array_equal(
            scores[name]["exact"], np.asarray([int(row["exact"]) for row in rows[name]])
        ):
            raise ValueError(
                f"Recomputed normalization differs from retained exact match values for {name}"
            )
    verify_scores(record, scores["adapter"]["exact"])
    # The second retained export provides an additional answer level association.
    alternate_differences = [
        index
        for index, (left, right) in enumerate(zip(rows["adapter"], rows["gemini"]))
        if left["prediction"] != right.get("synaug_prediction")
    ]
    summary = {
        "normalization": NORMALIZATION,
        "unit": "fraction",
        "test_samples": len(references),
        "analysis_code": artifact(Path(__file__)),
        "package_versions": {
            name: version(name) for name in ("numpy", "scipy", "nltk")
        },
        "inference_rerun": record.get("inference_rerun", False),
        "inference_executed_by_this_command": False,
        "numeric_metrics_recomputed": True,
        "training_rerun": False,
        "analysis": "Rescoring answers from the specified checkpoint; the checkpoint result is distinct from the reported mean across training seeds",
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "paired test items",
            "training_seed_uncertainty_included": False,
        },
        "prediction_provenance": {
            "rule": "Explicit prediction file and its hash-verified evaluation record; no selection by test score",
            "excluded_reported_accuracies": list(EXCLUDED_REPORTED_ACCURACIES),
            "recorded_seed": record.get("seed"),
            "recorded_selection_metric": record.get("selection_metric"),
            "original_checkpoint_sha256": record["checkpoint_sha256"],
            "original_checkpoint_hash_verified": args.adapter_checkpoint is not None,
            "original_checkpoint": artifact(args.adapter_checkpoint) if args.adapter_checkpoint else None,
            "controlled_run_membership": checkpoint_provenance(
                args.checkpoint_provenance, record["checkpoint_sha256"]
            ),
            "original_checkpoint_prediction_file_hash_verified": True,
            "second_saved_prediction_export_matches": not alternate_differences,
            "second_saved_prediction_export_differing_rows": alternate_differences,
            "adapter_prediction_source": "Only the prediction file whose hash matches adapter_record is used for every adapter diagnostic",
        },
        "inputs": {
            name: artifact(path)
            for name, path in paths.items()
        },
        "groups": summarise(
            references, scores, args.bootstrap_samples, args.bootstrap_seed
        ),
        "tokenisation": tokenisation_summary(
            references,
            paths["token_counts"],
            scores["adapter"]["exact"],
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
    }
    if args.baseline_root:
        summary["baseline_controls"] = baseline_controls(references, args.baseline_root)
    if args.token_count_record:
        count_record = json.loads(args.token_count_record.read_text(encoding="utf-8"))
        if (
            count_record["token_counts_sha256"] != artifact(paths["token_counts"])["sha256"]
            or count_record["data_sha256"] != artifact(paths["data"])["sha256"]
        ):
            raise ValueError("Token count record does not identify the supplied counts and test split")
        summary["tokenisation"]["tokenisation_rerun"] = bool(count_record["tokenisation_rerun"])
        summary["tokenisation"]["source"] = "Fresh mBERT and BLIP token counts verified against test questions and the tokenisation record"
        summary["inputs"]["token_count_record"] = artifact(args.token_count_record)
    if args.checkpoint_provenance:
        summary["inputs"]["checkpoint_provenance"] = artifact(args.checkpoint_provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_item_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        names = ["sample_id", "img_id", "type"] + [
            f"{method}_{metric}" for method in scores for metric in scores[method]
        ]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for index, row in enumerate(references):
            writer.writerow(
                {
                    "sample_id": index,
                    "img_id": row["img_id"],
                    "type": row["type"],
                    **{
                        f"{method}_{metric}": float(values[index])
                        for method, metrics in scores.items()
                        for metric, values in metrics.items()
                    },
                }
            )
    write_diagnostics(summary, args.output_dir / "vivqa_diagnostics.tex")
    render_record = {
        "saved_summary": artifact(args.output_dir / "summary.json"),
        "renderer": artifact(Path(__file__)),
        "numeric_metrics_recomputed": True,
        "inference_rerun": summary["inference_rerun"],
        "inference_executed_by_this_command": False,
        "training_rerun": False,
        "input_hashes_verified": True,
        "controlled_run_membership": summary["prediction_provenance"]["controlled_run_membership"],
    }
    (args.output_dir / "render_provenance.json").write_text(
        json.dumps(render_record, indent=2) + "\n", encoding="utf-8"
    )
    if args.paper_output_dir:
        args.paper_output_dir.mkdir(parents=True, exist_ok=True)
        write_diagnostics(summary, args.paper_output_dir / "vivqa_diagnostics.tex")
        (args.paper_output_dir / "render_provenance.json").write_text(
            json.dumps(render_record, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "prediction_provenance": summary["prediction_provenance"],
                "overall": summary["groups"]["Overall"],
                "tokenisation": summary["tokenisation"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
