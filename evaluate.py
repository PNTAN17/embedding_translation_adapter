"""Evaluate model predictions."""

import argparse
from pathlib import Path

from embedding_transfer.data import file_record, read_records, write_json, write_jsonl
from embedding_transfer.metrics import predict_records, score_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("vivqa", "mocheg"))
    parser.add_argument("--references", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and not args.predictions_output:
        parser.error(
            "--checkpoint requires --predictions-output to preserve the evaluation record"
        )
    inputs = [args.references, args.predictions, args.checkpoint]
    outputs = [args.output, args.predictions_output]
    if len({path.resolve() for path in outputs if path}) != len(
        [path for path in outputs if path]
    ) or any(
        out.resolve() == inp.resolve()
        for out in outputs
        if out
        for inp in inputs
        if inp
    ):
        parser.error("Use distinct input and output paths")
    references = read_records(args.references)
    provenance = {"references": file_record(args.references)}
    if args.predictions:
        predictions = read_records(args.predictions)
        provenance["predictions"] = file_record(args.predictions)
    else:
        if args.task == "vivqa":
            from embedding_transfer.vivqa import ViVQAPredictor as Predictor
        else:
            from embedding_transfer.mocheg import MOCHEGPredictor as Predictor
        predictor = Predictor(args.checkpoint, args.device, args.local_files_only)
        predictions = predict_records(references, predictor, args.task, args.image_root)
        write_jsonl(predictions, args.predictions_output)
        provenance["checkpoint"] = file_record(args.checkpoint)
        provenance["predictions"] = file_record(args.predictions_output)
    result = {
        "task": args.task,
        "artifacts": provenance,
        "metrics": score_records(references, predictions, args.task),
    }
    write_json(result, args.output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
