"""Validate existing partitions, remove training overlaps, and save original IDs."""

import argparse
from pathlib import Path

from embedding_transfer.data import (
    file_record,
    prepare_partitions,
    read_records,
    record_id,
    write_json,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("parallel", "mocheg"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("train", "validation", "test")}
    if any(
        path.resolve().parent == args.output_dir.resolve() for path in paths.values()
    ):
        raise ValueError("Use a separate output directory to preserve source files")
    rows = {name: read_records(path) for name, path in paths.items()}
    clean, removed = prepare_partitions(
        rows["train"], rows["validation"], rows["test"], args.task
    )
    manifest = {
        "format_version": 1,
        "scope": "user_supplied_partitions_not_certified_paper_splits",
        "task": args.task,
        "normalization": "NFKC casefold whitespace; either language independently",
        "inputs": {name: file_record(path) for name, path in paths.items()},
        "original_ids": {
            name: [record_id(row) for row in records] for name, records in rows.items()
        },
        "retained_train_ids": [record_id(row) for row in clean],
        "removed_train_records": removed,
    }
    rows["train"] = clean
    outputs = {}
    for name, records in rows.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(records, path)
        outputs[name] = file_record(path)
    manifest["outputs"] = outputs
    write_json(manifest, args.output_dir / "split_manifest.json")
    print(f"retained_train={len(clean)} removed_train={len(removed)}")


if __name__ == "__main__":
    main()
