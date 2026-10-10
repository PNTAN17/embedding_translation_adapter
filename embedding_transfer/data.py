"""Dataset checks and provenance without heavyweight model dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def read_records(path: str | Path) -> list[dict]:
    path = Path(path).expanduser()
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".csv":
            records = list(csv.DictReader(handle))
        else:
            records = []
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path.name} at line {line_number}"
                    ) from error
    if not records or any(not isinstance(item, dict) for item in records):
        raise ValueError(f"{path.name} must contain nonempty object records")
    return records


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def record_id(record: dict) -> str:
    value = record.get("id")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, str))
        or not str(value).strip()
    ):
        raise ValueError(
            "Every record requires an original nonempty string or integer id"
        )
    return str(value)


def indexed_records(records: list[dict]) -> dict[str, dict]:
    result = {}
    for record in records:
        identifier = record_id(record)
        if identifier in result:
            raise ValueError(f"Duplicate record id {identifier!r}")
        result[identifier] = record
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict:
    path = Path(path).expanduser()
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(value, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def text_keys(record: dict, task: str) -> set[tuple[str, str]]:
    """Check either language independently, including MOCHEG evidence."""
    pairs = [(record.get("vi"), record.get("en"))]
    if task == "mocheg":
        vi = record.get("evidence_vi")
        en = record.get("evidence_en")
        if (
            not isinstance(vi, list)
            or not isinstance(en, list)
            or not vi
            or len(vi) != len(en)
        ):
            raise ValueError(
                "MOCHEG needs equal nonempty evidence_vi and evidence_en lists"
            )
        pairs = [(record.get("claim_vi"), record.get("claim_en")), *zip(vi, en)]
    result = set()
    for vi, en in pairs:
        for language, value in (("vi", vi), ("en", en)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Expected nonempty {language} text")
            result.add((language, normalized_text(value)))
    return result


def prepare_partitions(
    train: list[dict], validation: list[dict], test: list[dict], task: str
):
    partitions = {
        name: indexed_records(rows)
        for name, rows in (("train", train), ("validation", validation), ("test", test))
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if partitions[left].keys() & partitions[right].keys():
            raise ValueError(f"Original IDs overlap between {left} and {right}")
    held_out_keys = set().union(*(text_keys(row, task) for row in validation + test))
    seen = set()
    kept, removed = [], []
    for row in train:
        keys = text_keys(row, task)
        if task == "mocheg":
            duplicate_key = (
                normalized_text(row["claim_vi"]),
                normalized_text(row["claim_en"]),
                tuple(
                    (normalized_text(vi), normalized_text(en))
                    for vi, en in zip(row["evidence_vi"], row["evidence_en"])
                ),
            )
        else:
            duplicate_key = (normalized_text(row["vi"]), normalized_text(row["en"]))
        if keys & held_out_keys:
            removed.append({"id": record_id(row), "reason": "held_out_text_overlap"})
        elif duplicate_key in seen:
            removed.append(
                {"id": record_id(row), "reason": "duplicate_training_record"}
            )
        else:
            kept.append(row)
            seen.add(duplicate_key)
    if not kept:
        raise ValueError("No training records remain after exclusion")
    return kept, removed
