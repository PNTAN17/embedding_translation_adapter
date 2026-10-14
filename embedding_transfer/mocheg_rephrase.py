"""Generate paraphrases for MOCHEG claims."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .data import normalized_text, read_records, sha256_file, write_json, write_jsonl


PROMPT_VERSION = "mocheg-vi-paraphrase-v1"
GENERATOR_VERSION = 1
LIST_PREFIX = re.compile(r"^\s*(?:(?:[-*•])|(?:\(?\d{1,2}[.)\-:]))\s*")
SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)
SAFETY_THRESHOLD = "BLOCK_NONE"


class RephraseFailure(RuntimeError):
    """A paraphrasing request failed."""


class FatalProviderFailure(RuntimeError):
    """The provider configuration is invalid."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train",),
        required=True,
        help="Dataset split.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Response cache path.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Generation model ID.",
    )
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum pending claims to process; 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs without API calls.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_cache_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".cache.jsonl")


def metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".metadata.json")


def validate_args(args: argparse.Namespace) -> None:
    if args.split != "train":
        raise ValueError("MOCHEG paraphrasing is restricted to the training split")
    if args.input.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError("Input must be CSV or JSONL")
    if args.output.suffix.lower() != args.input.suffix.lower():
        raise ValueError("Input and output must use the same CSV or JSONL format")
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        raise ValueError("Input and output must be different files")
    cache = args.cache or default_cache_path(args.output)
    resolved_paths = {
        "input": args.input.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
        "cache": cache.expanduser().resolve(),
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise ValueError("Input, output, and cache must be different files")
    for name in (
        "variants",
        "workers",
        "max_attempts",
        "save_every",
        "max_output_tokens",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.retry_delay < 0 or args.limit < 0 or args.request_timeout <= 0:
        raise ValueError(
            "retry-delay and limit cannot be negative and request-timeout "
            "must be positive"
        )


def detect_schema(rows: list[dict], suffix: str) -> tuple[str, str]:
    """Identify input fields."""
    canonical = all("id" in row and "claim_vi" in row for row in rows)
    historical = all("claim_id" in row and "Claim_vi" in row for row in rows)
    if canonical:
        id_field, claim_field = "id", "claim_vi"
    elif historical:
        id_field, claim_field = "claim_id", "Claim_vi"
    else:
        raise ValueError(
            "Input must contain either id and claim_vi or claim_id and Claim_vi"
        )
    if suffix == ".jsonl":
        identifiers = [source_id(row, id_field) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Canonical JSONL input requires unique claim IDs")
    return id_field, claim_field


def source_id(row: dict, id_field: str) -> str:
    value = row.get(id_field)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Every row requires a string or integer {id_field}")
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"Every row requires a nonempty {id_field}")
    return identifier


def unique_claims(
    rows: list[dict], id_field: str, claim_field: str
) -> list[tuple[str, str]]:
    """Collect unique claims."""
    by_identifier: dict[str, str] = {}
    ordered: list[tuple[str, str]] = []
    for row in rows:
        identifier = source_id(row, id_field)
        value = row.get(claim_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Claim {identifier!r} requires nonempty string {claim_field}"
            )
        claim = value.strip()
        previous = by_identifier.get(identifier)
        if previous is None:
            by_identifier[identifier] = claim
            ordered.append((identifier, claim))
        elif normalized_text(previous) != normalized_text(claim):
            raise ValueError(
                f"Claim ID {identifier!r} maps to multiple Vietnamese claims"
            )
    return ordered


def build_prompt(claim_vi: str, variants: int) -> str:
    """Build the paraphrasing prompt."""
    return (
        f"Hãy diễn đạt lại câu tiếng Việt sau theo {variants} cách khác nhau, "
        "giữ nguyên ý nghĩa nhưng đổi từ ngữ và cấu trúc câu. "
        f"Chỉ trả về {variants} cách diễn đạt, mỗi cách trên một dòng, "
        "không đánh số, không thêm gì khác.\n\n"
        f"Câu gốc: {claim_vi}"
    )


def clean_variants(text: str, original: str, expected: int) -> list[str]:
    """Parse and validate paraphrases."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini returned an empty response")
    original_key = normalized_text(original)
    seen = {original_key}
    result = []
    for line in text.splitlines():
        candidate = LIST_PREFIX.sub("", line).strip()
        if not candidate:
            continue
        key = normalized_text(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) == expected:
            break
    if len(result) != expected:
        raise ValueError(
            f"Gemini returned {len(result)} distinct changed paraphrases, "
            f"expected {expected}"
        )
    return result


def cache_key(claim_vi: str, model: str, variants: int) -> tuple[str, str, int, str]:
    return model, PROMPT_VERSION, variants, normalized_text(claim_vi)


def load_cache(
    path: Path, model: str, variants: int
) -> dict[tuple[str, str, int, str], list[str]]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, int, str], list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            claim = item["claim_vi"]
            values = item["paraphrases"]
            item_model = item.get("model_id", model)
            item_variants = int(item.get("variants", variants))
            item_prompt = item.get("prompt_version", PROMPT_VERSION)
            if (
                item_model != model
                or item_variants != variants
                or item_prompt != PROMPT_VERSION
            ):
                continue
            key = cache_key(
                claim,
                item_model,
                item_variants,
            )
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise TypeError
            parsed = clean_variants("\n".join(values), claim, variants)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid rephrase cache at line {line_number}") from error
        if key[0] == model and key[2] == variants:
            cache[key] = parsed
    return cache


def cache_rows(
    cache: dict[tuple[str, str, int, str], list[str]],
    claim_text: dict[tuple[str, str, int, str], str],
) -> list[dict]:
    result = []
    for key in sorted(cache):
        model, prompt_version, variants, _ = key
        result.append(
            {
                "claim_vi": claim_text[key],
                "paraphrases": cache[key],
                "model_id": model,
                "prompt_version": prompt_version,
                "variants": variants,
            }
        )
    return result


def augmented_rows(
    rows: list[dict],
    id_field: str,
    claim_field: str,
    variants_by_claim: dict[str, list[str]],
    suffix: str,
) -> list[dict]:
    """Build augmented data records."""
    result = []
    for row in rows:
        identifier = source_id(row, id_field)
        values = [str(row[claim_field]).strip(), *variants_by_claim.get(identifier, [])]
        for variant_index, value in enumerate(values):
            item = dict(row)
            item[claim_field] = value
            item["variant_idx"] = variant_index
            if suffix == ".jsonl":
                item["source_id"] = row[id_field]
                if variant_index:
                    item[id_field] = f"{identifier}::paraphrase:{variant_index}"
            result.append(item)
    if suffix == ".jsonl":
        identifiers = [source_id(row, id_field) for row in result]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Generated paraphrase IDs collide with source IDs")
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty augmented dataset")
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_output(rows: list[dict], path: Path) -> None:
    if path.suffix.lower() == ".csv":
        write_csv(rows, path)
    else:
        write_jsonl(rows, path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def error_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def is_global_error(error: Exception) -> bool:
    status = error_status(error)
    if status in {401, 402, 403, 404}:
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "api key not valid",
            "api_key_invalid",
            "insufficient_quota",
            "model not found",
            "model_not_found",
        )
    )


def is_permanent_error(error: Exception) -> bool:
    return error_status(error) in {400, 401, 402, 403, 404, 422}


def build_gemini_rephraser(
    args: argparse.Namespace,
) -> tuple[Callable[[str], list[str]], str, Callable[[], None], Callable[[], list[str]]]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-generation.txt before calling Gemini"
        ) from error

    local = threading.local()
    clients = []
    clients_lock = threading.Lock()
    observed_versions: set[str] = set()
    versions_lock = threading.Lock()

    def client():
        value = getattr(local, "client", None)
        if value is None:
            value = genai.Client(
                http_options=types.HttpOptions(
                    timeout=max(1, round(args.request_timeout * 1000))
                )
            )
            local.client = value
            with clients_lock:
                clients.append(value)
        return value

    def rephrase(claim_vi: str) -> list[str]:
        response = client().models.generate_content(
            model=args.model,
            contents=build_prompt(claim_vi, args.variants),
            config=types.GenerateContentConfig(
                max_output_tokens=args.max_output_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category=category,
                        threshold=SAFETY_THRESHOLD,
                    )
                    for category in SAFETY_CATEGORIES
                ],
            ),
        )
        version = getattr(response, "model_version", None)
        if isinstance(version, str) and version.strip():
            with versions_lock:
                observed_versions.add(version.strip())
        return clean_variants(response.text, claim_vi, args.variants)

    def close() -> None:
        for value in clients:
            method = getattr(value, "close", None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def resolved_versions() -> list[str]:
        with versions_lock:
            return sorted(observed_versions)

    return rephrase, package_version("google-genai"), close, resolved_versions


def rephrase_with_retry(
    identifier: str,
    claim_vi: str,
    rephrase: Callable[[str], list[str]],
    max_attempts: int,
    retry_delay: float,
) -> list[str]:
    for attempt in range(max_attempts):
        try:
            return rephrase(claim_vi)
        except Exception as error:
            if is_global_error(error):
                raise FatalProviderFailure(
                    f"Gemini configuration failed with {type(error).__name__}"
                ) from None
            if is_permanent_error(error) or attempt + 1 == max_attempts:
                raise RephraseFailure(
                    f"claim {identifier!r} failed with {type(error).__name__}"
                ) from None
            delay = min(60.0, retry_delay * (2**attempt))
            time.sleep(delay + random.random() * min(1.0, retry_delay))
    raise AssertionError("unreachable")


def build_metadata(
    args: argparse.Namespace,
    input_hash: str,
    rows: list[dict],
    claims: list[tuple[str, str]],
    completed: int,
    failures: list[str],
    sdk_version: str,
    resolved_versions: Iterable[str],
    status: str,
) -> dict:
    return {
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "status": status,
        "split": args.split,
        "provider": "gemini",
        "model_id": args.model,
        "resolved_model_versions_observed": sorted(resolved_versions),
        "sdk_package": "google-genai",
        "sdk_version": sdk_version,
        "prompt_version": PROMPT_VERSION,
        "prompt_template": build_prompt("{claim_vi}", args.variants),
        "variants_requested": args.variants,
        "temperature": None,
        "safety_settings": [
            {"category": category, "threshold": SAFETY_THRESHOLD}
            for category in SAFETY_CATEGORIES
        ],
        "workers": args.workers,
        "max_output_tokens": args.max_output_tokens,
        "outer_max_attempts": args.max_attempts,
        "outer_retry_delay_seconds": args.retry_delay,
        "request_timeout_seconds": args.request_timeout,
        "input_file": args.input.name,
        "input_sha256": input_hash,
        "output_file": args.output.name,
        "output_sha256": sha256_file(args.output),
        "source_rows": len(rows),
        "unique_claim_ids": len(claims),
        "claims_with_complete_paraphrases": completed,
        "failed_claim_ids_in_last_run": failures,
        "augmented_rows": len(read_records(args.output)),
        "updated_at_utc": utc_now(),
        "credentials_recorded": False,
    }


def run(
    args: argparse.Namespace,
    rephrase: Callable[[str], list[str]] | None = None,
    sdk_version: str | None = None,
) -> int:
    validate_args(args)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    rows = read_records(args.input)
    suffix = args.input.suffix.lower()
    id_field, claim_field = detect_schema(rows, suffix)
    claims = unique_claims(rows, id_field, claim_field)
    cache_path = args.cache or default_cache_path(args.output)
    cache = load_cache(cache_path, args.model, args.variants)
    claim_text = {
        cache_key(claim, args.model, args.variants): claim for _, claim in claims
    }
    cache = {key: value for key, value in cache.items() if key in claim_text}

    unique_texts: list[tuple[str, str]] = []
    seen_text_keys = set()
    for identifier, claim in claims:
        key = cache_key(claim, args.model, args.variants)
        if key not in seen_text_keys:
            unique_texts.append((identifier, claim))
            seen_text_keys.add(key)

    pending = [
        (identifier, claim)
        for identifier, claim in unique_texts
        if cache_key(claim, args.model, args.variants) not in cache
    ]
    selected = pending[: args.limit] if args.limit else pending
    print(
        f"source_rows={len(rows)} unique_claim_ids={len(claims)} "
        f"unique_claim_texts={len(unique_texts)} "
        f"cached_texts={len(unique_texts) - len(pending)} "
        f"selected={len(selected)} "
        f"model={args.model}",
        flush=True,
    )
    if args.dry_run:
        return 0

    resolved_versions: Callable[[], list[str]] = lambda: []
    close: Callable[[], None] = lambda: None
    if rephrase is None and selected:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        rephrase, sdk_version, close, resolved_versions = build_gemini_rephraser(args)
    elif rephrase is None:
        rephrase = lambda _claim: []
    sdk_version = sdk_version or "test-double"

    failures: list[str] = []
    variants_by_claim: dict[str, list[str]] = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(selected), args.save_every):
                batch = selected[start : start + args.save_every]
                futures: dict[Future[list[str]], tuple[str, str]] = {
                    executor.submit(
                        rephrase_with_retry,
                        identifier,
                        claim,
                        rephrase,
                        args.max_attempts,
                        args.retry_delay,
                    ): (identifier, claim)
                    for identifier, claim in batch
                }
                fatal = None
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    identifier, claim = futures[future]
                    try:
                        values = future.result()
                        key = cache_key(claim, args.model, args.variants)
                        cache[key] = values
                        claim_text[key] = claim
                    except FatalProviderFailure as error:
                        fatal = error
                        failures.append(identifier)
                        for unfinished in futures:
                            unfinished.cancel()
                    except RephraseFailure:
                        failures.append(identifier)
                write_jsonl(cache_rows(cache, claim_text), cache_path)
                if fatal is not None:
                    raise fatal

        for identifier, claim in claims:
            values = cache.get(cache_key(claim, args.model, args.variants))
            if values is not None:
                variants_by_claim[identifier] = values
        augmented = augmented_rows(
            rows, id_field, claim_field, variants_by_claim, suffix
        )
        write_output(augmented, args.output)
        completed = sum(identifier in variants_by_claim for identifier, _ in claims)
        status = "complete" if completed == len(claims) else "partial"
        write_json(
            build_metadata(
                args,
                sha256_file(args.input),
                rows,
                claims,
                completed,
                failures,
                sdk_version,
                resolved_versions(),
                status,
            ),
            metadata_path(args.output),
        )
        print(
            f"status={status} completed_claims={completed}/{len(claims)} "
            f"augmented_rows={len(augmented)} output={args.output}",
            flush=True,
        )
        return 0 if status == "complete" else 1
    finally:
        close()


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
