#!/usr/bin/env python3
"""Generate bilingual text pairs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from filelock import FileLock, Timeout

from embedding_transfer.data import (
    read_records,
    record_id,
    sha256_file,
    write_json,
    write_jsonl,
)


PROMPT_VERSION = "vi-en-translation-v1"
GENERATOR_VERSION = 1
GENERATOR_SHA256 = sha256_file(Path(__file__).resolve())
SYSTEM_PROMPT = (
    "Translate one Vietnamese text into natural English. Preserve its meaning, "
    "named entities, numbers, and factual content. Do not answer the text, explain "
    "the translation, or add information. Return only the English translation."
)


class TranslationFailure(RuntimeError):
    """A translation request failed."""


class FatalProviderFailure(RuntimeError):
    """The provider configuration is invalid."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gemini", "openai"), required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Provider model ID.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum pending rows to process; 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs without API calls.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".metadata.json")


def transaction_path(output: Path) -> Path:
    return output.with_suffix(".transaction.json")


def staged_output_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".next")


def lock_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".lock")


def acquire_output_lock(output: Path):
    path = lock_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(
            f"Another generation process is using output {output.name!r}"
        ) from None
    return lock


def release_output_lock(lock: FileLock) -> None:
    lock.release()


def validate_args(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError("Input must be CSV or JSONL")
    if args.output.suffix.lower() != ".jsonl":
        raise ValueError("Output must be JSONL so it can be used by the trainers")
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        raise ValueError("Input and output must be different files")
    for name in ("workers", "max_attempts", "save_every", "max_output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(args.retry_delay)
        or not math.isfinite(args.request_timeout)
        or args.retry_delay < 0
        or args.limit < 0
        or args.request_timeout <= 0
    ):
        raise ValueError(
            "retry-delay and limit cannot be negative and request-timeout "
            "must be positive"
        )


def source_rows(path: Path) -> list[dict]:
    rows = read_records(path)
    seen: set[str] = set()
    result = []
    for row in rows:
        identifier = record_id(row)
        if identifier in seen:
            raise ValueError(f"Duplicate record id {identifier!r}")
        seen.add(identifier)
        text = row.get("vi")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Record {identifier!r} requires nonempty string field 'vi'"
            )
        result.append({"id": row["id"], "vi": text, "en": ""})
    if not result:
        raise ValueError(f"{path.name} contains no records")
    return result


def resumed_rows(source: list[dict], output: Path) -> list[dict]:
    if not output.exists():
        return source
    saved = read_records(output)
    if len(saved) != len(source):
        raise ValueError("Existing output has a different number of records")
    result = []
    for index, (expected, actual) in enumerate(zip(source, saved)):
        if (
            record_id(actual) != record_id(expected)
            or actual.get("vi") != expected["vi"]
        ):
            raise ValueError(f"Existing output does not match input at row {index}")
        translation = actual.get("en", "")
        if translation is None:
            translation = ""
        if not isinstance(translation, str):
            raise ValueError(
                f"Existing output has a nonstring translation at row {index}"
            )
        result.append({**expected, "en": translation})
    return result


def recover_transaction(output: Path) -> None:
    """Recover an interrupted save."""
    marker = transaction_path(output)
    if not marker.exists():
        return
    try:
        transaction = json.loads(marker.read_text(encoding="utf-8"))
        expected_hash = transaction["output_sha256"]
        metadata = transaction["metadata"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Generation recovery journal is invalid") from error
    if (
        not isinstance(expected_hash, str)
        or not isinstance(metadata, dict)
        or metadata.get("output_sha256") != expected_hash
    ):
        raise ValueError("Generation recovery journal is inconsistent")

    staged = staged_output_path(output)
    if staged.exists() and sha256_file(staged) == expected_hash:
        output.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(output)
    elif not output.exists() or sha256_file(output) != expected_hash:
        raise ValueError("Cannot recover the interrupted generation checkpoint")
    write_json(metadata, metadata_path(output))
    marker.unlink()


def load_resume_metadata(args: argparse.Namespace, input_hash: str) -> dict | None:
    path = metadata_path(args.output)
    if not args.output.exists():
        if path.exists():
            raise ValueError("Metadata exists without its output JSONL")
        return None
    if not path.exists():
        raise ValueError("Existing output has no metadata and cannot be resumed safely")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Existing generation metadata is invalid") from error
    if not isinstance(saved, dict):
        raise ValueError("Existing generation metadata must be a JSON object")
    expected = {
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": GENERATOR_SHA256,
        "provider": args.provider,
        "model_id": args.model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "input_sha256": input_hash,
        "max_output_tokens": args.max_output_tokens,
        "outer_max_attempts": args.max_attempts,
        "outer_retry_delay_seconds": args.retry_delay,
        "request_timeout_seconds": args.request_timeout,
        "workers": args.workers,
        "save_every": args.save_every,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Existing generation metadata differs in {key!r}")
    if saved.get("output_sha256") != sha256_file(args.output):
        raise ValueError("Existing output hash differs from generation metadata")
    return saved


def check_credentials(provider: str) -> None:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
    elif not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")


def clean_translation(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The provider returned an empty translation")
    return value.strip()


def completed_openai_translation(response: object) -> str:
    """Extract a completed translation."""
    if getattr(response, "status", None) != "completed":
        raise ValueError("The provider did not complete the translation")
    return clean_translation(getattr(response, "output_text", None))


def completed_gemini_translation(response: object) -> str:
    """Extract a completed translation."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ValueError("The provider returned no completed translation")
    reason = getattr(candidates[0], "finish_reason", None)
    if getattr(reason, "value", reason) != "STOP":
        raise ValueError("The provider did not complete the translation")
    return clean_translation(getattr(response, "text", None))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def client_pool(factory: Callable[[], object]):
    local = threading.local()
    clients: list[object] = []
    lock = threading.Lock()

    def get_client():
        client = getattr(local, "client", None)
        if client is None:
            client = factory()
            local.client = client
            with lock:
                clients.append(client)
        return client

    def close_clients() -> None:
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return get_client, close_clients


def build_translator(
    args: argparse.Namespace,
    expected_resolved_versions: set[str] | None = None,
) -> tuple[Callable[[str], str], str, Callable[[], None], Callable[[], list[str]]]:
    observed_versions = set(expected_resolved_versions or ())
    version_lock = threading.Lock()

    def observe_model_version(value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        version = value.strip()
        with version_lock:
            if observed_versions and version not in observed_versions:
                raise FatalProviderFailure(
                    "The provider resolved the model ID to a different version"
                )
            observed_versions.add(version)

    def resolved_versions() -> list[str]:
        with version_lock:
            return sorted(observed_versions)

    if args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install the generation dependencies from requirements-generation.txt"
            ) from error

        get_client, close_clients = client_pool(
            lambda: OpenAI(max_retries=0, timeout=args.request_timeout)
        )

        def translate(text: str) -> str:
            client = get_client()
            response = client.responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=text,
                max_output_tokens=args.max_output_tokens,
            )
            observe_model_version(getattr(response, "model", None))
            return completed_openai_translation(response)

        return (
            translate,
            package_version("openai"),
            close_clients,
            resolved_versions,
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "Install the generation dependencies from requirements-generation.txt"
        ) from error

    get_client, close_clients = client_pool(
        lambda: genai.Client(
            http_options=types.HttpOptions(
                timeout=max(1, round(args.request_timeout * 1000))
            )
        )
    )

    def translate(text: str) -> str:
        client = get_client()
        response = client.models.generate_content(
            model=args.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=args.max_output_tokens,
            ),
        )
        observe_model_version(getattr(response, "model_version", None))
        return completed_gemini_translation(response)

    return (
        translate,
        package_version("google-genai"),
        close_clients,
        resolved_versions,
    )


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
    name = type(error).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}:
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "api_key_invalid",
            "api key invalid",
            "api key not valid",
            "insufficient_quota",
            "unsupported parameter",
            "unknown parameter",
            "model_not_found",
            "model not found",
        )
    )


def is_permanent_error(error: Exception) -> bool:
    return error_status(error) in {400, 401, 402, 403, 404, 422} or type(
        error
    ).__name__ in {
        "AuthenticationError",
        "BadRequestError",
        "PermissionDeniedError",
    }


def translate_with_retry(
    identifier: str,
    text: str,
    translate: Callable[[str], str],
    max_attempts: int,
    retry_delay: float,
) -> str:
    for attempt in range(max_attempts):
        try:
            return clean_translation(translate(text))
        except FatalProviderFailure:
            raise
        except Exception as error:
            if is_global_error(error):
                raise FatalProviderFailure(
                    f"provider configuration failed with {type(error).__name__}"
                ) from None
            if is_permanent_error(error) or attempt + 1 == max_attempts:
                raise TranslationFailure(
                    f"record {identifier!r} failed with {type(error).__name__}"
                ) from None
            delay = min(60.0, retry_delay * (2**attempt))
            time.sleep(delay + random.random() * min(1.0, retry_delay))
    raise AssertionError("unreachable")


def build_metadata(
    args: argparse.Namespace,
    rows: list[dict],
    input_hash: str,
    created_at: str,
    sdk_version: str,
    status: str,
    failed_ids: list[str],
    output_hash: str,
    resolved_model_versions: list[str],
) -> dict:
    completed = sum(bool(row["en"].strip()) for row in rows)
    result = {
        "format_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": GENERATOR_SHA256,
        "status": status,
        "provider": args.provider,
        "model_id": args.model,
        "model_id_source": "requested command line value",
        "resolved_model_versions_observed": resolved_model_versions,
        "provider_api": (
            "OpenAI Responses API"
            if args.provider == "openai"
            else "Gemini generateContent API"
        ),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "source_language": "vi",
        "target_language": "en",
        "source_field": "vi",
        "target_field": "en",
        "temperature": None if args.provider == "openai" else 0.0,
        "max_output_tokens": args.max_output_tokens,
        "outer_max_attempts": args.max_attempts,
        "outer_retry_delay_seconds": args.retry_delay,
        "request_timeout_seconds": args.request_timeout,
        "workers": args.workers,
        "save_every": args.save_every,
        "input_file": args.input.name,
        "input_sha256": input_hash,
        "output_file": args.output.name,
        "output_sha256": output_hash,
        "rows_total": len(rows),
        "rows_completed": completed,
        "rows_failed_in_last_run": len(failed_ids),
        "failed_ids_in_last_run": failed_ids,
        "sdk_package": "openai" if args.provider == "openai" else "google-genai",
        "sdk_version": sdk_version,
        "created_at_utc": created_at,
        "updated_at_utc": utc_now(),
    }
    return result


def save_checkpoint(
    args: argparse.Namespace,
    rows: list[dict],
    input_hash: str,
    created_at: str,
    sdk_version: str,
    status: str,
    failed_ids: list[str],
    resolved_model_versions: list[str],
) -> None:
    staged = staged_output_path(args.output)
    write_jsonl(rows, staged)
    output_hash = sha256_file(staged)
    metadata = build_metadata(
        args,
        rows,
        input_hash,
        created_at,
        sdk_version,
        status,
        failed_ids,
        output_hash,
        resolved_model_versions,
    )
    write_json(
        {
            "format_version": 1,
            "output_sha256": output_hash,
            "metadata": metadata,
        },
        transaction_path(args.output),
    )
    staged.replace(args.output)
    write_json(metadata, metadata_path(args.output))
    transaction_path(args.output).unlink()


def run(
    args: argparse.Namespace,
    translate: Callable[[str], str] | None = None,
    sdk_version: str | None = None,
) -> int:
    validate_args(args)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    output_lock = acquire_output_lock(args.output)
    try:
        return _run_locked(args, translate, sdk_version)
    finally:
        release_output_lock(output_lock)


def _run_locked(
    args: argparse.Namespace,
    translate: Callable[[str], str] | None,
    sdk_version: str | None,
) -> int:
    input_hash = sha256_file(args.input)
    source = source_rows(args.input)
    recover_transaction(args.output)
    saved_metadata = load_resume_metadata(args, input_hash)
    rows = resumed_rows(source, args.output)
    pending = [index for index, row in enumerate(rows) if not row["en"].strip()]
    selected = pending[: args.limit] if args.limit else pending
    print(
        f"records={len(rows)} completed={len(rows) - len(pending)} "
        f"selected={len(selected)} provider={args.provider} model={args.model}",
        flush=True,
    )
    if args.dry_run or not selected:
        return 0

    recorded_versions = set()
    if saved_metadata:
        values = saved_metadata.get("resolved_model_versions_observed", [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError("Existing resolved model version metadata is invalid")
        recorded_versions.update(values)

    close_translator: Callable[[], None] = lambda: None

    def resolved_versions() -> list[str]:
        return sorted(recorded_versions)

    if translate is None:
        check_credentials(args.provider)
        (
            translate,
            detected_version,
            close_translator,
            resolved_versions,
        ) = build_translator(args, recorded_versions)
        sdk_version = detected_version
    created_at = (
        str(saved_metadata.get("created_at_utc")) if saved_metadata else utc_now()
    )
    sdk_version = sdk_version or "test-double"
    if saved_metadata and saved_metadata.get("sdk_version") != sdk_version:
        raise ValueError("Existing generation metadata differs in 'sdk_version'")

    try:
        save_checkpoint(
            args,
            rows,
            input_hash,
            created_at,
            sdk_version,
            "running",
            [],
            resolved_versions(),
        )

        failures: list[str] = []
        consecutive_failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(selected), args.save_every):
                batch = selected[start : start + args.save_every]
                futures: dict[Future[str], int] = {
                    executor.submit(
                        translate_with_retry,
                        record_id(rows[index]),
                        rows[index]["vi"],
                        translate,
                        args.max_attempts,
                        args.retry_delay,
                    ): index
                    for index in batch
                }
                fatal_error = None
                batch_successes = 0
                batch_failures = 0
                for future in as_completed(futures):
                    index = futures[future]
                    if future.cancelled():
                        continue
                    try:
                        rows[index]["en"] = future.result()
                        batch_successes += 1
                    except FatalProviderFailure as error:
                        if fatal_error is None:
                            fatal_error = error
                            for unfinished in futures:
                                unfinished.cancel()
                        failures.append(record_id(rows[index]))
                        batch_failures += 1
                    except TranslationFailure:
                        failures.append(record_id(rows[index]))
                        batch_failures += 1
                if batch_successes:
                    consecutive_failures = 0
                else:
                    consecutive_failures += batch_failures
                if fatal_error is None and consecutive_failures >= max(3, args.workers):
                    fatal_error = FatalProviderFailure(
                        "Repeated request failures stopped the generation job"
                    )
                completed = sum(bool(row["en"].strip()) for row in rows)
                if fatal_error:
                    status = "failed"
                else:
                    status = "complete" if completed == len(rows) else "partial"
                save_checkpoint(
                    args,
                    rows,
                    input_hash,
                    created_at,
                    sdk_version,
                    status,
                    failures,
                    resolved_versions(),
                )
                print(f"saved={completed}/{len(rows)}", flush=True)
                if fatal_error:
                    raise fatal_error
    finally:
        close_translator()

    if failures:
        print(
            f"failed={len(failures)} ids={','.join(failures[:20])}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same command.", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
