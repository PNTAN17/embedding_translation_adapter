import contextlib
import csv
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch

import generate_pairs


def arguments(root: Path, **overrides):
    values = {
        "provider": "openai",
        "model": "test-model",
        "input": root / "source.jsonl",
        "output": root / "pairs.jsonl",
        "workers": 2,
        "max_attempts": 3,
        "retry_delay": 0.0,
        "request_timeout": 30.0,
        "save_every": 1,
        "max_output_tokens": 64,
        "limit": 0,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class PairGenerationTests(unittest.TestCase):
    def test_generates_trainer_ready_pairs_and_completed_rerun_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(
                args.input,
                [
                    {"id": "q1", "vi": "Có bao nhiêu người?", "unused": 1},
                    {"id": 2, "vi": "Chiếc xe màu gì?"},
                ],
            )
            translations = {
                "Có bao nhiêu người?": "How many people are there?",
                "Chiếc xe màu gì?": "What colour is the car?",
            }
            calls = []

            def translate(text):
                calls.append(text)
                return translations[text]

            self.assertEqual(generate_pairs.run(args, translate, "test-sdk"), 0)
            rows = generate_pairs.read_records(args.output)
            self.assertEqual(
                rows,
                [
                    {
                        "id": "q1",
                        "vi": "Có bao nhiêu người?",
                        "en": "How many people are there?",
                    },
                    {
                        "id": 2,
                        "vi": "Chiếc xe màu gì?",
                        "en": "What colour is the car?",
                    },
                ],
            )
            self.assertCountEqual(calls, list(translations))
            before = args.output.read_bytes()

            def should_not_run(_text):
                raise AssertionError("completed rows must not be translated again")

            self.assertEqual(generate_pairs.run(args, should_not_run, "test-sdk"), 0)
            self.assertEqual(args.output.read_bytes(), before)

    def test_csv_input_preserves_order_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "vi"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "a", "vi": "Một, hai"},
                        {"id": "b", "vi": "Dòng một\ndòng hai"},
                    ]
                )
            args = arguments(root, input=source, provider="gemini")
            generated = {
                "Một, hai": "One, two",
                "Dòng một\ndòng hai": "Line one\nline two",
            }
            self.assertEqual(generate_pairs.run(args, generated.__getitem__, "test"), 0)
            rows = generate_pairs.read_records(args.output)
            self.assertEqual([row["id"] for row in rows], ["a", "b"])
            self.assertEqual([row["en"] for row in rows], list(generated.values()))

    def test_resume_translates_only_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root, max_attempts=1)
            write_jsonl(
                args.input,
                [{"id": "a", "vi": "một"}, {"id": "b", "vi": "hai"}],
            )

            def first_run(text):
                if text == "hai":
                    raise RuntimeError("temporary")
                return text.upper()

            self.assertEqual(generate_pairs.run(args, first_run, "test"), 1)
            calls = []

            def translate(text):
                calls.append(text)
                return "two"

            with self.assertRaisesRegex(ValueError, "sdk_version"):
                generate_pairs.run(args, translate, "different-sdk")
            self.assertEqual(generate_pairs.run(args, translate, "test"), 0)
            self.assertEqual(calls, ["hai"])
            self.assertEqual(generate_pairs.read_records(args.output)[0]["en"], "MỘT")

    def test_changed_model_or_source_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            self.assertEqual(generate_pairs.run(args, lambda _text: "one", "test"), 0)
            with self.assertRaisesRegex(ValueError, "model_id"):
                generate_pairs.run(
                    arguments(root, model="different-model", dry_run=True),
                    lambda _text: "one",
                    "test",
                )
            with self.assertRaisesRegex(ValueError, "workers"):
                generate_pairs.run(
                    arguments(root, workers=1, dry_run=True),
                    lambda _text: "one",
                    "test",
                )
            write_jsonl(args.input, [{"id": "a", "vi": "đã đổi"}])
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                generate_pairs.run(args, lambda _text: "changed", "test")

    def test_missing_key_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    generate_pairs.run(args)
            self.assertFalse(args.output.exists())
            self.assertFalse(generate_pairs.metadata_path(args.output).exists())

    def test_provider_error_does_not_leak_environment_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root, max_attempts=1)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            secret = "unit-test-secret-value"

            def fail(_text):
                raise RuntimeError(f"provider request contained {secret}")

            stderr = io.StringIO()
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(generate_pairs.run(args, fail, "test"), 1)
            self.assertNotIn(secret, stderr.getvalue())
            self.assertNotIn(secret, args.output.read_text(encoding="utf-8"))
            self.assertNotIn(
                secret,
                generate_pairs.metadata_path(args.output).read_text(encoding="utf-8"),
            )

    def test_transient_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            attempts = 0

            def flaky(_text):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise TimeoutError("temporary")
                return "one"

            with patch("generate_pairs.random.random", return_value=0.0):
                self.assertEqual(generate_pairs.run(args, flaky, "test"), 0)
            self.assertEqual(attempts, 2)
            self.assertEqual(generate_pairs.read_records(args.output)[0]["en"], "one")

    def test_outer_quotes_in_a_translation_are_preserved(self):
        self.assertEqual(generate_pairs.clean_translation('  "Hello."  '), '"Hello."')

    def test_openai_adapter_uses_responses_api_without_explicit_key(self):
        calls = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    output_text="one", model="resolved-openai", status="completed"
                )

        class Client:
            def __init__(self, **kwargs):
                self.responses = Responses()
                self.options = kwargs

            def close(self):
                self.closed = True

        module = ModuleType("openai")
        module.OpenAI = Client
        args = arguments(Path("."))
        with patch.dict(sys.modules, {"openai": module}):
            translate, _version, close, versions = generate_pairs.build_translator(args)
            self.assertEqual(translate("một"), "one")
            self.assertEqual(versions(), ["resolved-openai"])
            close()
        self.assertEqual(
            calls,
            [
                {
                    "model": "test-model",
                    "instructions": generate_pairs.SYSTEM_PROMPT,
                    "input": "một",
                    "max_output_tokens": 64,
                }
            ],
        )

    def test_gemini_adapter_uses_generate_content_without_explicit_key(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    text="one",
                    model_version="resolved-gemini",
                    candidates=[SimpleNamespace(finish_reason="STOP")],
                )

        class Client:
            def __init__(self, **kwargs):
                self.models = Models()
                self.options = kwargs

            def close(self):
                self.closed = True

        class HttpOptions:
            def __init__(self, **kwargs):
                self.values = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.values = kwargs

        google = ModuleType("google")
        genai = ModuleType("google.genai")
        genai.Client = Client
        genai.types = SimpleNamespace(
            GenerateContentConfig=GenerateContentConfig,
            HttpOptions=HttpOptions,
        )
        google.genai = genai
        args = arguments(Path("."), provider="gemini")
        with patch.dict(sys.modules, {"google": google, "google.genai": genai}):
            translate, _version, close, versions = generate_pairs.build_translator(args)
            self.assertEqual(translate("một"), "one")
            self.assertEqual(versions(), ["resolved-gemini"])
            close()
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertEqual(calls[0]["contents"], "một")
        self.assertEqual(
            calls[0]["config"].values,
            {
                "system_instruction": generate_pairs.SYSTEM_PROMPT,
                "temperature": 0.0,
                "max_output_tokens": 64,
            },
        )

    def test_incomplete_provider_text_is_not_saved_as_a_translation(self):
        responses = (
            (
                generate_pairs.completed_openai_translation,
                SimpleNamespace(output_text="The car is", status="incomplete"),
            ),
            (
                generate_pairs.completed_gemini_translation,
                SimpleNamespace(
                    text="The car is",
                    candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
                ),
            ),
        )
        for extract, response in responses:
            with self.subTest(provider=extract.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    args = arguments(Path(directory), max_attempts=1)
                    write_jsonl(args.input, [{"id": "a", "vi": "Xe màu đỏ"}])
                    self.assertEqual(
                        generate_pairs.run(
                            args, lambda _text: extract(response), "test"
                        ),
                        1,
                    )
                    self.assertEqual(
                        generate_pairs.read_records(args.output)[0]["en"], ""
                    )
                    metadata = json.loads(
                        generate_pairs.metadata_path(args.output).read_text()
                    )
                    self.assertEqual(metadata["rows_completed"], 0)
                    self.assertEqual(metadata["failed_ids_in_last_run"], ["a"])

    def test_missing_completion_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            generate_pairs.completed_openai_translation(
                SimpleNamespace(output_text="one")
            )
        with self.assertRaisesRegex(ValueError, "complete"):
            generate_pairs.completed_gemini_translation(
                SimpleNamespace(text="one", candidates=[])
            )
        self.assertEqual(
            generate_pairs.completed_gemini_translation(
                SimpleNamespace(
                    text=" one ",
                    candidates=[
                        SimpleNamespace(finish_reason=SimpleNamespace(value="STOP"))
                    ],
                )
            ),
            "one",
        )

    def test_nonfinite_timeout_or_retry_delay_is_rejected(self):
        for name in ("request_timeout", "retry_delay"):
            for value in (float("nan"), float("inf")):
                with self.subTest(option=name, value=value):
                    with self.assertRaises(ValueError):
                        generate_pairs.validate_args(
                            arguments(Path("."), **{name: value})
                        )

    def test_auth_failure_stops_before_later_batches(self):
        class AuthenticationError(RuntimeError):
            status_code = 401

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root, workers=1, save_every=1)
            write_jsonl(
                args.input,
                [{"id": index, "vi": f"câu {index}"} for index in range(3)],
            )
            calls = []

            def fail(text):
                calls.append(text)
                raise AuthenticationError("secret provider detail")

            with self.assertRaises(generate_pairs.FatalProviderFailure):
                generate_pairs.run(args, fail, "test")
            self.assertEqual(calls, ["câu 0"])
            metadata = json.loads(
                generate_pairs.metadata_path(args.output).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")

    def test_repeated_bad_requests_stop_before_all_rows(self):
        class BadRequestError(RuntimeError):
            status_code = 400

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(
                root,
                workers=2,
                save_every=2,
                max_attempts=1,
            )
            write_jsonl(
                args.input,
                [{"id": index, "vi": f"câu {index}"} for index in range(8)],
            )
            calls = []

            def fail(text):
                calls.append(text)
                raise BadRequestError("bad request")

            with self.assertRaises(generate_pairs.FatalProviderFailure):
                generate_pairs.run(args, fail, "test")
            self.assertLess(len(calls), 8)

    def test_second_writer_for_the_same_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root, dry_run=True)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            lock = generate_pairs.acquire_output_lock(args.output)
            try:
                with self.assertRaisesRegex(RuntimeError, "Another generation"):
                    generate_pairs.run(args, lambda _text: "one", "test")
            finally:
                generate_pairs.release_output_lock(lock)

    def test_interrupted_checkpoint_transaction_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            rows = [{"id": "a", "vi": "một", "en": "one"}]
            staged = generate_pairs.staged_output_path(args.output)
            generate_pairs.write_jsonl(rows, staged)
            output_hash = generate_pairs.sha256_file(staged)
            metadata = generate_pairs.build_metadata(
                args,
                rows,
                generate_pairs.sha256_file(args.input),
                "created",
                "test",
                "complete",
                [],
                output_hash,
                [],
            )
            generate_pairs.write_json(
                {
                    "format_version": 1,
                    "output_sha256": output_hash,
                    "metadata": metadata,
                },
                generate_pairs.transaction_path(args.output),
            )

            generate_pairs.recover_transaction(args.output)

            self.assertEqual(generate_pairs.read_records(args.output), rows)
            self.assertFalse(staged.exists())
            self.assertFalse(generate_pairs.transaction_path(args.output).exists())
            self.assertEqual(
                generate_pairs.sha256_file(args.output),
                json.loads(
                    generate_pairs.metadata_path(args.output).read_text(
                        encoding="utf-8"
                    )
                )["output_sha256"],
            )

    def test_invalid_source_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(
                args.input,
                [{"id": "same", "vi": "một"}, {"id": "same", "vi": "hai"}],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate record id"):
                generate_pairs.run(args, lambda text: text, "test")
            write_jsonl(args.input, [{"id": "a", "vi": "  "}])
            with self.assertRaisesRegex(ValueError, "nonempty"):
                generate_pairs.run(args, lambda text: text, "test")

    def test_nonobject_resume_metadata_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            write_jsonl(args.input, [{"id": "a", "vi": "một"}])
            self.assertEqual(generate_pairs.run(args, lambda _text: "one", "test"), 0)
            generate_pairs.metadata_path(args.output).write_text(
                "[]\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "JSON object"):
                generate_pairs.run(args, lambda _text: "one", "test")

    def test_command_line_has_no_api_key_argument(self):
        source = Path(generate_pairs.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--api-key", source)

    def test_release_text_files_contain_no_api_key_like_secret(self):
        root = Path(generate_pairs.__file__).parent
        pattern = re.compile(
            rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{20,})"
        )
        matches = []
        for path in root.rglob("*"):
            if path.is_file():
                tail = b""
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        data = tail + chunk
                        if pattern.search(data):
                            matches.append(str(path.relative_to(root)))
                            break
                        tail = data[-512:]
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
