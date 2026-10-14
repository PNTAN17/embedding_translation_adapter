import copy
from pathlib import Path
import tempfile
import unittest

import torch

import train_vivqa
from embedding_transfer.data import file_record
from embedding_transfer.provenance import parse_config


ROOT = Path(__file__).resolve().parents[1]


def arguments(*options):
    return train_vivqa.build_parser().parse_args(
        [
            "--train-pairs",
            "train.jsonl",
            "--validation-pairs",
            "validation.jsonl",
            *options,
        ]
    )


class ValidationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [
            {"id": "a", "vi": "câu hỏi một", "en": "question one"},
            {"id": "b", "vi": "câu hỏi hai", "en": "question two"},
        ]
        self.tasks = [
            {
                "id": row["id"],
                "question_vi": row["vi"],
                "answer_en": "yes",
                "image": "image.jpg",
            }
            for row in self.pairs
        ]

    def test_full_validation_rejects_subset_but_generic_subset_is_supported(self):
        full = arguments(
            "--selection-metric", "exact_match", "--validation-task-scope", "full"
        )
        with self.assertRaisesRegex(ValueError, "cover every"):
            train_vivqa.validate_protocol(full, self.pairs, self.tasks[:1])
        train_vivqa.validate_protocol(full, self.pairs, self.tasks[::-1])
        train_vivqa.validate_protocol(arguments(), self.pairs, self.tasks[:1])

    def test_expected_count_is_optional_and_explicit(self):
        train_vivqa.validate_protocol(arguments(), self.pairs, self.tasks)
        strict = arguments("--expected-validation-count", "5000")
        with self.assertRaisesRegex(ValueError, "count"):
            train_vivqa.validate_protocol(strict, self.pairs, self.tasks)

    def test_invalid_task_fields_and_duplicate_ids_fail_before_training(self):
        for field in ("question_vi", "answer_en", "image"):
            rows = copy.deepcopy(self.tasks)
            rows[0][field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                train_vivqa.validate_protocol(arguments(), self.pairs, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            train_vivqa.validate_protocol(arguments(), self.pairs, [self.tasks[0]] * 2)
        rows = copy.deepcopy(self.tasks)
        rows[0]["question_vi"] = "different question"
        with self.assertRaisesRegex(ValueError, "questions must match"):
            train_vivqa.validate_protocol(arguments(), self.pairs, rows)

    def test_stages_require_correct_start_and_protect_parent_checkpoint(self):
        options = (
            ("--training-stage", "stage1"),
            (
                "--training-stage",
                "stage1",
                "--no-early-stopping",
                "--init-checkpoint",
                "stage1.pt",
            ),
            ("--training-stage", "stage2"),
            ("--init-checkpoint", "same.pt", "--output", "same.pt"),
        )
        for option in options:
            with self.subTest(option=option), self.assertRaises(ValueError):
                train_vivqa.validate_protocol(
                    arguments(*option), self.pairs, self.tasks
                )

    def test_checkpoint_and_history_outputs_cannot_overwrite_inputs(self):
        for name in (
            "train_pairs",
            "validation_pairs",
            "validation_task_data",
            "config",
        ):
            for output_file in ("result.pt", "result.run.json", "result.pt.tmp"):
                args = arguments("--output", "result.pt")
                setattr(args, name, Path(output_file))
                with (
                    self.subTest(name=name, output=output_file),
                    self.assertRaisesRegex(ValueError, "overwrite"),
                ):
                    train_vivqa.validate_protocol(args, self.pairs, self.tasks)

    def test_paper_configs_set_two_stage_budgets_and_full_validation(self):
        for stage in (1, 2):
            args = parse_config(
                train_vivqa.build_parser(),
                [
                    "--config",
                    str(ROOT / "configs" / f"vivqa_stage{stage}.json"),
                    "--train-pairs",
                    "train.jsonl",
                    "--validation-pairs",
                    "validation.jsonl",
                ],
            )
            with self.subTest(stage=stage):
                self.assertEqual(args.training_stage, f"stage{stage}")
                self.assertEqual(args.selection_metric, "exact_match")
                self.assertEqual(args.validation_task_scope, "full")
                self.assertEqual(args.expected_validation_count, 5000)
                self.assertEqual(args.epochs, 30)
                self.assertEqual(args.learning_rate, 2e-4 if stage == 1 else 1e-4)
                self.assertEqual(args.early_stopping, stage == 2)
                self.assertEqual(args.patience, 10)


class CheckpointInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "stage1.pt"
        self.config = {
            "input_embedding_dim": 768,
            "output_embedding_dim": 768,
            "hidden_dim": 16,
            "num_encoder_layers": 0,
            "num_decoder_layers": 0,
            "num_heads": 2,
            "dropout": 0.0,
            "max_seq_length": 4,
        }
        self.source = train_vivqa.EmbeddingTransferAdapter(**self.config)
        self.inputs = {
            name: {"sha256": name + "_hash"}
            for name in ("train_pairs", "validation_pairs", "validation_task_data")
        }
        self.payload = {
            "format_version": 1,
            "task": "vivqa",
            "adapter_config": self.config,
            "adapter_state_dict": self.source.state_dict(),
            "optimizer_state_dict": {"deliberately_not_a_valid_optimizer": True},
            "scheduler_state_dict": {"last_epoch": 99},
            "metadata": {
                "source_model": train_vivqa.SOURCE_MODEL,
                "target_model": train_vivqa.TARGET_MODEL,
                "training_stage": "stage1",
                "seed": 42,
                "selection_split": "validation",
                "selection_metric": "exact_match",
                "selected_epoch": 7,
                "provenance": {"inputs": self.inputs},
            },
        }
        self.args = arguments(
            "--training-stage",
            "stage2",
            "--selection-metric",
            "exact_match",
            "--init-checkpoint",
            str(self.checkpoint),
        )

    def initialize(self, payload=None, inputs=None):
        torch.save(self.payload if payload is None else payload, self.checkpoint)
        provenance = {
            "inputs": {
                **(self.inputs if inputs is None else inputs),
                "init_checkpoint": file_record(self.checkpoint),
            }
        }
        target = train_vivqa.EmbeddingTransferAdapter(**self.config)
        metadata = train_vivqa.initialize_adapter(
            target, self.config, self.args, provenance
        )
        return target, metadata

    def test_stage2_loads_exact_weights_without_restoring_training_state(self):
        target, metadata = self.initialize()
        for name, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[name], value), name)
        self.assertTrue(
            all(parameter.requires_grad for parameter in target.parameters())
        )
        self.assertEqual(metadata["mode"], "adapter_weights_only")
        self.assertEqual(metadata["optimizer_and_scheduler"], "new")
        self.assertEqual(metadata["source_selected_epoch"], 7)
        self.assertEqual(
            metadata["checkpoint"]["sha256"], file_record(self.checkpoint)["sha256"]
        )

    def test_wrong_task_architecture_model_seed_or_stage_is_rejected(self):
        changes = [
            ("task", None, "mocheg"),
            ("adapter_config", "dropout", 0.1),
            ("metadata", "source_model", "different-source"),
            ("metadata", "target_model", "different-target"),
            ("metadata", "seed", 43),
            ("metadata", "training_stage", "standalone"),
            ("metadata", "selection_metric", "alignment_loss"),
            ("metadata", "selection_split", "test"),
        ]
        for field, key, value in changes:
            payload = copy.deepcopy(self.payload)
            if key is None:
                payload[field] = value
            else:
                payload[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                self.initialize(payload)

    def test_stage2_rejects_changed_training_or_validation_artifacts(self):
        for name in self.inputs:
            inputs = copy.deepcopy(self.inputs)
            inputs[name]["sha256"] = "changed"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.initialize(inputs=inputs)

    def test_incomplete_weights_are_rejected_strictly(self):
        payload = copy.deepcopy(self.payload)
        del payload["adapter_state_dict"][next(iter(payload["adapter_state_dict"]))]
        with self.assertRaises(RuntimeError):
            self.initialize(payload)

    def test_nonfinite_weights_fail_before_initialization(self):
        payload = copy.deepcopy(self.payload)
        name = next(iter(payload["adapter_state_dict"]))
        payload["adapter_state_dict"][name].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "invalid adapter weights"):
            self.initialize(payload)


if __name__ == "__main__":
    unittest.main()
