import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from recompute_supplementary import (
    align_rows,
    artifact,
    bootstrap,
    checkpoint_provenance,
    score_rows,
    verify_record,
    verify_scores,
)


class SupplementaryTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            {
                "img_id": "11",
                "type": "0",
                "question": "one",
                "answer_english": "red car",
            },
            {"img_id": "11", "type": "0", "question": "two", "answer_english": "blue"},
        ]

    def test_alignment_uses_question_identity_and_original_ids(self):
        first = {**self.references[0], "sample_id": "0", "prediction": "red"}
        second = {**self.references[1], "sample_id": "1", "prediction": "blue"}
        self.assertEqual(align_rows(self.references, [second, first]), [first, second])
        with self.assertRaises(ValueError):
            align_rows(self.references, [first, first])
        with self.assertRaises(ValueError):
            align_rows(self.references, [self.references[1], self.references[0]])

    def test_normalized_answers_drive_metrics_not_cached_scores(self):
        predictions = [
            {"prediction": "red", "exact": 1},
            {"prediction": "blue", "exact": 0},
        ]
        result = score_rows(
            self.references, predictions, "prediction", normalizer=lambda text: text
        )
        np.testing.assert_array_equal(result["exact"], [0, 1])
        np.testing.assert_allclose(result["token_f1"], [2 / 3, 1])

    def test_bootstrap_preserves_pairing(self):
        adapter = np.asarray([1, 0, 1, 0], dtype=float)
        identical_baseline = adapter.copy()
        result = bootstrap(adapter - identical_baseline, samples=100, seed=2026)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual(result, bootstrap(adapter - identical_baseline, 100, 2026))

    def test_valid_run_is_not_rejected_by_its_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.csv"
            predictions.write_text("prediction\nblue\n")
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"independently selected checkpoint")
            record = {
                "test": {"samples": 4, "exact_accuracy": 0.25},
                "predictions_sha256": artifact(predictions)["sha256"],
                "checkpoint_sha256": artifact(checkpoint)["sha256"],
            }
            verify_record(record, predictions, checkpoint)
            verify_scores(record, np.asarray([1.0, 0.0, 0.0, 0.0]))

    def test_nonfinite_metrics_and_invalid_sample_counts_are_rejected(self):
        for accuracy in (float("nan"), float("inf"), -0.1, 1.1, "0.5", True):
            with self.subTest(accuracy=accuracy):
                with self.assertRaises(ValueError):
                    verify_scores(
                        {"test": {"samples": 2, "exact_accuracy": accuracy}},
                        np.array([1, 0]),
                    )
        for samples in (0, -1, 2.0, "2", True):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    verify_scores(
                        {"test": {"samples": samples, "exact_accuracy": 0.5}},
                        np.array([1, 0]),
                    )

    def test_prediction_scoring_does_not_silently_discard_extra_rows(self):
        with self.assertRaisesRegex(ValueError, "same nonzero"):
            score_rows(
                self.references,
                [{"prediction": "red"}],
                "prediction",
                normalizer=lambda text: text,
            )

    def test_record_requires_exact_prediction_and_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.csv"
            predictions.write_text("prediction\nblue\n")
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"original checkpoint")
            record = {
                "test": {"samples": 2, "exact_accuracy": 0.5},
                "predictions_sha256": artifact(predictions)["sha256"],
                "checkpoint_sha256": artifact(checkpoint)["sha256"],
            }
            verify_record(record, predictions)
            verify_record(record, predictions, checkpoint)
            checkpoint.write_bytes(b"different checkpoint with identical metadata")
            with self.assertRaises(ValueError):
                verify_record(record, predictions, checkpoint)
            predictions.write_text("prediction\nred\n")
            with self.assertRaises(ValueError):
                verify_record(record, predictions)

    def test_aggregate_match_does_not_allow_wrong_sample_count(self):
        record = {"test": {"samples": 2, "exact_accuracy": 0.5}}
        verify_scores(record, np.asarray([1.0, 0.0]))
        with self.assertRaises(ValueError):
            verify_scores(record, np.asarray([1.0, 0.0, 1.0, 0.0]))
        with self.assertRaises(ValueError):
            verify_scores(record, np.asarray([1.0, 1.0]))

    def test_author_confirmation_is_bound_to_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "accepted_hash",
                        "controlled_run_membership": {
                            "basis": "author_confirmation",
                            "statement": "One checkpoint from the current controlled runs.",
                            "training_records_independently_audited": False,
                        },
                    }
                )
            )
            value = checkpoint_provenance(path, "accepted_hash")
            self.assertEqual(value["basis"], "author_confirmation")
            self.assertFalse(value["training_records_independently_audited"])
            with self.assertRaises(ValueError):
                checkpoint_provenance(path, "different_hash")

    def test_author_confirmation_cannot_claim_an_independent_training_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "accepted_hash",
                        "controlled_run_membership": {
                            "basis": "author_confirmation",
                            "statement": "One checkpoint from the controlled runs.",
                            "training_records_independently_audited": True,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "independently audit"):
                checkpoint_provenance(path, "accepted_hash")


if __name__ == "__main__":
    unittest.main()
