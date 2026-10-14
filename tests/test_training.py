import json
import tempfile
import unittest
from pathlib import Path

import torch

from embedding_transfer.training import (
    AlignmentLoss,
    load_mocheg_jsonl,
    load_parallel_jsonl,
    make_plateau_scheduler,
    require_disjoint,
)


class TrainingUtilityTests(unittest.TestCase):
    def test_plateau_scheduler_reduces_lr_when_negative_task_metric_ties(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = make_plateau_scheduler(optimizer)
        for _ in range(4):
            scheduler.step(-100.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_half_precision_squared_error_remains_finite(self):
        prediction = torch.full(
            (2, 3, 4), 500.0, dtype=torch.float16, requires_grad=True
        )
        target = torch.ones_like(prediction)
        mask = torch.ones((2, 3), dtype=torch.long)
        loss = AlignmentLoss()(prediction, target, mask, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_empty_attention_row_fails_explicitly(self):
        with self.assertRaises(ValueError):
            AlignmentLoss()(
                torch.randn(1, 3, 4),
                torch.randn(1, 3, 4),
                torch.zeros(1, 3),
                torch.ones(1, 3),
            )

    def test_overlap_in_one_language_is_not_hidden_by_new_translation(self):
        with self.assertRaises(ValueError):
            require_disjoint(
                [{"id": "t", "vi": " câu hỏi", "en": "first"}],
                [{"id": "v", "vi": "CÂU HỎI", "en": "second"}],
            )

    def test_alignment_loss_is_finite(self):
        criterion = AlignmentLoss()
        prediction = torch.randn(3, 5, 8, requires_grad=True)
        target = torch.randn(3, 5, 8)
        prediction_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
        )
        target_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        loss = criterion(prediction, target, prediction_mask, target_mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_jsonl_loading_and_separation_check(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_path = directory / "train.jsonl"
            validation_path = directory / "validation.jsonl"
            train_path.write_text(
                json.dumps({"vi": "xin chào", "en": "hello"}) + "\n",
                encoding="utf-8",
            )
            validation_path.write_text(
                json.dumps({"vi": "tạm biệt", "en": "goodbye"}) + "\n",
                encoding="utf-8",
            )
            train = load_parallel_jsonl(train_path)
            validation = load_parallel_jsonl(validation_path)
            require_disjoint(train, validation)
            with self.assertRaises(ValueError):
                require_disjoint(train, train)

    def test_mocheg_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mocheg.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "claim_vi": "tuyên bố",
                        "claim_en": "claim",
                        "evidence_vi": ["bằng chứng"],
                        "evidence_en": ["evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_mocheg_jsonl(path)
            self.assertEqual(records[0]["evidence"][0]["en"], "evidence")

    def test_mocheg_evidence_limit_applies_after_empty_pair_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            record = {
                "claim_vi": "tuyên bố",
                "claim_en": "claim",
                "evidence_vi": [" ", "bằng chứng một", "bằng chứng hai"],
                "evidence_en": ["", "first evidence", "second evidence"],
            }
            path.write_text(json.dumps(record) + "\n")
            self.assertEqual(
                load_mocheg_jsonl(path, 1)[0]["evidence"],
                [{"vi": "bằng chứng một", "en": "first evidence"}],
            )
            record["evidence_en"][0] = "unpaired translation"
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                load_mocheg_jsonl(path, 1)


if __name__ == "__main__":
    unittest.main()
