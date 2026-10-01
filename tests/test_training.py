import json
import tempfile
import unittest
from pathlib import Path

import torch

from embedding_transfer.training import (
    AlignmentLoss,
    load_mocheg_jsonl,
    load_parallel_jsonl,
    require_disjoint,
)


class TrainingUtilityTests(unittest.TestCase):
    def test_alignment_loss_is_finite(self):
        criterion = AlignmentLoss()
        prediction = torch.randn(3, 5, 8, requires_grad=True)
        target = torch.randn(3, 5, 8)
        prediction_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
        )
        target_mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]
        )
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


if __name__ == "__main__":
    unittest.main()
