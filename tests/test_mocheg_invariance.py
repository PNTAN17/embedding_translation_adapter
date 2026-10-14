import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from embedding_transfer.stance import TextImageStance


class MOCHEGInvarianceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TextImageStance(hidden_dim=8, image_dim=6).eval()
        self.claim = torch.randn(2, 4, 8)
        self.claim_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        self.evidence = torch.randn(2, 2, 5, 8)
        self.evidence_mask = torch.tensor(
            [[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]]]
        )
        self.evidence_count = torch.tensor([1, 2])
        self.images = torch.randn(2, 2, 3, 6)

    def _logits(self, image_count, selection=slice(None)):
        return self.model(
            self.claim[selection],
            self.claim_mask[selection],
            self.evidence[selection],
            self.evidence_mask[selection],
            self.evidence_count[selection],
            self.images[selection],
            image_count[selection],
        )

    @torch.inference_mode()
    def test_missing_image_claim_keeps_its_head_in_a_mixed_batch(self):
        # Make the two heads predict different labels irrespective of features,
        # so this detects routing errors without depending on numerical noise.
        self.model.classifier_text.weight.zero_()
        self.model.classifier_text.bias.copy_(torch.tensor([3.0, 1.0, 0.0]))
        self.model.classifier.weight.zero_()
        self.model.classifier.bias.copy_(torch.tensor([0.0, 1.0, 3.0]))
        image_count = torch.tensor([0, 1])
        separate = torch.cat(
            [self._logits(image_count, slice(i, i + 1)) for i in range(2)]
        )
        batched = self._logits(image_count)
        torch.testing.assert_close(batched, separate)
        self.assertEqual(batched.argmax(-1).tolist(), [0, 2])

        reversed_order = torch.tensor([1, 0])
        torch.testing.assert_close(
            self._logits(image_count, reversed_order), separate[reversed_order]
        )

    @torch.inference_mode()
    def test_multimodal_batch_matches_separate_claims(self):
        image_count = torch.tensor([1, 2])
        separate = torch.cat(
            [self._logits(image_count, slice(i, i + 1)) for i in range(2)]
        )
        torch.testing.assert_close(self._logits(image_count), separate)

    @torch.inference_mode()
    def test_padding_and_unused_evidence_do_not_change_predictions(self):
        image_count = torch.tensor([0, 1])
        expected = self._logits(image_count)
        self.claim[self.claim_mask == 0] = 1234.0
        self.evidence[self.evidence_mask == 0] = -4321.0
        self.images[0] = 1234.0
        self.images[1, 1] = -4321.0
        actual = self._logits(image_count)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)


class MOCHEGValidationTests(unittest.TestCase):
    def test_output_files_cannot_overwrite_any_training_input(self):
        from train_mocheg import validate_output_paths

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "adapter.pt"
            for destination in (
                output,
                output.with_suffix(".pt.tmp"),
                output.with_suffix(".run.json"),
                output.with_suffix(".run.json.tmp"),
            ):
                destination.write_text("preserve input", encoding="utf-8")
                for name in (
                    "train_pairs",
                    "validation_pairs",
                    "validation_task_data",
                    "phomt_train",
                    "config",
                    "stance_checkpoint",
                ):
                    args = argparse.Namespace(output=output, **{name: destination})
                    with self.subTest(destination=destination.name, input=name):
                        with self.assertRaisesRegex(ValueError, f"overwrite {name}"):
                            validate_output_paths(args)
                        self.assertEqual(destination.read_text(), "preserve input")

    def test_output_alias_cannot_overwrite_stance_checkpoint(self):
        from train_mocheg import validate_output_paths

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stance.pt"
            source.write_text("preserve stance", encoding="utf-8")
            alias = Path(directory) / "alias.pt"
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "overwrite stance_checkpoint"):
                validate_output_paths(
                    argparse.Namespace(output=alias, stance_checkpoint=source)
                )
            self.assertEqual(source.read_text(), "preserve stance")

    def test_task_selection_rejects_a_partial_development_partition(self):
        import train_mocheg

        args = argparse.Namespace(
            patience=5,
            seed=42,
            device="cpu",
            train_pairs=Path("train.jsonl"),
            validation_pairs=Path("dev.jsonl"),
            phomt_train=None,
            validation_task_data=Path("dev_task.jsonl"),
            max_evidence=5,
            selection_metric="micro_macro_f1_sum",
            output=Path("outputs/mocheg.pt"),
        )
        training = [{"id": "train", "vi": "huấn luyện", "en": "training"}]
        development = [
            {"id": "dev1", "vi": "một", "en": "one", "evidence": []},
            {"id": "dev2", "vi": "hai", "en": "two", "evidence": []},
        ]
        partial_task = [
            {"id": "dev1", "claim_vi": "một", "evidence_vi": [], "label": "NEI"}
        ]
        with (
            patch.object(train_mocheg, "parse_config", return_value=args),
            patch.object(train_mocheg, "validate_training_args"),
            patch.object(
                train_mocheg, "load_mocheg_jsonl", side_effect=[training, development]
            ),
            patch.object(train_mocheg, "read_records", return_value=partial_task),
            self.assertRaisesRegex(ValueError, "requires every validation pair ID"),
        ):
            train_mocheg.main()


if __name__ == "__main__":
    unittest.main()
