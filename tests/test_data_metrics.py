import argparse
import json
from pathlib import Path
import tempfile
import unittest

from embedding_transfer.data import prepare_partitions
from embedding_transfer.metrics import normalize_answer, score_records, token_f1
from embedding_transfer.provenance import parse_config, save_history


class DataMetricTests(unittest.TestCase):
    def test_distinct_translations_survive_training_deduplication(self):
        train = [
            {"id": "t1", "vi": "câu hỏi", "en": "question"},
            {"id": "t2", "vi": "câu hỏi", "en": "another translation"},
            {"id": "t3", "vi": "CÂU HỎI", "en": "question"},
        ]
        clean, removed = prepare_partitions(train, [], [], "parallel")
        self.assertEqual([row["id"] for row in clean], ["t1", "t2"])
        self.assertEqual(removed, [{"id": "t3", "reason": "duplicate_training_record"}])

    def test_shared_training_evidence_preserves_distinct_claims(self):
        train = [
            {
                "id": "t1",
                "claim_vi": "một",
                "claim_en": "one",
                "evidence_vi": ["chung"],
                "evidence_en": ["shared"],
            },
            {
                "id": "t2",
                "claim_vi": "hai",
                "claim_en": "two",
                "evidence_vi": ["chung"],
                "evidence_en": ["shared"],
            },
        ]
        clean, removed = prepare_partitions(train, [], [], "mocheg")
        self.assertEqual(len(clean), 2)
        self.assertEqual(removed, [])

    def test_training_text_leakage_detected_across_different_translations(self):
        train = [
            {"id": "t1", "vi": " CÂU   HỎI ", "en": "alternate translation"},
            {"id": "t2", "vi": "giữ", "en": "keep"},
        ]
        validation = [{"id": "v1", "vi": "câu hỏi", "en": "question"}]
        test = [{"id": "x1", "vi": "khác", "en": "other"}]
        clean, removed = prepare_partitions(train, validation, test, "parallel")
        self.assertEqual([row["id"] for row in clean], ["t2"])
        self.assertEqual(removed, [{"id": "t1", "reason": "held_out_text_overlap"}])
        self.assertEqual(len(validation), 1)

    def test_original_id_collision_is_an_error(self):
        first = [{"id": 1, "vi": "a", "en": "b"}]
        second = [{"id": "1", "vi": "c", "en": "d"}]
        with self.assertRaises(ValueError):
            prepare_partitions(first, second, [], "parallel")

    def test_normalization_and_multiset_f1(self):
        self.assertEqual(
            normalize_answer(
                "The 2 RED cars!", lambda text: text.replace("cars", "car")
            ),
            "two red car",
        )
        self.assertEqual(normalize_answer("01 21", lambda text: text), "01 21")
        self.assertAlmostEqual(token_f1("red red car", "red car"), 0.8)
        self.assertEqual(token_f1("", ""), 1)
        self.assertEqual(token_f1("", "red"), 0)

    def test_vivqa_scoring_joins_by_id(self):
        references = [
            {"id": "a", "answer_en": "red car"},
            {"id": "b", "answer_en": "two"},
        ]
        predictions = [
            {"id": "b", "prediction": "two"},
            {"id": "a", "prediction": "red"},
        ]
        result = score_records(
            references, predictions, "vivqa", normalizer=lambda value: value
        )
        self.assertEqual(result["exact_match"], 50)
        self.assertEqual(result["exact_match_correct"], 1)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["token_f1"], 100 * (1 + 2 / 3) / 2)
        with self.assertRaises(ValueError):
            score_records(references, predictions[:1], "vivqa")
        with self.assertRaises(ValueError):
            score_records(references, predictions + predictions[:1], "vivqa")

    def test_mocheg_fixed_label_macro_average(self):
        references = [
            {"id": "a", "label": "supported"},
            {"id": "b", "label": "refuted"},
        ]
        predictions = [
            {"id": "a", "label": "supported"},
            {"id": "b", "label": "supported"},
        ]
        result = score_records(references, predictions, "mocheg")
        self.assertEqual(result["micro_f1"], 50)
        self.assertEqual(result["micro_f1_correct"], 1)
        self.assertAlmostEqual(result["macro_f1"], 100 * (2 / 3) / 3)

    def test_config_overrides_and_complete_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps({"seed": 42, "input": "data.jsonl"}))
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", type=Path, required=True)
            parser.add_argument("--seed", type=int)
            args = parse_config(parser, ["--config", str(path), "--seed", "44"])
            self.assertEqual(args.seed, 44)
            self.assertEqual(args.input, Path("data.jsonl"))
            history = [{"epoch": 1, "score": 2}, {"epoch": 2, "score": 1}]
            save_history(root / "best.pt", {}, history, 1)
            result = json.loads((root / "best.run.json").read_text())
            self.assertEqual(result["selected_epoch"], 1)
            self.assertEqual(len(result["history"]), 2)

    def test_config_rejects_silent_integer_conversion_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in (
                ("epochs", 30.9),
                ("epochs", True),
                ("learning_rate", "nan"),
            ):
                with self.subTest(key=key, value=value):
                    path.write_text(json.dumps({key: value}))
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--epochs", type=int)
                    parser.add_argument("--learning-rate", type=float)
                    with self.assertRaises(ValueError):
                        parse_config(parser, ["--config", str(path)])


if __name__ == "__main__":
    unittest.main()
