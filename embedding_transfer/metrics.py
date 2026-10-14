"""Evaluation metrics and utilities."""

from collections import Counter
import re

from .data import indexed_records


NUMBER_WORDS = {
    str(index): word
    for index, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
LABELS = ("refuted", "supported", "NEI")
NORMALIZATION = "english_articles_punctuation_pos_lemma_digits_0_to_20_v1"


def lemmatize(text: str) -> str:
    from nltk import pos_tag, word_tokenize
    from nltk.stem import WordNetLemmatizer

    try:
        tagged = pos_tag(word_tokenize(text))
        lemma = WordNetLemmatizer()
        return " ".join(
            lemma.lemmatize(
                token, {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[0], "n")
            )
            for token, tag in tagged
        )
    except LookupError as error:
        raise RuntimeError(
            "Install the documented NLTK resources before scoring. Normalization never silently falls back to a different metric."
        ) from error


def normalize_answer(text: str, lemmatizer=None) -> str:
    if not isinstance(text, str):
        raise ValueError("Predictions and references must be strings")
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = (lemmatizer or lemmatize)(text)
    return " ".join(NUMBER_WORDS.get(token, token) for token in text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = prediction.split(), reference.split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def score_records(
    references: list[dict],
    predictions: list[dict],
    task: str,
    normalizer=normalize_answer,
) -> dict:
    expected, actual = indexed_records(references), indexed_records(predictions)
    if expected.keys() != actual.keys():
        raise ValueError(
            f"Prediction IDs do not match references; missing={len(expected.keys() - actual.keys())}, unexpected={len(actual.keys() - expected.keys())}"
        )
    count = len(expected)
    if not count:
        raise ValueError("Cannot score an empty split")
    if task == "vivqa":
        pairs = [
            (normalizer(actual[key]["prediction"]), normalizer(row["answer_en"]))
            for key, row in expected.items()
        ]
        return {
            "count": count,
            "exact_match_correct": sum(pred == ref for pred, ref in pairs),
            "unit": "percent",
            "normalization": NORMALIZATION,
            "exact_match": 100 * sum(pred == ref for pred, ref in pairs) / count,
            "token_f1": 100 * sum(token_f1(pred, ref) for pred, ref in pairs) / count,
        }
    if task != "mocheg":
        raise ValueError(f"Unknown task {task!r}")
    pairs = [(actual[key]["label"], row["label"]) for key, row in expected.items()]
    if any(pred not in LABELS or ref not in LABELS for pred, ref in pairs):
        raise ValueError(f"MOCHEG labels must use {LABELS}")
    per_class = {}
    for label in LABELS:
        tp = sum(pred == ref == label for pred, ref in pairs)
        fp = sum(pred == label and ref != label for pred, ref in pairs)
        fn = sum(pred != label and ref == label for pred, ref in pairs)
        per_class[label] = (
            100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        )
    return {
        "count": count,
        "micro_f1_correct": sum(pred == ref for pred, ref in pairs),
        "unit": "percent",
        "micro_f1": 100 * sum(pred == ref for pred, ref in pairs) / count,
        "macro_f1": sum(per_class.values()) / len(LABELS),
        "per_class_f1": per_class,
    }


def predict_records(
    records: list[dict], predictor, task: str, image_root="."
) -> list[dict]:
    from pathlib import Path

    indexed_records(records)
    output = []
    for row in records:
        if task == "vivqa":
            result = {
                "prediction": predictor.predict(
                    Path(image_root) / row["image"], row["question_vi"]
                )
            }
        else:
            result = predictor.predict(
                row["claim_vi"],
                row["evidence_vi"],
                [Path(image_root) / item for item in row.get("images", [])],
            )
        output.append({"id": row["id"], **result})
    return output
