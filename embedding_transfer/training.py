"""Small training utilities shared by the two released trainers."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class AlignmentLoss(nn.Module):
    """Five-term embedding alignment objective used in the paper."""

    def __init__(
        self,
        token_mse_weight: float = 1.0,
        sentence_weight: float = 0.5,
        contrastive_weight: float = 0.02,
        cosine_weight: float = 0.02,
        greedy_token_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.token_mse_weight = token_mse_weight
        self.sentence_weight = sentence_weight
        self.contrastive_weight = contrastive_weight
        self.cosine_weight = cosine_weight
        self.greedy_token_weight = greedy_token_weight

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    @staticmethod
    def _sentence_variance(
        x: torch.Tensor, pooled: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x.dtype)
        counts = weights.sum(1).clamp(min=1.0)
        return (((x - pooled.unsqueeze(1)) ** 2) * weights).sum(1) / counts

    @staticmethod
    def _contrastive(
        prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        if prediction.size(0) < 2:
            return prediction.new_zeros(())
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = prediction @ target.T / temperature
        labels = torch.arange(prediction.size(0), device=prediction.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    @staticmethod
    def _greedy_token_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        similarity = torch.bmm(
            F.normalize(prediction, dim=-1),
            F.normalize(target, dim=-1).transpose(1, 2),
        )
        valid = prediction_mask.bool().unsqueeze(2) & target_mask.bool().unsqueeze(1)
        similarity = similarity.masked_fill(~valid, torch.finfo(similarity.dtype).min)
        precision = similarity.max(2).values
        recall = similarity.max(1).values
        prediction_weights = prediction_mask.to(similarity.dtype)
        target_weights = target_mask.to(similarity.dtype)
        precision = (precision * prediction_weights).sum(1) / prediction_weights.sum(
            1
        ).clamp(min=1.0)
        recall = (recall * target_weights).sum(1) / target_weights.sum(1).clamp(min=1.0)
        score = 2 * precision * recall / (precision + recall + 1e-8)
        return (1 - score).mean()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Alignment tensors must have equal shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim != 3 or prediction.shape[-1] < 2:
            raise ValueError(
                "Alignment requires batch, sequence, and at least two features"
            )
        for mask in (prediction_mask, target_mask):
            if (
                mask.shape != prediction.shape[:2]
                or not ((mask == 0) | (mask == 1)).all()
                or (mask.sum(1) == 0).any()
            ):
                raise ValueError(
                    "Alignment masks must be binary and contain a valid token in every row"
                )
        # Variances and squared errors can overflow in float16 under autocast.
        prediction, target = prediction.float(), target.float()
        lengths = torch.maximum(prediction_mask.sum(1), target_mask.sum(1))
        lengths = lengths.clamp(min=1, max=prediction.size(1))
        positions = torch.arange(prediction.size(1), device=prediction.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        valid_float = valid.to(prediction.dtype)

        token_error = ((prediction - target) ** 2).mean(-1)
        token_mse = (token_error * valid_float).sum(1) / lengths
        variance_error = (prediction.var(dim=-1) - target.var(dim=-1)) ** 2
        token_variance = (variance_error * valid_float).sum(1) / lengths
        token_loss = token_mse.mean() + 0.5 * token_variance.mean()

        prediction_pooled = self._masked_mean(prediction, prediction_mask)
        target_pooled = self._masked_mean(target, target_mask)
        common_mask = prediction_mask * target_mask
        sentence_mean = F.mse_loss(prediction_pooled, target_pooled)
        sentence_variance = F.mse_loss(
            self._sentence_variance(prediction, prediction_pooled, common_mask),
            self._sentence_variance(target, target_pooled, common_mask),
        )
        sentence_loss = sentence_mean + 0.5 * sentence_variance
        cosine_loss = (1 - F.cosine_similarity(prediction_pooled, target_pooled)).mean()

        return (
            self.token_mse_weight * token_loss
            + self.sentence_weight * sentence_loss
            + self.contrastive_weight
            * self._contrastive(prediction_pooled, target_pooled)
            + self.cosine_weight * cosine_loss
            + self.greedy_token_weight
            * self._greedy_token_loss(prediction, target, prediction_mask, target_mask)
        )


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _text(record: dict, key: str, source: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} requires a nonempty string field named {key!r}")
    return value.strip()


def load_parallel_jsonl(path: str | Path) -> list[dict]:
    result = []
    for index, record in enumerate(_read_jsonl(path), 1):
        source = f"{Path(path).name} record {index}"
        result.append(
            {
                "vi": _text(record, "vi", source),
                "en": _text(record, "en", source),
                **({"id": record["id"]} if "id" in record else {}),
            }
        )
    return result


def load_mocheg_jsonl(path: str | Path, max_evidence: int = 5) -> list[dict]:
    if not 1 <= max_evidence <= 5:
        raise ValueError("max_evidence must lie between one and five")
    result = []
    for index, record in enumerate(_read_jsonl(path), 1):
        source = f"{Path(path).name} record {index}"
        evidence_vi = record.get("evidence_vi")
        evidence_en = record.get("evidence_en")
        if not isinstance(evidence_vi, list) or not isinstance(evidence_en, list):
            raise ValueError(f"{source} requires evidence_vi and evidence_en lists")
        if len(evidence_vi) != len(evidence_en) or not evidence_vi:
            raise ValueError(f"{source} requires equal nonempty evidence lists")
        pairs = []
        for position, (vi, en) in enumerate(
            zip(evidence_vi[:max_evidence], evidence_en[:max_evidence]), 1
        ):
            pair_source = f"{source} evidence {position}"
            pairs.append(
                {
                    "vi": _text({"vi": vi}, "vi", pair_source),
                    "en": _text({"en": en}, "en", pair_source),
                }
            )
        result.append(
            {
                "vi": _text(record, "claim_vi", source),
                "en": _text(record, "claim_en", source),
                "evidence": pairs,
                **({"id": record["id"]} if "id" in record else {}),
            }
        )
    return result


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _fingerprint(vi: str, en: str) -> str:
    value = f"{_normalized(vi)}\0{_normalized(en)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def pair_fingerprints(records: Iterable[dict]) -> set[str]:
    fingerprints = set()
    for record in records:
        fingerprints.add(_fingerprint(record["vi"], record["en"]))
        for pair in record.get("evidence", []):
            fingerprints.add(_fingerprint(pair["vi"], pair["en"]))
    return fingerprints


def require_disjoint(
    train_records: Iterable[dict], validation_records: Iterable[dict]
) -> None:
    def keys(records):
        result = set()
        for record in records:
            if "id" in record:
                result.add(("id", str(record["id"])))
            for pair in (record, *record.get("evidence", [])):
                for language in ("vi", "en"):
                    result.add((language, _normalized(pair[language])))
        return result

    overlap = keys(train_records) & keys(validation_records)
    if overlap:
        raise ValueError(
            f"Training and validation share {len(overlap)} identifier(s) or normalized text(s)."
        )


def _collate(records: list[dict]) -> list[dict]:
    return records


def make_loader(
    records: list[dict], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        generator=generator,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def save_payload(payload: dict, output: str | Path) -> None:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def make_plateau_scheduler(optimizer):
    """Support losses and negated maximization metrics without treating ties as gains."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold_mode="abs",
        threshold=0.0,
    )
