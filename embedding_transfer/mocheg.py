"""Inference for Vietnamese multimodal claim verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, CLIPVisionModel

from .checkpoints import load_adapter, load_payload
from .stance import TextImageStance
from .vivqa import choose_device


LABELS = ("refuted", "supported", "NEI")


class MOCHEGPredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.device = choose_device(device)
        self.local_files_only = local_files_only
        payload = load_payload(checkpoint, "mocheg")
        self.max_length = payload["adapter_config"]["max_seq_length"]
        self.max_evidence = payload.get("metadata", {}).get("max_evidence", 5)
        if not isinstance(self.max_evidence, int) or not 1 <= self.max_evidence <= 5:
            raise ValueError("Checkpoint max_evidence must lie between one and five")
        self.adapter = load_adapter(payload, self.device)
        self.stance = TextImageStance().to(self.device).eval()
        self.stance.load_state_dict(payload["stance_state_dict"], strict=True)
        for parameter in self.stance.parameters():
            parameter.requires_grad_(False)
        del payload

        mbert = BertModel.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=local_files_only
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=local_files_only
        )
        self.source_word_embeddings = mbert.embeddings.word_embeddings.to(
            self.device
        ).eval()
        for parameter in self.source_word_embeddings.parameters():
            parameter.requires_grad_(False)
        del mbert
        self.vision_model = None
        self.image_processor = None

    @classmethod
    def from_modules(
        cls,
        adapter,
        stance_state,
        source_embeddings,
        tokenizer,
        device,
        local_files_only=False,
        max_evidence=5,
    ):
        instance = cls.__new__(cls)
        instance.device = device
        instance.local_files_only = local_files_only
        instance.max_length = adapter.max_seq_length
        instance.max_evidence = max_evidence
        instance.adapter = adapter
        instance.stance = TextImageStance().to(device).eval()
        instance.stance.load_state_dict(stance_state, strict=True)
        for parameter in instance.stance.parameters():
            parameter.requires_grad_(False)
        instance.source_word_embeddings = source_embeddings
        instance.tokenizer = tokenizer
        instance.vision_model = instance.image_processor = None
        return instance

    def _encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        source = self.source_word_embeddings(tokens.input_ids)
        adapted = self.adapter(source, tokens.attention_mask)
        return adapted, tokens.attention_mask

    def _load_vision(self) -> None:
        if self.vision_model is not None:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=self.local_files_only,
        )
        self.vision_model = (
            CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=self.local_files_only,
            )
            .to(self.device)
            .eval()
        )
        for parameter in self.vision_model.parameters():
            parameter.requires_grad_(False)

    def _encode_images(self, paths: list[str | Path]) -> torch.Tensor:
        self._load_vision()
        images = []
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        pixels = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        return self.vision_model(pixel_values=pixels).last_hidden_state

    @torch.inference_mode()
    def predict(
        self,
        claim: str,
        evidence: list[str],
        images: list[str | Path] | None = None,
    ) -> dict:
        if not claim.strip():
            raise ValueError("Claim must not be empty.")
        evidence = [item for item in evidence if item.strip()][: self.max_evidence]
        if not evidence:
            raise ValueError("At least one nonempty textual evidence item is required.")
        images = list(images or [])[:5]

        claim_embedding, claim_mask = self._encode_text([claim])
        evidence_embedding, evidence_mask = self._encode_text(evidence)
        evidence_embedding = evidence_embedding.unsqueeze(0)
        evidence_mask = evidence_mask.unsqueeze(0)
        evidence_count = torch.tensor(
            [len(evidence)], dtype=torch.long, device=self.device
        )

        image_embedding = image_count = None
        if images:
            image_embedding = self._encode_images(images).unsqueeze(0)
            image_count = torch.tensor(
                [len(images)], dtype=torch.long, device=self.device
            )
        logits = self.stance(
            claim_embedding,
            claim_mask,
            evidence_embedding,
            evidence_mask,
            evidence_count,
            image_embedding,
            image_count,
        )
        probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
        label = LABELS[int(logits.argmax(dim=-1).item())]
        return {
            "label": label,
            "probabilities": dict(zip(LABELS, probabilities)),
            "text_evidence_used": len(evidence),
            "images_used": len(images),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        action="append",
        help="Vietnamese textual evidence. Repeat for multiple items.",
    )
    parser.add_argument(
        "--image", action="append", default=[], type=Path, help="Repeat as needed."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MOCHEG checkpoint path.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = MOCHEGPredictor(
        args.checkpoint, args.device, local_files_only=args.local_files_only
    )
    result = predictor.predict(args.claim, args.evidence, args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
