"""Inference for Vietnamese visual question answering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    BertModel,
    BertTokenizer,
    BlipForQuestionAnswering,
    BlipProcessor,
)

from .checkpoints import default_checkpoint, load_adapter, load_payload


class ViVQAPredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.device = choose_device(device)
        payload = load_payload(checkpoint, "vivqa")
        self.max_length = payload["adapter_config"]["max_seq_length"]
        self.adapter = load_adapter(payload, self.device)
        del payload

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base",
            local_files_only=local_files_only,
            use_fast=False,
        )
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", local_files_only=local_files_only
        ).to(self.device).eval()
        mbert = BertModel.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=local_files_only
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=local_files_only
        )
        self.source_embeddings = mbert.embeddings.to(self.device).eval()
        for module in (self.model, self.source_embeddings):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        del mbert

    @torch.inference_mode()
    def predict(self, image_path: str | Path, question: str) -> str:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            pixel_values = self.processor.image_processor(
                images=image, return_tensors="pt"
            )["pixel_values"].to(self.device)

        image_embeddings = self.model.vision_model(
            pixel_values=pixel_values, return_dict=True
        ).last_hidden_state
        image_mask = torch.ones(
            image_embeddings.shape[:-1], dtype=torch.long, device=self.device
        )
        source = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        source_embeddings = self.source_embeddings(source.input_ids).float()
        adapted = self.adapter(source_embeddings, source.attention_mask)

        norms = adapted.norm(dim=-1)
        threshold = norms[source.attention_mask.bool()].mean() * 0.1
        attention_mask = (norms > threshold).long() | source.attention_mask
        question_output = self.model.text_encoder(
            input_ids=None,
            inputs_embeds=adapted,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_mask,
            return_dict=True,
        )
        bos = torch.full(
            (1, 1),
            self.model.text_decoder.config.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        output = self.model.text_decoder.generate(
            input_ids=bos,
            max_length=15,
            min_length=2,
            num_beams=1,
            eos_token_id=self.processor.tokenizer.sep_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        return self.processor.decode(output[0], skip_special_tokens=True).strip()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint("vivqa_adapter.pt"),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = ViVQAPredictor(
        args.checkpoint, args.device, local_files_only=args.local_files_only
    )
    answer = predictor.predict(args.image, args.question)
    print(json.dumps({"answer": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
