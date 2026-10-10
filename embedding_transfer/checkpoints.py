"""Checkpoint loading shared by the two inference entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from .adapter import EmbeddingTransferAdapter, build_adapter


def load_payload(path: str | Path, expected_task: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if payload.get("task") != expected_task:
        raise ValueError(
            f"Expected a {expected_task} checkpoint, found {payload.get('task')!r}"
        )
    required = ("adapter_config", "adapter_state_dict") + (
        ("stance_state_dict",) if expected_task == "mocheg" else ()
    )
    for key in required:
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise ValueError(f"Checkpoint requires a nonempty dictionary {key!r}")
    config = payload["adapter_config"]
    expected_output = 768 if expected_task == "vivqa" else 512
    if (
        config.get("input_embedding_dim") != 768
        or config.get("output_embedding_dim") != expected_output
    ):
        raise ValueError("Checkpoint dimensions do not match the frozen task interface")
    if not isinstance(config.get("max_seq_length"), int) or not 1 <= config[
        "max_seq_length"
    ] <= (512 if expected_task == "vivqa" else 77):
        raise ValueError("Checkpoint sequence length exceeds the frozen backbone limit")
    return payload


def load_adapter(payload: dict, device: torch.device) -> EmbeddingTransferAdapter:
    adapter = build_adapter(payload["adapter_config"])
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.to(device).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter
