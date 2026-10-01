"""Checkpoint loading shared by the two inference entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from .adapter import EmbeddingTransferAdapter, build_adapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def default_checkpoint(name: str) -> Path:
    return REPOSITORY_ROOT / "checkpoints" / name


def load_payload(path: str | Path, expected_task: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if payload.get("task") != expected_task:
        raise ValueError(
            f"Expected a {expected_task} checkpoint, found {payload.get('task')!r}"
        )
    return payload


def load_adapter(payload: dict, device: torch.device) -> EmbeddingTransferAdapter:
    adapter = build_adapter(payload["adapter_config"])
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.to(device).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter
