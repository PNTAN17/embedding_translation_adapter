"""Frozen MOCHEG stance head used by the released checkpoint."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextImageStance(nn.Module):
    def __init__(self, hidden_dim: int = 512, image_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.txt_attn = nn.MultiheadAttention(
            hidden_dim, 1, kdim=hidden_dim, vdim=hidden_dim, batch_first=True
        )
        self.linear_fuse_p = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier_text = nn.Linear(hidden_dim, 3)
        self.img_attn = nn.MultiheadAttention(
            hidden_dim, 1, kdim=image_dim, vdim=image_dim, batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, 3)

    def _text_stance(
        self,
        claim: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_count: torch.Tensor,
    ) -> torch.Tensor:
        batch, claim_length, hidden = claim.shape
        slots, evidence_length = evidence.size(1), evidence.size(2)
        repeated_claim = claim[:, None].expand(-1, slots, -1, -1).reshape(
            batch * slots, claim_length, hidden
        )
        repeated_mask = claim_mask[:, None].expand(-1, slots, -1).reshape(
            batch * slots, claim_length
        )
        flat_evidence = evidence.reshape(batch * slots, evidence_length, hidden)
        key_padding_mask = evidence_mask.reshape(batch * slots, evidence_length) == 0
        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask[all_padding, 0] = False
        matched, _ = self.txt_attn(
            repeated_claim,
            flat_evidence,
            flat_evidence,
            key_padding_mask=key_padding_mask,
        )
        fused = F.leaky_relu(
            self.linear_fuse_p(
                torch.cat([matched - repeated_claim, matched * repeated_claim], dim=-1)
            )
        )
        fused = fused.masked_fill((repeated_mask == 0).unsqueeze(-1), -1e4)
        pooled = fused.max(dim=1).values.reshape(batch, slots, hidden)
        slot_ids = torch.arange(slots, device=claim.device).unsqueeze(0)
        valid_slots = (slot_ids < evidence_count.unsqueeze(1)).to(pooled.dtype)
        pooled = pooled * valid_slots.unsqueeze(-1)
        return pooled.sum(dim=1) / evidence_count.clamp(min=1).unsqueeze(1)

    def _image_stance(
        self,
        claim: torch.Tensor,
        claim_mask: torch.Tensor,
        images: torch.Tensor,
        image_count: torch.Tensor,
    ) -> torch.Tensor:
        batch, claim_length, hidden = claim.shape
        slots, patch_count, image_dim = images.shape[1:]
        repeated_claim = claim[:, None].expand(-1, slots, -1, -1).reshape(
            batch * slots, claim_length, hidden
        )
        repeated_mask = claim_mask[:, None].expand(-1, slots, -1).reshape(
            batch * slots, claim_length
        )
        flat_images = images.reshape(batch * slots, patch_count, image_dim)
        matched, _ = self.img_attn(repeated_claim, flat_images, flat_images)
        fused = F.leaky_relu(
            self.linear_fuse_p(
                torch.cat([matched - repeated_claim, matched * repeated_claim], dim=-1)
            )
        )
        fused = fused.masked_fill((repeated_mask == 0).unsqueeze(-1), -1e4)
        pooled = fused.max(dim=1).values.reshape(batch, slots, hidden)
        slot_ids = torch.arange(slots, device=claim.device).unsqueeze(0)
        valid_slots = (slot_ids < image_count.unsqueeze(1)).to(pooled.dtype)
        pooled = pooled * valid_slots.unsqueeze(-1)
        return pooled.sum(dim=1) / image_count.clamp(min=1).unsqueeze(1)

    def forward(
        self,
        claim: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_count: torch.Tensor,
        images: torch.Tensor | None = None,
        image_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text = self._text_stance(
            claim, claim_mask, evidence, evidence_mask, evidence_count
        )
        if images is None or image_count is None or bool((image_count == 0).all()):
            return self.classifier_text(text)
        image = self._image_stance(claim, claim_mask, images, image_count)
        return self.classifier(torch.cat([text, image], dim=-1))
