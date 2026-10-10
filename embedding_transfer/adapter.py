"""Source-conditioned dual-branch embedding adapter used by both tasks."""

from __future__ import annotations

import torch
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.self_attn(
            normalized, normalized, normalized, key_padding_mask=padding_mask
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        source_padding_mask: torch.Tensor | None,
        target_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(target)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
        )
        target = target + attended
        attended, _ = self.cross_attn(
            self.norm2(target),
            memory,
            memory,
            key_padding_mask=source_padding_mask,
        )
        target = target + attended
        return target + self.mlp(self.norm3(target))


class TokenAlignment(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.alignment_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        aligned, _ = self.alignment_attn(
            tokens, context, context, key_padding_mask=padding_mask
        )
        gate = self.gate(torch.cat([tokens, aligned], dim=-1))
        return self.norm(tokens * (1.0 - gate) + aligned * gate)


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.gate = nn.Linear(dim, dim * 2)
        self.output = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.linear(x) * torch.sigmoid(self.gate(x))
        return self.norm(x + self.output(gated))


class EmbeddingTransferAdapter(nn.Module):
    """Map mBERT token embeddings into a frozen target model's text space.

    Attribute names intentionally match the training implementation so that
    the released state dictionaries can be loaded with ``strict=True``.
    """

    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        hidden_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        dropout: float,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.input_projection = nn.Linear(input_embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_embedding_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_norm = nn.LayerNorm(output_embedding_dim, eps=1e-6)
        self.residual_gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.contrastive_head = nn.Sequential(
            nn.Linear(output_embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.pos_bias = nn.Parameter(torch.zeros(1, max_seq_length, hidden_dim))
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.token_alignment = TokenAlignment(hidden_dim, num_heads // 2, dropout)
        self.glu = GatedLinearUnit(hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.translation_head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_branch = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.branch_gate = nn.Linear(hidden_dim * 2, 1)
        gate_hidden_dim = max(hidden_dim // 2, 128)
        self.branch_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4, eps=1e-6),
            nn.Linear(hidden_dim * 4, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.gate_temperature = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _encode(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_projection(embeddings))
        x = self.dropout(x + self.pos_bias[:, : x.size(1)])
        padding_mask = mask == 0
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        return self.glu(self.encoder_norm(x))

    def _decode(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding_mask = mask == 0
        target = self.token_alignment(encoded, encoded, padding_mask)
        length = target.size(1)
        mask_value = -1e4 if target.dtype == torch.float16 else float("-inf")
        causal_mask = torch.triu(
            torch.full((length, length), mask_value, device=target.device), diagonal=1
        )
        target_padding_mask = torch.zeros(
            mask.shape, dtype=target.dtype, device=target.device
        ).masked_fill(padding_mask, float("-inf"))
        for layer in self.decoder_layers:
            target = layer(
                target, encoded, causal_mask, padding_mask, target_padding_mask
            )
        return self.decoder_norm(target)

    def translate_embeddings(
        self, source_embeddings: torch.Tensor, source_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if source_embeddings.size(1) > self.max_seq_length:
            raise ValueError(
                f"Sequence length {source_embeddings.size(1)} exceeds "
                f"checkpoint limit {self.max_seq_length}."
            )
        identity = self.input_projection(source_embeddings)
        projected_identity = self.input_norm(identity)
        token_correction = self.token_branch(projected_identity)

        encoded = self._encode(source_embeddings, source_attention_mask)
        decoded = self._decode(encoded, source_attention_mask)
        context_correction = self.translation_head(decoded)

        gate_logits = self.branch_gate(
            torch.cat([token_correction, context_correction], dim=-1)
        )
        global_context = self._masked_mean(encoded, source_attention_mask)
        global_context = global_context.unsqueeze(1).expand(-1, identity.size(1), -1)
        gate_input = torch.cat(
            [projected_identity, token_correction, context_correction, global_context],
            dim=-1,
        )
        temperature = self.gate_temperature.exp().clamp(min=0.1, max=10.0)
        gate_logits = (gate_logits + self.branch_gate_mlp(gate_input)) / temperature
        gate = torch.sigmoid(gate_logits)
        correction = gate * context_correction + (1.0 - gate) * token_correction
        return self.output_norm(
            self.output_projection(identity + self.residual_gate * correction)
        )

    def forward(
        self, source_embeddings: torch.Tensor, source_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.translate_embeddings(source_embeddings, source_attention_mask)


def build_adapter(config: dict) -> EmbeddingTransferAdapter:
    required = {
        "input_embedding_dim",
        "output_embedding_dim",
        "hidden_dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "dropout",
        "max_seq_length",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Checkpoint is missing adapter settings: {sorted(missing)}")
    return EmbeddingTransferAdapter(**{key: config[key] for key in required})
