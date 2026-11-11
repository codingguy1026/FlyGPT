"""Tiny causal Transformer initialized from random weights.

Experimental next-character language modeling, not a pretrained model and not
a trained connectome. All weight loading is explicit and local.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from papiti_lm import MODEL_NAME
from papiti_lm.corpus import ASSISTANT, BOS, EOS, PAD, UNK, USER, CharTokenizer, JamoTokenizer, SubwordTokenizer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 192
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        if not 6 < self.vocab_size < 65536:
            raise ValueError("invalid vocabulary size")
        if not 32 <= self.context <= 1024:
            raise ValueError("context must be 32..1024")
        if not 32 <= self.d_model <= 512 or self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads and 32..512")
        if not 1 <= self.heads <= 16 or not 1 <= self.layers <= 8:
            raise ValueError("heads or layers outside experimental limits")
        if not 0 <= self.dropout < 1:
            raise ValueError("invalid dropout")


class PapitiTransformer(nn.Module):
    """Token/position embeddings, causal self-attention and next-token head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.normalizer = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # All parameters (including embeddings) start randomly initialized.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        batch, length = input_ids.shape
        if length > self.config.context or length < 1:
            raise ValueError("input longer than configured context")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=input_ids.device
        ).triu(1)
        padding = input_ids.eq(PAD)
        hidden = self.transformer(
            hidden, mask=causal, src_key_padding_mask=padding
        )
        return self.head(self.normalizer(hidden))

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )


@torch.inference_mode()
def generate(
    model: PapitiTransformer,
    tokenizer: CharTokenizer,
    user: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_k: int = 24,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> str:
    """Autoregressively sample a new answer; no canned answers or lookup."""
    if not user.strip() or len(user) > 2048:
        raise ValueError("user input must be 1..2048 characters")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be 1..256")
    if not 0 <= temperature <= 2.0:
        raise ValueError("temperature must be 0..2")
    if not 1 <= top_k <= 65536:
        raise ValueError("top_k must be positive and bounded")
    if not 1 <= repetition_penalty <= 2:
        raise ValueError("repetition_penalty must be between 1 and 2")
    # Small demo vocabularies may contain fewer items than the default top_k.
    # The actual sampling set is clamped to logits.numel() below.
    device = next(model.parameters()).device
    # Preserve role delimiters, trimming only oldest user characters.
    message = tokenizer.encode(user)
    message = message[-(model.config.context - 3):]
    ids = [BOS, USER, *message, ASSISTANT]
    produced: list[int] = []
    rng = None
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    model.eval()
    for _ in range(max_new_tokens):
        window = torch.tensor(
            [ids[-model.config.context:]], dtype=torch.long, device=device
        )
        logits = model(window)[0, -1].clone()
        logits[[PAD, BOS, USER, ASSISTANT, UNK]] = -float("inf")
        # Optional repetition penalty acts on generated output only. Never
        # suppress characters because they appeared in the user's prompt.
        if repetition_penalty > 1.0:
            for token in set(produced[-32:]):
                if logits[token] >= 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty
        if temperature == 0:
            next_token = int(logits.argmax())
        else:
            logits /= max(temperature, 1e-6)
            limit = min(top_k, logits.numel())
            top_values, top_ids = torch.topk(logits, limit)
            probabilities = F.softmax(top_values, dim=-1)
            picked = torch.multinomial(probabilities, 1, generator=rng)
            next_token = int(top_ids[picked].item())
        if next_token == EOS:
            break
        produced.append(next_token)
        ids.append(next_token)
    return tokenizer.decode(produced)


def save_checkpoint(
    path: str | Path,
    model: PapitiTransformer,
    tokenizer: CharTokenizer,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({
            "format_version": 1,
            "model_name": MODEL_NAME,
            "config": asdict(model.config),
            "characters": tokenizer.characters,
            "tokenizer_type": (
                "bpe" if isinstance(tokenizer, SubwordTokenizer)
                else "jamo" if isinstance(tokenizer, JamoTokenizer) else "char"
            ),
            "bpe_merges": (
                [list(pair) for pair in tokenizer.merges]
                if isinstance(tokenizer, SubwordTokenizer) else []
            ),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        }, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Only load a locally created, trusted checkpoint via weights_only mode."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint")
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("checkpoint belongs to a different model")
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        tokenizer = SubwordTokenizer(
            checkpoint["characters"], checkpoint.get("bpe_merges", [])
        )
    elif tokenizer_type == "jamo":
        tokenizer = JamoTokenizer(checkpoint["characters"])
    elif tokenizer_type == "char":
        tokenizer = CharTokenizer(checkpoint["characters"])
    else:
        raise ValueError("unsupported tokenizer type")
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary/config mismatch")
    model = PapitiTransformer(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint.get("metadata", {})
