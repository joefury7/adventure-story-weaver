"""
Transformer Architecture for Adventure Story Weaver
Implements from scratch:
- Scaled Dot-Product Causal Self-Attention
- Multi-Head Attention projection
- Position-wise Feed-Forward Network
- Pre-LayerNorm Transformer Block with Residual Skip Connections
- AdventureGPT Decoder Model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.embeddings import StoryEmbeddings


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism with scaled dot-product attention
    and lower-triangular causal mask to enforce autoregressive behavior.
    """
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_heads == 0, f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})"
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.block_size = block_size

        # Linear projections for Key, Query, and Value
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular matrix of ones
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, return_attention_weights: bool = False):
        B, T, C = x.shape  # Batch, Time, Channels/Embedding

        # Project and reshape into multi-head dimensions: (B, T, C) -> (B, n_heads, T, head_size)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        # Scaled dot-product attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        # Mask out future positions (strictly autoregressive)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted combination: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ v
        # Concatenate head outputs: (B, nh, T, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))

        if return_attention_weights:
            return out, att
        return out


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with 4x expansion."""
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Block:
    x = x + SelfAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor, return_attention_weights: bool = False):
        if return_attention_weights:
            attn_out, weights = self.attn(self.ln1(x), return_attention_weights=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, weights

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class AdventureGPT(nn.Module):
    """
    Decoder-only Generative Pretrained Transformer for Adventure Story generation.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_embd: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Token + Positional embeddings
        self.embeddings = StoryEmbeddings(vocab_size, n_embd, block_size)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_heads, block_size, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output projection
        self.lm_head.weight = self.embeddings.token_embeddings.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target next-token indices, optional
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy scalar loss, or None if targets is None
        """
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = idx.shape[1]

        x = self.embeddings(idx)  # (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            if targets.shape[1] > self.block_size:
                targets = targets[:, -self.block_size:]
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
