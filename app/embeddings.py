"""
Embeddings Module for Adventure Story Weaver
Combines token embeddings with learned positional embeddings.
"""

import torch
import torch.nn as nn


class StoryEmbeddings(nn.Module):
    """
    Transforms discrete token IDs into dense continuous representations
    augmented with positional encodings.
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of shape (B, T) with values in [0, vocab_size - 1]
        Returns:
            Tensor of shape (B, T, n_embd)
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds maximum block size {self.block_size}")

        tok_emb = self.token_embeddings(idx)  # (B, T, n_embd)
        positions = torch.arange(T, device=idx.device)  # (T)
        pos_emb = self.position_embeddings(positions)  # (T, n_embd)

        return tok_emb + pos_emb
