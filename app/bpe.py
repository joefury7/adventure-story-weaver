"""
Byte-Pair Encoding (BPE) Tokenizer for Adventure Story Weaver
Implements BPE tokenization from scratch:
- Iterative pair-frequency counting and merging
- Invertible encode / decode mappings
- Serialization to dictionary / JSON
"""

import json
from collections import Counter
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self):
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.is_trained: bool = False

    def train(self, text: str, num_merges: int = 50, special_tokens: List[str] | None = None) -> "BPETokenizer":
        """
        Train BPE on text by iteratively merging the most frequent adjacent token pair.
        """
        if not text:
            raise ValueError("Training text cannot be empty.")

        special = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        
        # 1. Start with initial character-level tokens
        # Split text into words or character sequences
        chars = sorted(list(set(text)))
        
        # Initialize vocabulary with special tokens + unique characters
        self.vocab = {}
        current_id = 0
        for tok in special:
            self.vocab[tok] = current_id
            current_id += 1
        for c in chars:
            if c not in self.vocab:
                self.vocab[c] = current_id
                current_id += 1

        # Represent text as list of individual characters
        tokens = list(text)

        # 2. Iteratively merge the most frequent adjacent pair
        self.merges = {}
        for merge_idx in range(num_merges):
            if len(tokens) < 2:
                break
            # Count adjacent pair frequencies
            pair_counts = Counter(zip(tokens[:-1], tokens[1:]))
            if not pair_counts:
                break
            # Pick most common pair
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                # If the pair only occurred once, stop merging
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_id
                current_id += 1

            # Replace pair in tokens list
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.is_trained = True
        return self

    def encode(self, text: str) -> List[int]:
        """Convert a text string into a list of integer token IDs."""
        if not text:
            return []
        if not self.is_trained:
            raise RuntimeError("BPETokenizer must be trained before encoding.")

        tokens = list(text)
        # Apply learned merges in chronological training order
        for (p0, p1), merged in self.merges.items():
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back into string."""
        unk_token = ""
        pieces = []
        for tid in ids:
            tok = self.inverse_vocab.get(tid, unk_token)
            # Skip special control tokens during string reconstruction
            if tok in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                continue
            pieces.append(tok)
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def to_dict(self) -> dict:
        return {
            "merges": [list(k) + [v] for k, v in self.merges.items()],
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        tok = cls()
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {idx: token for token, idx in tok.vocab.items()}
        tok.merges = {(row[0], row[1]): row[2] for row in data["merges"]}
        tok.is_trained = True
        return tok
