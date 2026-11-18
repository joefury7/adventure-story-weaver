"""
Advanced unit tests for Adventure Story Weaver
Verifies mathematical invariants, tokenizer serialization, decoding controls,
and CLI edge cases.
"""

import io
import sys
import pytest
import torch
from app.bpe import BPETokenizer
from app.corpus import load_raw_corpus, preprocess_text, split_corpus, get_corpus_stats
from app.embeddings import StoryEmbeddings
from app.transformer import CausalSelfAttention, TransformerBlock, AdventureGPT
from app.train import get_batch, estimate_loss, train_model, save_checkpoint, load_checkpoint
from app.generation import StoryWeaver, generate_tokens
from app.cli import main


def test_bpe_roundtrip_and_serialization(tmp_path):
    text = "The dark cavern echoes with the sound of dripping water and distant groans."
    tokenizer = BPETokenizer().train(text, num_merges=12)
    
    # Round-trip reconstruction
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
    
    # JSON serialization roundtrip
    data_dict = tokenizer.to_dict()
    restored = BPETokenizer.from_dict(data_dict)
    assert restored.encode(text) == encoded
    assert restored.decode(encoded) == text


def test_attention_causal_invariance_and_softmax_sum():
    n_embd = 32
    n_heads = 4
    block_size = 16
    attn = CausalSelfAttention(n_embd=n_embd, n_heads=n_heads, block_size=block_size)
    
    x = torch.randn(2, block_size, n_embd)
    out, weights = attn(x, return_attention_weights=True)
    
    # 1. Output dimension matches input
    assert out.shape == x.shape
    
    # 2. Attention weights shape: (B, n_heads, T, T)
    assert weights.shape == (2, 4, 16, 16)
    
    # 3. Softmax rows must sum to 1.0 along the key dimension
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    # 4. Upper triangular values (future peeking) must be identically 0.0
    for b in range(2):
        for h in range(4):
            upper_tri = torch.triu(weights[b, h], diagonal=1)
            assert torch.all(upper_tri == 0.0)


def test_context_window_sliding_beyond_block_size():
    vocab_size = 30
    block_size = 8
    n_embd = 16
    model = AdventureGPT(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_heads=2, n_layers=2)
    
    # Provide input sequence longer than block_size (T = 14 > 8)
    long_idx = torch.randint(0, vocab_size, (1, 14))
    logits, _ = model(long_idx)
    
    # Output logits must match the cropped block_size (1, 8, vocab_size)
    assert logits.shape == (1, block_size, vocab_size)


def test_cli_subcommands(capsys):
    # Test 'hello'
    main(["hello"])
    out_hello = capsys.readouterr().out
    assert "is ready" in out_hello

    # Test 'inspect-tokens'
    main(["inspect-tokens", "The ancient crypt"])
    out_inspect = capsys.readouterr().out
    assert "BPE Token Inspection" in out_inspect
    assert "Token count" in out_inspect

    # Test 'stats'
    main(["stats"])
    out_stats = capsys.readouterr().out
    assert "Adventure Corpus Telemetry" in out_stats
    assert "Character Count" in out_stats

    # Test 'history'
    main(["history"])
    out_hist = capsys.readouterr().out
    assert "ADVENTURE STORY WEAVER - SAVED JOURNAL" in out_hist

    # Test 'compare'
    main(["compare", "--prompt", "You see a glowing chest", "--max-tokens", "20"])
    out_comp = capsys.readouterr().out
    assert "MODEL COMPARISON" in out_comp
    assert "From-Scratch AdventureGPT" in out_comp

