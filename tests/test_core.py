"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)

# Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
def test_milestone_1():
    result = milestone_1()
    assert isinstance(result, dict)
    assert "clean_text" in result
    assert "train_text" in result
    assert "val_text" in result
    assert "stats" in result
    assert len(result["clean_text"]) > 1000
    assert len(result["train_text"]) > len(result["val_text"])
    assert result["stats"]["word_count"] > 100
    assert "You stand before" in result["clean_text"] or "Citadel" in result["clean_text"]



# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
def test_milestone_2():
    sample = "You draw your steel broadsword and tread cautiously across the creaking floorboards."
    res = milestone_2(sample_text=sample, num_merges=15, n_embd=32, block_size=16)

    assert "tokenizer" in res
    assert "token_ids" in res
    assert "decoded_text" in res
    assert "embeddings" in res
    assert "embedded_shape" in res

    # BPE round-trip check
    assert res["decoded_text"] == sample
    # Encoded sequence should be strictly shorter than raw character length due to merges
    assert len(res["token_ids"]) < len(sample)
    # Embedding tensor shape should be (1, block_size, 32)
    assert res["embedded_shape"] == (1, 16, 32)



# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    res = milestone_3(n_embd=32, n_heads=4, block_size=16)

    assert "attention" in res
    assert "transformer_block" in res
    assert res["attn_output_shape"] == (2, 16, 32)
    assert res["block_output_shape"] == (2, 16, 32)
    assert res["attention_weights_shape"] == (2, 4, 16, 16)
    assert res["is_causal"] is True

    # Test residual gradient flow
    block = res["transformer_block"]
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)



# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
def test_milestone_4(tmp_path):
    res = milestone_4(max_iters=5, batch_size=4, n_embd=32, block_size=16)

    assert "model" in res
    assert "tokenizer" in res
    assert "initial_loss" in res
    assert "final_loss" in res
    assert "history" in res

    assert res["initial_loss"] > 0.0
    assert res["final_loss"] > 0.0
    assert len(res["history"]) >= 1

    # Verify model checkpointing
    from app.train import save_checkpoint, load_checkpoint
    ckpt_path = tmp_path / "test_gpt.pt"
    save_checkpoint(res["model"], ckpt_path)
    assert ckpt_path.exists()

    loaded_model = load_checkpoint(ckpt_path)
    assert loaded_model.vocab_size == res["model"].vocab_size
    assert loaded_model.n_embd == res["model"].n_embd



# Milestone 5: Implement a text generation feature that creates coherent story continuations.
def test_milestone_5():
    prompt = "You stand before the obsidian archway"
    res = milestone_5(prompt=prompt, max_new_tokens=25, temperature=0.7)

    assert "weaver" in res
    assert "greedy_continuation" in res
    assert "sampled_continuation" in res

    # The generated story must contain the original prompt
    assert prompt in res["greedy_continuation"]
    assert prompt in res["sampled_continuation"]

    # The generated text must extend beyond the prompt
    assert len(res["greedy_continuation"]) > len(prompt)
    assert len(res["sampled_continuation"]) > len(prompt)



# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
def test_milestone_6():
    prompt = "You step into the darkened tunnel"
    res = milestone_6(prompt=prompt, max_tokens=25)

    assert "output" in res
    assert res["success"] is True
    assert "ADVENTURE STORY WEAVER" in res["output"]
    assert prompt in res["output"]


