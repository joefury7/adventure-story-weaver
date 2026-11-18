"""Your code. One stub per milestone; build them in order and keep each one small."""

import torch
from app.corpus import load_raw_corpus, preprocess_text, split_corpus, get_corpus_stats
from app.bpe import BPETokenizer
from app.embeddings import StoryEmbeddings


# --- Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
# Hint: Corpus Preparation: Gather and preprocess your raw text data.
def milestone_1(corpus_path=None):
    raw_text = load_raw_corpus(corpus_path)
    clean_text = preprocess_text(raw_text)
    train_text, val_text = split_corpus(clean_text, train_ratio=0.9)
    stats = get_corpus_stats(clean_text)
    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "train_text": train_text,
        "val_text": val_text,
        "stats": stats,
    }



# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
# Hint: BPE & Embeddings: Apply byte-pair encoding and create token embeddings.
def milestone_2(sample_text: str | None = None, num_merges: int = 20, n_embd: int = 32, block_size: int = 16):
    if sample_text is None:
        sample_text = "You stand before the obsidian archway of the Sunken Citadel. A cold draft carries the scent of damp moss."
    tokenizer = BPETokenizer().train(sample_text, num_merges=num_merges)
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)

    # Generate embeddings from the tokenized story text
    eff_block_size = max(block_size, len(token_ids[:block_size]))
    embeddings = StoryEmbeddings(vocab_size=tokenizer.vocab_size, n_embd=n_embd, block_size=eff_block_size)
    input_tensor = torch.tensor([token_ids[:eff_block_size]], dtype=torch.long)
    embedded_vectors = embeddings(input_tensor)

    return {
        "tokenizer": tokenizer,
        "token_ids": token_ids,
        "decoded_text": decoded_text,
        "vocab_size": tokenizer.vocab_size,
        "embeddings": embeddings,
        "embedded_shape": tuple(embedded_vectors.shape),
    }


from app.transformer import CausalSelfAttention, TransformerBlock, AdventureGPT


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal Self-Attention: Implement the masked attention mechanism and build a transformer block.
def milestone_3(n_embd: int = 32, n_heads: int = 4, block_size: int = 16):
    attn = CausalSelfAttention(n_embd=n_embd, n_heads=n_heads, block_size=block_size)
    block = TransformerBlock(n_embd=n_embd, n_heads=n_heads, block_size=block_size)

    dummy_x = torch.randn(2, block_size, n_embd)
    attn_out, weights = attn(dummy_x, return_attention_weights=True)
    block_out = block(dummy_x)

    # Upper triangle above diagonal must be 0 in causal attention weights
    upper_tri = torch.triu(weights[0, 0], diagonal=1)
    is_causal = bool(torch.all(upper_tri == 0).item())

    return {
        "attention": attn,
        "transformer_block": block,
        "attn_output_shape": tuple(attn_out.shape),
        "block_output_shape": tuple(block_out.shape),
        "attention_weights_shape": tuple(weights.shape),
        "is_causal": is_causal,
    }



from app.train import get_batch, estimate_loss, train_model, save_checkpoint, load_checkpoint


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
# Hint: Training Loop: Define the optimization process and train your GPT model.
def milestone_4(max_iters: int = 10, batch_size: int = 8, n_embd: int = 32, block_size: int = 16):
    corpus_info = milestone_1()
    train_text = corpus_info["train_text"][:3000]
    val_text = corpus_info["val_text"][:800]

    tokenizer = BPETokenizer().train(train_text, num_merges=10)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    model = AdventureGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=2,
        n_layers=2,
    )

    initial_loss_dict = estimate_loss(model, train_ids, val_ids, block_size=block_size, batch_size=batch_size, eval_iters=3)
    train_result = train_model(
        model,
        train_ids,
        val_ids,
        max_iters=max_iters,
        lr=3e-3,
        eval_interval=max(1, max_iters // 2),
        batch_size=batch_size,
        verbose=False,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "initial_loss": initial_loss_dict["train"],
        "final_loss": train_result["final_train_loss"],
        "history": train_result["history"],
        "iterations": max_iters,
    }



from app.generation import StoryWeaver, generate_tokens


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
# Hint: Text Generation: Implement greedy or sampling-based text generation.
def milestone_5(prompt: str = "You stand before the obsidian archway", max_new_tokens: int = 40, temperature: float = 0.8):
    m4_res = milestone_4(max_iters=5, batch_size=4, n_embd=32, block_size=32)
    model = m4_res["model"]
    tokenizer = m4_res["tokenizer"]

    weaver = StoryWeaver(model, tokenizer)
    greedy_story = weaver.weave(prompt, max_new_tokens=max_new_tokens, greedy=True)
    sampled_story = weaver.weave(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=10, top_p=0.9)

    return {
        "weaver": weaver,
        "prompt": prompt,
        "greedy_continuation": greedy_story,
        "sampled_continuation": sampled_story,
        "tokens_generated": max_new_tokens,
    }



# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# Hint: CLI Development: Create a simple command-line interface for user interaction.
def milestone_6(prompt: str = "You enter the cavern", max_tokens: int = 30):
    import io
    import sys
    from app.cli import main

    captured_out = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = captured_out
    try:
        main(["generate", "--prompt", prompt, "--max-tokens", str(max_tokens), "--greedy"])
    finally:
        sys.stdout = orig_stdout

    output_text = captured_out.getvalue()
    return {
        "output": output_text,
        "prompt": prompt,
        "success": bool(len(output_text) > 0 and prompt in output_text),
    }

