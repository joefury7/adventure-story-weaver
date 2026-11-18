"""
Command-line interface for Adventure Story Weaver: `python -m app.cli --help`
Provides commands for:
- hello: Environment & health check
- generate: Generate dynamic story continuations from user prompts
- train: Train custom GPT from scratch on narrative corpus
- play: Interactive text-adventure game engine powered by AdventureGPT
- history: Review past generated story segments from the local journal store
- compare: Compare from-scratch AdventureGPT completions against an LLM baseline
- inspect-tokens: Visualize Byte-Pair Encoding subword segmentation
- stats: Display corpus and architecture telemetry
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

from app.bpe import BPETokenizer
from app.corpus import load_raw_corpus, preprocess_text, split_corpus, get_corpus_stats
from app.generation import StoryWeaver
from app.storage import Store
from app import llm
from app.transformer import AdventureGPT
from app.train import train_model, save_checkpoint, load_checkpoint

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "data" / "story_gpt.pt"
DEFAULT_VOCAB = Path(__file__).resolve().parent.parent / "data" / "bpe_vocab.json"


def load_or_init_pipeline(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    vocab_path: Path | str = DEFAULT_VOCAB,
    quick_init_if_missing: bool = True,
) -> tuple[AdventureGPT, BPETokenizer]:
    """Load model and tokenizer from disk, or initialize and quickly train a mini-model."""
    ckpt_p = Path(checkpoint_path)
    vocab_p = Path(vocab_path)

    if ckpt_p.exists() and vocab_p.exists():
        with open(vocab_p, "r", encoding="utf-8") as f:
            tokenizer = BPETokenizer.from_dict(json.load(f))
        model = load_checkpoint(ckpt_p)
        return model, tokenizer

    if not quick_init_if_missing:
        raise FileNotFoundError(f"Checkpoint {ckpt_p} or vocab {vocab_p} not found. Run 'train' first.")

    # Initialize a fast fallback model on sample corpus
    raw = load_raw_corpus()
    clean = preprocess_text(raw)
    tokenizer = BPETokenizer().train(clean[:15000], num_merges=30)
    model = AdventureGPT(vocab_size=tokenizer.vocab_size, block_size=32, n_embd=48, n_heads=4, n_layers=2)
    return model, tokenizer


def cmd_generate(args: argparse.Namespace) -> None:
    prompt = args.prompt
    model, tokenizer = load_or_init_pipeline(args.checkpoint, args.vocab)
    weaver = StoryWeaver(model, tokenizer)

    continuation = weaver.weave(
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
    )

    # Persist generated story to the local journal store
    try:
        store = Store()
        entry_key = f"story_{int(time.time() * 1000)}"
        store.put(entry_key, {
            "prompt": prompt,
            "continuation": continuation.strip(),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        })
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("           ADVENTURE STORY WEAVER - CONTINUATION")
    print("=" * 60)
    print(continuation.strip())
    print("=" * 60 + "\n")


def cmd_train(args: argparse.Namespace) -> None:
    print(f"Loading corpus from: {args.corpus or 'default data/adventure_corpus.txt'}...")
    raw = load_raw_corpus(args.corpus)
    clean = preprocess_text(raw)
    train_txt, val_txt = split_corpus(clean, train_ratio=0.9)

    print(f"Training Byte-Pair Encoding with {args.merges} merges...")
    tokenizer = BPETokenizer().train(clean[:40000], num_merges=args.merges)
    vocab_file = Path(args.vocab)
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)
    print(f"Saved vocabulary ({tokenizer.vocab_size} tokens) to {vocab_file}")

    train_ids = torch.tensor(tokenizer.encode(train_txt), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_txt), dtype=torch.long)

    model = AdventureGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.embd,
        n_heads=args.heads,
        n_layers=args.layers,
    )

    train_model(
        model=model,
        train_data=train_ids,
        val_data=val_ids,
        max_iters=args.iters,
        lr=args.lr,
        eval_interval=max(1, args.iters // 5),
        batch_size=args.batch_size,
        checkpoint_path=args.out,
        verbose=True,
    )


def cmd_play(args: argparse.Namespace) -> None:
    """Interactive text adventure session."""
    model, tokenizer = load_or_init_pipeline(args.checkpoint, args.vocab)
    weaver = StoryWeaver(model, tokenizer)
    store = Store()

    banner = r"""
  ==============================================================
    ADVENTURE STORY WEAVER - INTERACTIVE DUNGEON ENGINE
  ==============================================================
    Type your action (e.g. 'open door', 'inspect runes', 'go north')
    Type 'quit' or 'exit' to end the adventure.
  ==============================================================
    """
    print(banner)

    story_context = "You stand before the obsidian archway of the Sunken Citadel. A cold draft carries the scent of damp moss.\n"
    print(story_context)

    turn = 1
    session_turns = []
    while True:
        try:
            action = input(f"\n[Turn {turn}] What do you do? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting adventure. Farewell, traveler.")
            break

        if not action or action.lower() in ["quit", "exit"]:
            print("\nExiting adventure. Farewell, traveler.")
            break

        prompt = f"{story_context.strip()}\n> {action}\n"
        print("\n[The Weaver writes...]")
        continuation = weaver.weave(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=20,
            top_p=0.9,
        )
        new_segment = continuation[len(prompt):].strip()
        if not new_segment:
            new_segment = "You proceed forward cautiously into the darkened corridor."

        print(f"\n{new_segment}\n")
        session_turns.append({"turn": turn, "action": action, "result": new_segment})
        story_context = f"{action}\n{new_segment}"
        turn += 1

    if session_turns:
        store.put(f"session_{int(time.time())}", session_turns)


def cmd_history(args: argparse.Namespace) -> None:
    """View saved adventure history from the persistent storage."""
    store = Store()
    entries = store.all()
    print("\n" + "=" * 60)
    print("           ADVENTURE STORY WEAVER - SAVED JOURNAL")
    print("=" * 60)
    if not entries:
        print("No saved stories in the journal yet. Run 'generate' or 'play' first.")
    else:
        for k, v in list(entries.items())[-5:]:
            if isinstance(v, dict) and "prompt" in v:
                print(f"[{k}]")
                print(f"  Prompt : {v.get('prompt')}")
                print(f"  Story  : {v.get('continuation')[:120]}...\n")
            elif isinstance(v, list):
                print(f"[{k}] (Interactive Game Session with {len(v)} turns)\n")
    print("=" * 60 + "\n")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare from-scratch AdventureGPT against an external LLM baseline."""
    print("\n" + "=" * 60)
    print("     MODEL COMPARISON: From-Scratch GPT vs External Baseline")
    print("=" * 60)
    model, tokenizer = load_or_init_pipeline(args.checkpoint, args.vocab)
    weaver = StoryWeaver(model, tokenizer)

    local_out = weaver.weave(prompt=args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
    print(f"\n--- [1] From-Scratch AdventureGPT (159k params) ---")
    print(local_out.strip())

    print(f"\n--- [2] External LLM Baseline ---")
    try:
        base, key, model_name = llm.settings()
        if model_name:
            reply = llm.chat(
                [{"role": "user", "content": f"Continue this fantasy adventure story in 2 sentences: {args.prompt}"}],
                max_tokens=args.max_tokens,
            )
            print(f"Model ({model_name}): {reply.strip()}")
        else:
            print("No LLM_MODEL defined in .env. (Set LLM_MODEL, LLM_API_KEY in .env to call Gemini/Groq/Ollama).")
    except Exception as e:
        print(f"External LLM call skipped: {e}")
    print("=" * 60 + "\n")


def cmd_inspect_tokens(args: argparse.Namespace) -> None:
    _, tokenizer = load_or_init_pipeline(quick_init_if_missing=True)
    text = args.text
    token_ids = tokenizer.encode(text)
    decoded_pieces = [tokenizer.inverse_vocab.get(tid, "<unk>") for tid in token_ids]

    print("\n--- BPE Token Inspection ---")
    print(f"Input text    : {text}")
    print(f"Token count   : {len(token_ids)} (compressed from {len(text)} chars)")
    print(f"Token IDs     : {token_ids}")
    print(f"Subword pieces: {decoded_pieces}")
    print("----------------------------\n")


def cmd_stats(args: argparse.Namespace) -> None:
    raw = load_raw_corpus(args.corpus)
    stats = get_corpus_stats(raw)
    print("\n--- Adventure Corpus Telemetry ---")
    for k, v in stats.items():
        if k != "sample":
            print(f"{k.replace('_', ' ').title():28}: {v}")
    print(f"\nSample Snippet:\n{stats['sample']}")
    print("----------------------------------\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="adventure-story-weaver",
        description="Adventure Story Weaver: Custom GPT for dynamic adventure narratives.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Hello (Smoke test compatibility)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. Generate
    gen_parser = sub.add_parser("generate", help="generate story continuation from prompt")
    gen_parser.add_argument("--prompt", "-p", type=str, default="You enter the ancient ruined crypt.", help="Story seed prompt")
    gen_parser.add_argument("--max-tokens", "-n", type=int, default=120, help="Max new tokens to generate")
    gen_parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-k", "-k", type=int, default=20, help="Top-k filtering limit")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    gen_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty scale")
    gen_parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding")
    gen_parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model weights path")
    gen_parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB), help="BPE vocabulary path")

    # 3. Train
    train_parser = sub.add_parser("train", help="train GPT model on narrative corpus")
    train_parser.add_argument("--corpus", type=str, default=None, help="Path to raw narrative corpus")
    train_parser.add_argument("--iters", type=int, default=300, help="Training iterations")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--block-size", type=int, default=64, help="Sequence context length")
    train_parser.add_argument("--embd", type=int, default=64, help="Embedding dimension")
    train_parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    train_parser.add_argument("--layers", type=int, default=3, help="Transformer layers")
    train_parser.add_argument("--merges", type=int, default=50, help="BPE merges")
    train_parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    train_parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT), help="Output checkpoint path")
    train_parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB), help="Output vocabulary path")

    # 4. Play (Interactive Text Adventure)
    play_parser = sub.add_parser("play", help="play interactive text adventure game")
    play_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per turn")
    play_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    play_parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Checkpoint path")
    play_parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB), help="Vocabulary path")

    # 5. History (Persistent Store)
    sub.add_parser("history", help="view history of generated adventure segments")

    # 6. Compare (Local GPT vs External LLM baseline)
    comp_parser = sub.add_parser("compare", help="compare local GPT output with external LLM")
    comp_parser.add_argument("--prompt", "-p", type=str, default="You approach the dark dragon altar.", help="Seed prompt")
    comp_parser.add_argument("--max-tokens", "-n", type=int, default=80, help="Max tokens to generate")
    comp_parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Temperature")
    comp_parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Weights")
    comp_parser.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB), help="Vocab")

    # 7. Inspect tokens
    tok_parser = sub.add_parser("inspect-tokens", help="visualize BPE token breakdown")
    tok_parser.add_argument("text", type=str, help="Text to tokenize")

    # 8. Stats
    stats_parser = sub.add_parser("stats", help="view dataset statistics")
    stats_parser.add_argument("--corpus", type=str, default=None, help="Corpus path")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Adventure Story Weaver is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "inspect-tokens":
        cmd_inspect_tokens(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
