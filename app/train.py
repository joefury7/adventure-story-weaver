"""
Training Engine for Adventure Story Weaver
Implements batch extraction, loss evaluation, AdamW optimization, gradient clipping,
and model weight checkpointing.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from app.bpe import BPETokenizer
from app.transformer import AdventureGPT


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous chunks of length block_size for x and targets y."""
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be greater than block_size ({block_size})")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    eval_iters: int = 10,
) -> Dict[str, float]:
    """Estimate average cross-entropy loss on train and validation subsets."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean().item())
    model.train()
    return out


def train_model(
    model: AdventureGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    max_iters: int = 400,
    lr: float = 3e-3,
    eval_interval: int = 100,
    batch_size: int = 16,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
) -> Dict:
    """
    Train AdventureGPT with AdamW and track held-out validation loss.
    """
    block_size = model.block_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    history: List[Dict[str, float]] = []

    if verbose:
        print(f"Starting training on device '{next(model.parameters()).device}'...")
        print(f"Total Parameters: {model.get_num_params():,}")

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            eval_res = estimate_loss(model, train_data, val_data, block_size, batch_size)
            eval_res["step"] = it
            history.append(eval_res)
            if verbose:
                print(f"Step {it:4d}: Train Loss {eval_res['train']:.4f} | Val Loss {eval_res['val']:.4f}")

        x, y = get_batch(train_data, block_size, batch_size)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if checkpoint_path:
        save_checkpoint(model, checkpoint_path)
        if verbose:
            print(f"Saved model checkpoint to: {checkpoint_path}")

    return {
        "final_train_loss": history[-1]["train"] if history else loss.item(),
        "final_val_loss": history[-1]["val"] if history else loss.item(),
        "history": history,
        "iterations": max_iters,
    }


def save_checkpoint(model: AdventureGPT, path: str | Path) -> None:
    """Save model architecture config and state dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
            "n_embd": model.n_embd,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
        },
    }
    torch.save(state, str(path))


def load_checkpoint(path: str | Path) -> AdventureGPT:
    """Load model architecture and weights from saved checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    state = torch.load(str(path), map_location="cpu")
    cfg = state["config"]
    model = AdventureGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
