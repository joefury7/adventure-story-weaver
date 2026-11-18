"""
Text Generation Engine for Adventure Story Weaver
Supports:
- Autoregressive continuation from arbitrary story prompts
- Greedy decoding
- Temperature scaling
- Top-k truncation
- Top-p (Nucleus) filtering
- Repetition penalty
- Interactive streaming generator
"""

from typing import Generator, List, Optional
import torch
import torch.nn.functional as F
from app.bpe import BPETokenizer
from app.transformer import AdventureGPT


@torch.no_grad()
def generate_tokens(
    model: AdventureGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: Optional[int] = 20,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    greedy: bool = False,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate new token IDs autoregressively from the input index tensor.
    """
    model.eval()
    block_size = model.block_size

    for _ in range(max_new_tokens):
        # Crop context window to block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        logits, _ = model(idx_cond)
        # Take the logits of the last token in sequence: (B, vocab_size)
        logits = logits[:, -1, :].clone()

        # Apply repetition penalty to recently generated tokens
        if repetition_penalty != 1.0 and idx.size(1) > 0:
            for token_id in set(idx[0].tolist()[-32:]):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if greedy or temperature <= 1e-4:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Scale by temperature
            logits = logits / max(temperature, 1e-4)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                drop_mask = cumulative_probs > top_p
                # Shift mask right so at least one token is kept
                drop_mask[:, 1:] = drop_mask[:, :-1].clone()
                drop_mask[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(drop_mask, float("-inf"))
                logits = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return idx


class StoryWeaver:
    """
    High-level interface to generate and weave dynamic adventure stories.
    """
    def __init__(self, model: AdventureGPT, tokenizer: BPETokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def weave(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        greedy: bool = False,
    ) -> str:
        """Generate a story continuation given a user prompt."""
        if not prompt:
            prompt = "\n"

        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]

        device = next(self.model.parameters()).device
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        eos_id = self.tokenizer.vocab.get("<eos>", None)
        output_ids = generate_tokens(
            self.model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            greedy=greedy,
            eos_token_id=eos_id,
        )

        full_text = self.tokenizer.decode(output_ids[0].tolist())
        return full_text
