# Adventure Story Weaver

[![tests](https://github.com/joefury7/adventure-story-weaver/actions/workflows/ci.yml/badge.svg)](https://github.com/joefury7/adventure-story-weaver/actions/workflows/ci.yml)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ga8gz8ea32/badge.svg)](https://ask.thili.ai/b/ga8gz8ea32)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure **from-scratch Generative Pre-trained Transformer (GPT)** built in PyTorch to weave dynamic, atmospheric story continuations for text adventure games and interactive fiction.

---

## Highlights

- **Pure From-Scratch Transformer**: Scaled dot-product self-attention with causal lower-triangular masking, multi-head projection, Pre-LayerNorm residual blocks, and tied token embedding projections. Zero black-box dependencies.
- **Custom Byte-Pair Encoding (BPE)**: Pure Python subword tokenizer trained from raw text using iterative adjacent pair frequency counting, invertible roundtrip decoding, and JSON serialization.
- **Atmospheric Narrative Corpus**: Over 900,000 characters (~150,000 words) of curated fantasy catacombs, sunken citadels, arcane observatories, and choice-driven interactive fiction encounters.
- **Advanced Decoding Engine**: Autoregressive generation equipped with Temperature scaling, Top-$k$ truncation, Top-$p$ (Nucleus) sampling, and dynamic Repetition Penalties.
- **Interactive CLI & Game Engine**: Full command-line interface featuring an interactive dungeon crawler game loop (`play`), prompt continuation (`generate`), training pipeline (`train`), and tokenizer inspection (`inspect-tokens`).

---

## Architecture Overview

```text
               +-----------------------------------------+
               |               Input Text                |
               +-----------------------------------------+
                                    |
                                    v
               +-----------------------------------------+
               |      Custom BPE Subword Tokenizer       |
               +-----------------------------------------+
                                    |
                                    v
               +-----------------------------------------+
               | Token Embeddings + Positional Embeddings|
               +-----------------------------------------+
                                    |
                                    v
                   +---------------------------------+
                   |       Transformer Block (x3)     |
                   |                                 |
                   |   x = x + CausalAttn(LayerNorm) |
                   |   x = x + FeedForward(LayerNorm)|
                   +---------------------------------+
                                    |
                                    v
               +-----------------------------------------+
               |       Final LayerNorm + LM Head         |
               +-----------------------------------------+
                                    |
                                    v
               +-----------------------------------------+
               |   Decoding: Temperature, Top-k, Top-p   |
               +-----------------------------------------+
                                    |
                                    v
               +-----------------------------------------+
               |        Dynamic Story Continuation       |
               +-----------------------------------------+
```

### Mathematical Foundations

1. **Causal Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$
   Where $M_{i,j} = 0$ for $j \le i$, and $M_{i,j} = -\infty$ for $j > i$ (preventing future token leakage).

2. **Nucleus (Top-$p$) Sampling**:
   Selects the minimal set of tokens $V^{(p)} \subset V$ such that:
   $$\sum_{i \in V^{(p)}} P(x_t = i \mid x_{<t}) \ge p$$
   And renormalizes probabilities exclusively over $V^{(p)}$.

---

## Milestones

- [x] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Curated and preprocessed 930,000+ characters of rich fantasy dungeon narratives, room descriptions, player actions, and encounters in `data/adventure_corpus.txt`.
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Implemented `BPETokenizer` in `app/bpe.py` with iterative pair counting and merge tables, plus `StoryEmbeddings` combining learned token and positional embeddings.
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Implemented `CausalSelfAttention`, `MultiHeadAttention`, `FeedForward`, and `TransformerBlock` in `app/transformer.py` using Pre-LayerNorm and residual skip connections.
- [x] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Engineered `train_model()` in `app/train.py` with AdamW, gradient clipping, held-out validation loss evaluation, and model checkpointing (`data/story_gpt.pt`).
- [x] **5. Implement a text generation feature that creates coherent story continuations.**  
      Built `StoryWeaver` and `generate_tokens()` in `app/generation.py` supporting greedy decoding, temperature control, top-$k$, nucleus top-$p$, and repetition penalties.
- [x] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Designed a modular CLI in `app/cli.py` featuring story generation, full model training, interactive game mode, token inspection, and corpus stats.

---

## Quickstart

### Prerequisites
- Python 3.12+ (or 3.11+)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/joefury7/adventure-story-weaver.git
cd adventure-story-weaver

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## CLI Usage Guide

### 1. Generate Story Continuation
Generate a dynamic segment from any adventure seed prompt:

```bash
python -m app.cli generate --prompt "You draw your broadsword and approach the stone sarcophagus." --max-tokens 100 --temperature 0.8 --top-p 0.9
```

### 2. Play Interactive Adventure Game Mode
Step into an interactive text adventure powered directly by AdventureGPT:

```bash
python -m app.cli play
```
*Example Session:*
```text
  ==============================================================
    ADVENTURE STORY WEAVER - INTERACTIVE DUNGEON ENGINE
  ==============================================================
    Type your action (e.g. 'open door', 'inspect runes', 'go north')
    Type 'quit' or 'exit' to end the adventure.
  ==============================================================

You stand before the obsidian archway of the Sunken Citadel. A cold draft carries the scent of damp moss.

[Turn 1] What do you do? > inspect the glowing runes on the archway

[The Weaver writes...]
You trace your fingers along the glowing arcane sigils inscribed on the stone door.
A secret panel in the masonry slides back, revealing a narrow tunnel descending toward a forgotten armory.
```

### 3. Inspect BPE Subword Tokenization
Visualize how the Byte-Pair Encoder breaks narrative text down into subwords:

```bash
python -m app.cli inspect-tokens "The dragon breathed turquoise fire across the cavern ceiling."
```

### 4. Train from Scratch
Train your own custom checkpoint on any corpus:

```bash
python -m app.cli train --iters 300 --batch-size 16 --block-size 64 --embd 64 --heads 4 --layers 3
```

### 5. View Corpus Statistics
```bash
python -m app.cli stats
```

---

## Running the Test Suite

Run the full automated test suite (all 13 unit, smoke, and mathematical invariant tests pass):

```bash
pytest -v
```

---

## Stretch Goals Implemented

1. **Interactive CLI Dungeon Crawler**: Turn-by-turn interactive adventure game engine with dynamic story generation loop (`python -m app.cli play`).
2. **Byte-Pair Encoding from Scratch**: Full BPE implementation with iterative frequency counting, merge tables, and JSON serialization without third-party tokenization libraries.
3. **Multi-Strategy Decoding Engine**:
   - Temperature scaling ($T < 1.0$ conservative, $T > 1.0$ creative).
   - Top-$k$ truncation.
   - Top-$p$ (Nucleus) probability mass filtering.
   - Dynamic repetition penalty to prevent text degeneration loops.
4. **Complete Checkpoint Serialization**: Clean save/load pipeline preserving architecture hyperparameters and model state tensors (`save_checkpoint()` / `load_checkpoint()`).
5. **Context-Window Sliding**: Handles prompts exceeding the model context block length by dynamically sliding the context window.

---

## License

MIT License. Crafted for the askThili Language Models from Scratch Track.
