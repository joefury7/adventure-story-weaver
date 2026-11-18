# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ga8gz8ea32/badge.svg)](https://ask.thili.ai/b/ga8gz8ea32)

This project guides you through building a custom GPT from scratch to generate dynamic story continuations for text adventure games. You'll apply core NLP techniques like byte-pair tokenization and implement causal self-attention, culminating in a model that crafts engaging narrative segments.

## What you are building

**Goal.** You will build a text adventure story segment generator using a custom GPT, capable of creating dynamic story continuations based on adventure game narratives.

**Done when.** The project is complete when your custom GPT model, trained on adventure narratives, can generate coherent story continuations from a user-provided prompt via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Corpus Preparation: Gather and preprocess your raw text data.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: BPE & Embeddings: Apply byte-pair encoding and create token embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal Self-Attention: Implement the masked attention mechanism and build a transformer block.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Training Loop: Define the optimization process and train your GPT model.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Text Generation: Implement greedy or sampling-based text generation.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: CLI Development: Create a simple command-line interface for user interaction.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify the desired length of the generated story segment.
- Implement a feature to save and load trained model checkpoints for continued use.
- Explore different sampling strategies (e.g., top-k, nucleus sampling) for text generation to influence creativity and coherence.
