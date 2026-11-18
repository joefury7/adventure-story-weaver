"""Your code. One stub per milestone; build them in order and keep each one small."""

from app.corpus import load_raw_corpus, preprocess_text, split_corpus, get_corpus_stats


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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal Self-Attention: Implement the masked attention mechanism and build a transformer block.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
# Hint: Training Loop: Define the optimization process and train your GPT model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
# Hint: Text Generation: Implement greedy or sampling-based text generation.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# Hint: CLI Development: Create a simple command-line interface for user interaction.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
