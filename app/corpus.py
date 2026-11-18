"""
Corpus Module for Adventure Story Weaver
Handles loading, preprocessing, cleaning, and splitting adventure narrative text.
"""

import os
import re
from pathlib import Path

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "adventure_corpus.txt"

FALLBACK_ADVENTURE_TEXT = """
You stand before the obsidian archway of the Sunken Citadel. A cold draft carries the scent of damp moss.
The chamber is circular, carved from seamless volcanic glass. In the center rests a pedestal adorned with a silver skull.
> You examine the intricate silver skull upon the pedestal.
Without warning, the silver skull rattles against the marble pedestal. A spectral voice whispers secrets of the celestial spheres.
With breath held, you pocket the artifact and slip into the shadows, leaving the tomb undisturbed once more.
---
The heavy iron doors of the Citadel of Torment creak open, revealing a spiral staircase winding into subterranean dark.
Cobwebs drape like decaying curtains across the decaying wooden rafters. A desiccated chest sits undisturbed.
> You draw your steel broadsword and tread cautiously across the creaking floorboards.
A grinding screech echoes as the flagstones beneath your boots shift. From the darkness above, a stone gargoyle descends.
The monster collapses into a shower of harmless obsidian dust, leaving behind a glowing gem of wondrous radiance.
"""


def load_raw_corpus(path: str | Path | None = None) -> str:
    """Load the raw text corpus from disk, using fallback if the file does not exist."""
    corpus_file = Path(path) if path else DEFAULT_CORPUS_PATH
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    return FALLBACK_ADVENTURE_TEXT.strip()


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess raw text data:
    - Normalizes strange unicode characters and smart quotes
    - Standardizes line breaks and whitespace
    - Ensures clean punctuation for tokenization
    """
    if not text:
        return ""
    # Replace non-standard unicode apostrophes and quotation marks
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", " - ")
    # Normalize multiple whitespace characters except intentional newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_corpus(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split clean text into training and validation strings."""
    split_idx = int(len(text) * train_ratio)
    return text[:split_idx], text[split_idx:]


def get_corpus_stats(text: str) -> dict:
    """Calculate descriptive statistics for the adventure corpus."""
    characters = len(text)
    unique_chars = sorted(list(set(text)))
    words = len(text.split())
    stories = len(text.split("---"))
    return {
        "character_count": characters,
        "vocabulary_size_chars": len(unique_chars),
        "word_count": words,
        "estimated_story_segments": stories,
        "sample": text[:150] + ("..." if len(text) > 150 else ""),
    }
