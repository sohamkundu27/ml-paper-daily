"""
Grammatically-Guided Sparse Attention
Core POS-guided mask generation for efficient transformer attention.
"""

import numpy as np
from typing import Dict, Set, Tuple, Optional


# Default grammatical rules: which POS pairs should attend
GRAMMAR_RULES = {
    # Verbs attend to nouns, pronouns, and other verbs
    ("VERB", "NOUN"): True,
    ("VERB", "PRON"): True,
    ("VERB", "VERB"): True,
    ("VERB", "AUX"): True,  # auxiliary verbs
    ("VERB", "ADV"): True,  # adverbs modify verbs

    # Nouns attend to determiners, adjectives, other nouns
    ("NOUN", "DET"): True,
    ("NOUN", "ADJ"): True,
    ("NOUN", "NOUN"): True,
    ("NOUN", "PRON"): True,

    # Adjectives attend to nouns and adverbs
    ("ADJ", "NOUN"): True,
    ("ADJ", "ADV"): True,
    ("ADJ", "ADJ"): True,

    # Determiners attend to nouns and adjectives
    ("DET", "NOUN"): True,
    ("DET", "ADJ"): True,

    # Adverbs attend to verbs, adjectives, other adverbs
    ("ADV", "VERB"): True,
    ("ADV", "ADJ"): True,
    ("ADV", "ADV"): True,

    # Pronouns behave like nouns
    ("PRON", "VERB"): True,
    ("PRON", "NOUN"): True,
    ("PRON", "PRON"): True,
    ("PRON", "AUX"): True,

    # Prepositions attend to nouns, pronouns, verbs, and determiners
    ("ADP", "NOUN"): True,
    ("ADP", "PRON"): True,
    ("ADP", "VERB"): True,
    ("ADP", "DET"): True,

    # Punctuation attends to everything (permissive)
    ("PUNCT", "NOUN"): True,
    ("PUNCT", "VERB"): True,
    ("PUNCT", "PUNCT"): True,

    # Self-attention (every token attends to itself)
    # Handled separately via diagonal masking
}


def create_hard_mask(
    pos_tags: list,
    rules: Optional[Dict[Tuple[str, str], bool]] = None,
    allow_self_attention: bool = True
) -> np.ndarray:
    """
    Create a hard (binary) grammatical attention mask.

    Args:
        pos_tags: List of POS tags for tokens in the sequence.
        rules: Dict mapping (source_pos, target_pos) to allowed (True/False).
               If None, uses default GRAMMAR_RULES.
        allow_self_attention: If True, allow each token to attend to itself.

    Returns:
        Attention mask of shape (seq_len, seq_len) where 1 = attend, 0 = mask.
    """
    if rules is None:
        rules = GRAMMAR_RULES

    seq_len = len(pos_tags)
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)

    for i in range(seq_len):
        for j in range(seq_len):
            source_pos = pos_tags[i]
            target_pos = pos_tags[j]

            # Self-attention
            if i == j and allow_self_attention:
                mask[i, j] = 1.0
                continue

            # Check rule
            key = (source_pos, target_pos)
            if key in rules and rules[key]:
                mask[i, j] = 1.0

    return mask


def create_soft_mask(
    pos_tags: list,
    rules: Optional[Dict[Tuple[str, str], bool]] = None,
    allowed_weight: float = 1.0,
    default_weight: float = 0.1,
    allow_self_attention: bool = True
) -> np.ndarray:
    """
    Create a soft (weighted) grammatical attention mask.

    Soft masks encourage attention toward grammatically allowed pairs
    but don't completely forbid others.

    Args:
        pos_tags: List of POS tags.
        rules: Dict mapping (source_pos, target_pos) to allowed.
        allowed_weight: Weight assigned to allowed attention.
        default_weight: Weight assigned to disallowed attention.
        allow_self_attention: If True, self-attention always has weight 1.0.

    Returns:
        Soft attention mask of shape (seq_len, seq_len).
    """
    if rules is None:
        rules = GRAMMAR_RULES

    seq_len = len(pos_tags)
    mask = np.full((seq_len, seq_len), default_weight, dtype=np.float32)

    for i in range(seq_len):
        for j in range(seq_len):
            source_pos = pos_tags[i]
            target_pos = pos_tags[j]

            # Self-attention
            if i == j and allow_self_attention:
                mask[i, j] = 1.0
                continue

            # Check rule
            key = (source_pos, target_pos)
            if key in rules and rules[key]:
                mask[i, j] = allowed_weight

    return mask


def compute_sparsity(mask: np.ndarray) -> float:
    """Compute sparsity: fraction of masked-out (zero) positions."""
    total = mask.size
    masked = np.sum(mask == 0)
    return masked / total if total > 0 else 0.0
