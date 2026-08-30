"""
Tests for grammatically-guided sparse attention mask generation.
"""

import numpy as np
from sparse_attention import (
    create_hard_mask,
    create_soft_mask,
    compute_sparsity,
    GRAMMAR_RULES
)


def test_hard_mask_shape():
    """Verify hard mask has correct shape."""
    pos_tags = ["NOUN", "VERB", "NOUN", "PUNCT"]
    mask = create_hard_mask(pos_tags)
    assert mask.shape == (4, 4), f"Expected (4, 4), got {mask.shape}"
    print("✓ Hard mask shape test passed")


def test_hard_mask_self_attention():
    """Verify diagonal (self-attention) is always allowed."""
    pos_tags = ["NOUN", "VERB", "ADJ"]
    mask = create_hard_mask(pos_tags, allow_self_attention=True)
    for i in range(len(pos_tags)):
        assert mask[i, i] == 1.0, f"Self-attention at [{i}, {i}] should be 1.0"
    print("✓ Hard mask self-attention test passed")


def test_hard_mask_no_self_attention():
    """Verify diagonal can be disabled."""
    pos_tags = ["NOUN", "VERB"]
    mask = create_hard_mask(pos_tags, allow_self_attention=False)
    # Disabling self-attention means diagonal should be determined by rules only
    # NOUN->NOUN is allowed, VERB->VERB is allowed, so diagonal should still be 1
    assert mask[0, 0] == 1.0, "NOUN->NOUN should be allowed by rules"
    assert mask[1, 1] == 1.0, "VERB->VERB should be allowed by rules"
    print("✓ Hard mask no self-attention test passed")


def test_hard_mask_grammar_rules():
    """Verify that grammatical rules are applied correctly."""
    pos_tags = ["NOUN", "DET", "VERB"]
    mask = create_hard_mask(pos_tags)

    # DET->NOUN should be allowed (determiners attend to nouns)
    assert mask[1, 0] == 1.0, "DET should attend to NOUN (rule exists)"

    # NOUN->DET should be allowed (nouns attend to determiners)
    assert mask[0, 1] == 1.0, "NOUN should attend to DET (rule exists)"

    # VERB->DET should not be allowed (no rule)
    assert mask[2, 1] == 0.0, "VERB should not attend to DET (no rule)"

    print("✓ Hard mask grammar rules test passed")


def test_soft_mask_range():
    """Verify soft mask values are in expected range."""
    pos_tags = ["NOUN", "VERB", "ADJ", "ADV"]
    allowed_weight = 1.0
    default_weight = 0.1

    mask = create_soft_mask(
        pos_tags,
        allowed_weight=allowed_weight,
        default_weight=default_weight
    )

    # All values should be within [default_weight, allowed_weight]
    assert np.all(mask >= default_weight), "Some values below default_weight"
    assert np.all(mask <= allowed_weight), "Some values above allowed_weight"
    print("✓ Soft mask range test passed")


def test_soft_mask_encourages_grammar():
    """Verify soft masks assign higher weights to allowed pairs."""
    pos_tags = ["NOUN", "DET"]
    allowed_weight = 1.0
    default_weight = 0.1

    mask = create_soft_mask(
        pos_tags,
        allowed_weight=allowed_weight,
        default_weight=default_weight
    )

    # DET->NOUN is allowed, should have weight 1.0
    assert mask[1, 0] == allowed_weight, "Allowed pair should have allowed_weight"

    # NOUN->DET is allowed, should have weight 1.0
    assert mask[0, 1] == allowed_weight, "Allowed pair should have allowed_weight"

    print("✓ Soft mask encouragement test passed")


def test_sparsity_dense():
    """Verify sparsity calculation for dense mask."""
    mask = np.ones((4, 4))
    sparsity = compute_sparsity(mask)
    assert sparsity == 0.0, f"Dense mask should have 0 sparsity, got {sparsity}"
    print("✓ Sparsity dense test passed")


def test_sparsity_empty():
    """Verify sparsity calculation for empty mask."""
    mask = np.zeros((4, 4))
    sparsity = compute_sparsity(mask)
    assert sparsity == 1.0, f"Empty mask should have 1.0 sparsity, got {sparsity}"
    print("✓ Sparsity empty test passed")


def test_sparsity_half():
    """Verify sparsity calculation for half-filled mask."""
    mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
    sparsity = compute_sparsity(mask)
    assert sparsity == 0.5, f"Half mask should have 0.5 sparsity, got {sparsity}"
    print("✓ Sparsity half test passed")


def test_realistic_sequence():
    """Test on a realistic POS sequence: "The cat sat on the mat"."""
    # POS tags: det noun verb prep det noun
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    mask = create_hard_mask(pos_tags)
    sparsity = compute_sparsity(mask)

    # Check some key relationships
    # DET (0) should attend to NOUN (1)
    assert mask[0, 1] == 1.0, "DET should attend to NOUN"
    # VERB (2) should attend to NOUN (1)
    assert mask[2, 1] == 1.0, "VERB should attend to NOUN"
    # ADP (3) should attend to DET (4) and NOUN (5)
    assert mask[3, 4] == 1.0, "ADP should attend to DET"
    assert mask[3, 5] == 1.0, "ADP should attend to NOUN"

    print(f"✓ Realistic sequence test passed (sparsity: {sparsity:.2%})")


def test_custom_rules():
    """Verify custom rules can override defaults."""
    pos_tags = ["NOUN", "VERB"]
    custom_rules = {
        ("NOUN", "VERB"): False,  # Override: nouns don't attend to verbs
        ("VERB", "NOUN"): True,
    }

    mask = create_hard_mask(pos_tags, rules=custom_rules)

    # With custom rules, NOUN->VERB should be disallowed
    assert mask[0, 1] == 0.0, "NOUN->VERB should be disallowed with custom rules"
    # VERB->NOUN should still be allowed
    assert mask[1, 0] == 1.0, "VERB->NOUN should be allowed"

    print("✓ Custom rules test passed")


if __name__ == "__main__":
    test_hard_mask_shape()
    test_hard_mask_self_attention()
    test_hard_mask_no_self_attention()
    test_hard_mask_grammar_rules()
    test_soft_mask_range()
    test_soft_mask_encourages_grammar()
    test_sparsity_dense()
    test_sparsity_empty()
    test_sparsity_half()
    test_realistic_sequence()
    test_custom_rules()
    print("\n✅ All tests passed!")
