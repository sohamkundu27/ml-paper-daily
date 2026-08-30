# Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers

## Paper Details

**Title:** Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers

**arXiv:** https://arxiv.org/abs/2605.24518

**Authors:** Spandan Pratyush et al.

**Published:** May 23, 2026

## Summary

This paper addresses the quadratic complexity bottleneck in transformer self-attention by leveraging linguistic structure. The core insight is that not all token pairs need to attend to each other—instead, attention can be constrained based on the grammatical relationships between tokens (their Parts-of-Speech tags). The authors propose dynamic masking strategies (hard masks that strictly enforce grammatical rules, and soft masks that bias toward them) to reduce the computational graph while preserving linguistic coherence. This approach achieves significant efficiency gains without sacrificing language understanding performance.

## Plan: 4 Passes

**Pass 1:** Core POS-guided sparse attention mask generation. Build a lightweight module that takes a sequence of tokens with their POS tags and generates either hard or soft attention masks based on predefined grammatical rules (e.g., verbs attend to nouns, adjectives attend to nouns). Include a basic test with synthetic sequences.

**Pass 2:** Integrate sparse attention masking into a minimal transformer attention layer. Modify standard multi-head attention to apply the grammatical masks and verify computational savings.

**Pass 3:** Extend to full transformer blocks and add a simple NLTK-based POS tagger to automatically annotate sequences. Demonstrate relative speedup on synthetic and real text.

**Pass 4:** End-to-end demo: train a tiny transformer language model (or fine-tune a pretrained model) with and without grammatical masks on a toy dataset, showing efficiency vs. standard attention.

## Implemented vs. Simplified

### Pass 1 Implementation

**Core Components:**
- `sparse_attention.py`: Complete POS-guided mask generation supporting both hard and soft masking strategies.
- **Hard masks:** Strict binary (0/1) attention masks based on grammatical rules.
- **Soft masks:** Weighted masks (default_weight for disallowed, allowed_weight for permitted) that encourage but don't forbid non-grammatical attention.
- **Grammar Rules:** 20+ POS pair rules covering standard English (NOUN, VERB, ADJ, DET, ADP, PRON, ADV, AUX, PUNCT).
- **Utilities:** Sparsity computation (fraction of masked positions).

**Testing:**
- 11 comprehensive unit tests verifying mask generation, self-attention, rule application, soft/hard mask behavior.
- Tests on synthetic sequences and a realistic example ("The cat sat on the mat").
- On realistic sequences, achieves ~36% sparsity (63% of attention positions computed vs. 100% dense).

**What is stubbed/simplified:**
- No actual POS tagging; all tests use manually provided POS tag sequences (no NLTK integration yet).
- No transformer integration; purely mask generation as a standalone module.
- No end-to-end efficiency measurement or wall-clock speedup benchmarks yet.
- Rules are English-only; no multilingual or language-specific variants.
- No positional biasing or learnable rule weighting (rules are static).
