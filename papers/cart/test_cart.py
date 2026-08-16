"""
Tests for CART Pass 1: Core MLA block with LTI gate
Tests for CART Pass 2: Multi-layer prelude network integration
Tests for CART Pass 3: Stacked recurrent iterations with layer norm and sequence classification task
"""

import torch
import torch.nn as nn
from cart import (
    CartMLABlock,
    LearnedLTIGate,
    CartRecurrentCore,
    CartPrelude,
    Cart,
    CartSequenceClassifier,
    CartLanguageModel,
    StandardTransformerBaseline,
    create_dummy_context,
    create_dummy_raw_context,
    create_synthetic_length_classification_dataset,
    create_synthetic_language_dataset,
    train_classifier_step,
    evaluate_classifier,
    count_parameters,
    train_lm_step,
    evaluate_lm,
)


def test_mla_block_forward_shape():
    """Test that MLA block produces correct output shape."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    mla = CartMLABlock(dim, head_dim)
    x = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = mla(x, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_mla_block_gradient_flow():
    """Test that gradients flow through MLA block."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    mla = CartMLABlock(dim, head_dim)
    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = mla(x, context_k, context_v)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient not computed for input"
    assert mla.q_proj.weight.grad is not None, "Gradient not computed for q_proj"
    assert mla.out_proj.weight.grad is not None, "Gradient not computed for out_proj"


def test_lti_gate_spectral_radius():
    """Test that LTI gate maintains spectral radius in stable range."""
    gate = LearnedLTIGate(init_alpha=0.8)

    # Check initial spectral radius
    rho_init = gate.get_spectral_radius()
    assert 0.0 <= rho_init < 1.0, f"Spectral radius {rho_init} out of stable range [0, 1)"

    # Test recurrent update
    batch, seq_len, dim = 2, 10, 64
    x = torch.randn(batch, seq_len, dim)
    y = torch.randn(batch, seq_len, dim)

    x_new = gate(x, y)

    # Verify x_new = alpha * x + y
    expected = rho_init * x + y
    torch.testing.assert_close(x_new, expected, rtol=1e-5, atol=1e-6)


def test_lti_gate_clipping():
    """Test that LTI gate clips alpha to valid range."""
    gate = LearnedLTIGate(init_alpha=1.5)  # Invalid: > 1

    rho = gate.get_spectral_radius()
    assert 0.0 <= rho < 1.0, f"Spectral radius {rho} not clipped to [0, 1)"


def test_lti_gate_trainable():
    """Test that LTI gate alpha is trainable."""
    gate = LearnedLTIGate(init_alpha=0.8)
    alpha_init = gate.get_spectral_radius().item()

    batch, seq_len, dim = 2, 10, 64
    x = torch.randn(batch, seq_len, dim)
    y = torch.randn(batch, seq_len, dim)

    x_new = gate(x, y)
    loss = x_new.sum()
    loss.backward()

    assert gate.alpha.grad is not None, "Gradient not computed for alpha"

    # Verify alpha changed after update
    with torch.no_grad():
        gate.alpha -= 0.01 * gate.alpha.grad

    alpha_new = gate.get_spectral_radius().item()
    assert alpha_init != alpha_new, "Alpha not updated"


def test_recurrent_core_single_iteration():
    """Test recurrent core with single iteration."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    core = CartRecurrentCore(dim, head_dim, num_iterations=1)
    x_init = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = core(x_init, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_recurrent_core_multiple_iterations():
    """Test recurrent core with multiple iterations."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    core = CartRecurrentCore(dim, head_dim, num_iterations=3)
    x_init = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = core(x_init, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_recurrent_core_stability():
    """Test that recurrent core converges due to LTI gate stability."""
    batch, seq_len, dim, ctx_len, head_dim = 1, 5, 32, 10, 16

    # Use many iterations to check stability
    core = CartRecurrentCore(dim, head_dim, num_iterations=10)

    x_init = torch.ones(batch, seq_len, dim) * 0.1
    context_k = torch.ones(batch, ctx_len, head_dim) * 0.05
    context_v = torch.ones(batch, ctx_len, head_dim) * 0.05

    output = core(x_init, context_k, context_v)

    # Check that output is not NaN or Inf (stability check)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Spectral radius should be < 1
    rho = core.lti_gate.get_spectral_radius()
    assert rho < 1.0, f"Spectral radius {rho} >= 1, recurrence is unstable"


def test_prelude_forward_shape():
    """Test that CartPrelude produces correct K,V shapes."""
    batch, ctx_len, dim, head_dim = 2, 20, 64, 32

    prelude = CartPrelude(dim, head_dim, num_layers=2)
    context = create_dummy_raw_context(batch, ctx_len, dim)

    k, v = prelude(context)

    assert k.shape == (batch, ctx_len, head_dim), f"K shape {k.shape} != {(batch, ctx_len, head_dim)}"
    assert v.shape == (batch, ctx_len, head_dim), f"V shape {v.shape} != {(batch, ctx_len, head_dim)}"


def test_prelude_gradient_flow():
    """Test that gradients flow through CartPrelude."""
    batch, ctx_len, dim, head_dim = 2, 20, 64, 32

    prelude = CartPrelude(dim, head_dim, num_layers=2)
    context = create_dummy_raw_context(batch, ctx_len, dim)
    context.requires_grad = True

    k, v = prelude(context)
    loss = k.sum() + v.sum()
    loss.backward()

    assert context.grad is not None, "Gradient not computed for context"
    assert prelude.encoder[0].weight.grad is not None, "Gradient not computed for encoder"
    assert prelude.k_proj.weight.grad is not None, "Gradient not computed for k_proj"
    assert prelude.v_proj.weight.grad is not None, "Gradient not computed for v_proj"


def test_prelude_multiple_layers():
    """Test CartPrelude with different layer counts."""
    batch, ctx_len, dim, head_dim = 2, 15, 48, 32

    for num_layers in [1, 2, 3, 4]:
        prelude = CartPrelude(dim, head_dim, num_layers=num_layers)
        context = create_dummy_raw_context(batch, ctx_len, dim)

        k, v = prelude(context)

        assert k.shape == (batch, ctx_len, head_dim), f"Layer {num_layers}: K shape mismatch"
        assert v.shape == (batch, ctx_len, head_dim), f"Layer {num_layers}: V shape mismatch"


def test_cart_forward_shape():
    """Test that full Cart model produces correct output shape."""
    batch, seq_len, ctx_len, dim, head_dim = 2, 10, 20, 64, 32

    cart = Cart(dim, head_dim, prelude_layers=2, num_iterations=1)
    x_init = torch.randn(batch, seq_len, dim)
    context = create_dummy_raw_context(batch, ctx_len, dim)

    output = cart(x_init, context)

    assert output.shape == (batch, seq_len, dim), f"Output shape {output.shape} != {(batch, seq_len, dim)}"


def test_cart_gradient_flow():
    """Test that gradients flow through full Cart model."""
    batch, seq_len, ctx_len, dim, head_dim = 2, 10, 20, 64, 32

    cart = Cart(dim, head_dim, prelude_layers=2, num_iterations=1)
    x_init = torch.randn(batch, seq_len, dim, requires_grad=True)
    context = torch.randn(batch, ctx_len, dim, requires_grad=True)

    output = cart(x_init, context)
    loss = output.sum()
    loss.backward()

    assert x_init.grad is not None, "Gradient not computed for x_init"
    assert context.grad is not None, "Gradient not computed for context"
    assert cart.prelude.encoder[0].weight.grad is not None, "Gradient not computed for prelude"
    assert cart.core.mla.q_proj.weight.grad is not None, "Gradient not computed for core"


def test_cart_multiple_iterations():
    """Test Cart with multiple recurrent iterations."""
    batch, seq_len, ctx_len, dim, head_dim = 2, 10, 20, 64, 32

    cart_1iter = Cart(dim, head_dim, prelude_layers=2, num_iterations=1)
    cart_3iter = Cart(dim, head_dim, prelude_layers=2, num_iterations=3)

    x_init = torch.randn(batch, seq_len, dim)
    context = create_dummy_raw_context(batch, ctx_len, dim)

    output_1 = cart_1iter(x_init, context)
    output_3 = cart_3iter(x_init, context)

    assert output_1.shape == (batch, seq_len, dim), "1-iter output shape mismatch"
    assert output_3.shape == (batch, seq_len, dim), "3-iter output shape mismatch"

    # Outputs should differ due to different iteration counts
    assert not torch.allclose(output_1, output_3), "Outputs should differ with different iterations"


def test_cart_end_to_end():
    """End-to-end integration test: prelude → core refinement."""
    batch, seq_len, ctx_len, dim, head_dim = 1, 5, 10, 32, 16

    cart = Cart(dim, head_dim, prelude_layers=2, num_iterations=2)

    x_init = torch.ones(batch, seq_len, dim) * 0.1
    context = torch.ones(batch, ctx_len, dim) * 0.05

    output = cart(x_init, context)

    # Check stability
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Output should differ from input (due to MLA and gate)
    assert not torch.allclose(output, x_init, atol=1e-5), "Output should differ from input"


def test_recurrent_core_layer_norm():
    """Test that CartRecurrentCore has layer norm (Pass 3)."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    core = CartRecurrentCore(dim, head_dim, num_iterations=2)

    # Check that layer norm exists
    assert hasattr(core, 'norm'), "CartRecurrentCore should have layer norm in Pass 3"
    assert isinstance(core.norm, nn.LayerNorm), "norm should be LayerNorm"

    x_init = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = core(x_init, context_k, context_v)

    # Output should be stable
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_recurrent_core_residual_stability():
    """Test that residual connections maintain stability across iterations."""
    batch, seq_len, dim, ctx_len, head_dim = 1, 5, 32, 10, 16

    # Many iterations to test stability of residual + norm
    core = CartRecurrentCore(dim, head_dim, num_iterations=5)

    x_init = torch.ones(batch, seq_len, dim) * 0.1
    context_k = torch.ones(batch, ctx_len, head_dim) * 0.05
    context_v = torch.ones(batch, ctx_len, head_dim) * 0.05

    output = core(x_init, context_k, context_v)

    # Check stability
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_sequence_classifier_forward_shape():
    """Test CartSequenceClassifier forward pass."""
    batch, seq_len, ctx_len, dim, head_dim = 2, 10, 20, 64, 32

    classifier = CartSequenceClassifier(dim, head_dim, prelude_layers=2, num_iterations=2, num_classes=2)
    x_init = torch.randn(batch, seq_len, dim)
    context = create_dummy_raw_context(batch, ctx_len, dim)

    logits = classifier(x_init, context)

    assert logits.shape == (batch, 2), f"Expected shape (batch, 2), got {logits.shape}"


def test_sequence_classifier_gradient_flow():
    """Test that gradients flow through CartSequenceClassifier."""
    batch, seq_len, ctx_len, dim, head_dim = 2, 10, 20, 64, 32

    classifier = CartSequenceClassifier(dim, head_dim, prelude_layers=2, num_iterations=2)
    x_init = torch.randn(batch, seq_len, dim, requires_grad=True)
    context = torch.randn(batch, ctx_len, dim, requires_grad=True)

    logits = classifier(x_init, context)
    loss = logits.sum()
    loss.backward()

    assert x_init.grad is not None, "Gradient not computed for x_init"
    assert context.grad is not None, "Gradient not computed for context"


def test_synthetic_dataset_creation():
    """Test synthetic length classification dataset creation."""
    num_samples, seq_len_range, ctx_len, dim, threshold = 100, (5, 15), 20, 64, 8

    inputs, contexts, labels = create_synthetic_length_classification_dataset(
        num_samples, seq_len_range, ctx_len, dim, threshold
    )

    assert inputs.shape == (num_samples, seq_len_range[1], dim), f"Input shape mismatch: {inputs.shape}"
    assert contexts.shape == (num_samples, ctx_len, dim), f"Context shape mismatch: {contexts.shape}"
    assert labels.shape == (num_samples,), f"Label shape mismatch: {labels.shape}"

    # Labels should be binary
    assert set(labels.tolist()) <= {0, 1}, "Labels should be binary"


def test_classifier_training_step():
    """Test single training step for sequence classifier."""
    batch, seq_len, ctx_len, dim, head_dim = 4, 10, 20, 64, 32

    classifier = CartSequenceClassifier(dim, head_dim, prelude_layers=2, num_iterations=2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    x_init = torch.randn(batch, seq_len, dim)
    context = torch.randn(batch, ctx_len, dim)
    labels = torch.randint(0, 2, (batch,))

    # Record initial loss
    with torch.no_grad():
        initial_logits = classifier(x_init, context)
        initial_loss = loss_fn(initial_logits, labels).item()

    # Training step
    loss_val = train_classifier_step(classifier, x_init, context, labels, optimizer, loss_fn)

    assert isinstance(loss_val, float), "Loss should be a float"
    assert loss_val > 0, "Loss should be positive"


def test_classifier_evaluation():
    """Test evaluation function for sequence classifier."""
    num_samples, seq_len, ctx_len, dim, head_dim = 20, 10, 20, 64, 32

    classifier = CartSequenceClassifier(dim, head_dim, prelude_layers=2, num_iterations=2)

    inputs = torch.randn(num_samples, seq_len, dim)
    contexts = torch.randn(num_samples, ctx_len, dim)
    labels = torch.randint(0, 2, (num_samples,))

    loss, accuracy = evaluate_classifier(classifier, inputs, contexts, labels, batch_size=4)

    assert isinstance(loss, float), "Loss should be a float"
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy should be in [0, 1], got {accuracy}"


def test_classifier_full_pipeline():
    """End-to-end test: create dataset, train, and evaluate."""
    # Create synthetic dataset
    num_train, num_val = 50, 20
    seq_len_range, ctx_len, dim, threshold = (5, 15), 20, 32, 8

    train_inputs, train_contexts, train_labels = create_synthetic_length_classification_dataset(
        num_train, seq_len_range, ctx_len, dim, threshold
    )
    val_inputs, val_contexts, val_labels = create_synthetic_length_classification_dataset(
        num_val, seq_len_range, ctx_len, dim, threshold
    )

    # Create classifier
    classifier = CartSequenceClassifier(dim, head_dim=16, prelude_layers=1, num_iterations=2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Quick training loop (just 3 epochs to verify it runs)
    batch_size = 8
    for epoch in range(3):
        for i in range(0, num_train, batch_size):
            batch_end = min(i + batch_size, num_train)
            batch_inputs = train_inputs[i:batch_end]
            batch_contexts = train_contexts[i:batch_end]
            batch_labels = train_labels[i:batch_end]

            train_classifier_step(classifier, batch_inputs, batch_contexts, batch_labels, optimizer, loss_fn)

    # Evaluate
    val_loss, val_acc = evaluate_classifier(classifier, val_inputs, val_contexts, val_labels)

    assert not torch.isnan(torch.tensor(val_loss)), "Validation loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of range"


def test_cart_language_model_forward_shape():
    """Test CartLanguageModel forward pass shape."""
    batch, seq_len, ctx_len, vocab_size, dim = 2, 10, 8, 256, 64

    lm = CartLanguageModel(vocab_size, dim, head_dim=32, prelude_layers=2, num_iterations=2)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    context_ids = torch.randint(0, vocab_size, (batch, ctx_len))

    logits = lm(input_ids, context_ids)

    assert logits.shape == (batch, seq_len, vocab_size), f"Expected {(batch, seq_len, vocab_size)}, got {logits.shape}"


def test_cart_language_model_gradient_flow():
    """Test that gradients flow through CartLanguageModel."""
    batch, seq_len, ctx_len, vocab_size, dim = 2, 10, 8, 256, 64

    lm = CartLanguageModel(vocab_size, dim, head_dim=32, prelude_layers=2, num_iterations=2)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    context_ids = torch.randint(0, vocab_size, (batch, ctx_len))

    logits = lm(input_ids, context_ids)
    loss = logits.sum()
    loss.backward()

    assert lm.embed.weight.grad is not None, "Gradient not computed for embedding"
    assert lm.output_proj.weight.grad is not None, "Gradient not computed for output projection"
    assert lm.cart.prelude.encoder[0].weight.grad is not None, "Gradient not computed for prelude"


def test_transformer_baseline_forward_shape():
    """Test StandardTransformerBaseline forward pass shape."""
    batch, seq_len, vocab_size, dim = 2, 10, 256, 64

    transformer = StandardTransformerBaseline(vocab_size, dim, num_heads=4, num_layers=1)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits = transformer(input_ids)

    assert logits.shape == (batch, seq_len, vocab_size), f"Expected {(batch, seq_len, vocab_size)}, got {logits.shape}"


def test_transformer_baseline_gradient_flow():
    """Test that gradients flow through StandardTransformerBaseline."""
    batch, seq_len, vocab_size, dim = 2, 10, 256, 64

    transformer = StandardTransformerBaseline(vocab_size, dim, num_heads=4, num_layers=1)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits = transformer(input_ids)
    loss = logits.sum()
    loss.backward()

    assert transformer.embed.weight.grad is not None, "Gradient not computed for embedding"
    assert transformer.output_proj.weight.grad is not None, "Gradient not computed for output projection"


def test_synthetic_language_dataset():
    """Test synthetic language dataset creation."""
    num_samples, seq_len, ctx_len, vocab_size = 50, 10, 8, 256

    input_ids, context_ids, target_ids = create_synthetic_language_dataset(
        num_samples, seq_len, ctx_len, vocab_size
    )

    assert input_ids.shape == (num_samples, seq_len), f"Input shape mismatch: {input_ids.shape}"
    assert context_ids.shape == (num_samples, ctx_len), f"Context shape mismatch: {context_ids.shape}"
    assert target_ids.shape == (num_samples, seq_len), f"Target shape mismatch: {target_ids.shape}"

    # Check all IDs are valid
    assert (input_ids >= 0).all() and (input_ids < vocab_size).all(), "Invalid input IDs"
    assert (context_ids >= 0).all() and (context_ids < vocab_size).all(), "Invalid context IDs"
    assert (target_ids >= 0).all() and (target_ids < vocab_size).all(), "Invalid target IDs"


def test_count_parameters():
    """Test parameter counting."""
    vocab_size, dim = 256, 64

    cart_lm = CartLanguageModel(vocab_size, dim, head_dim=32, prelude_layers=2, num_iterations=2)
    transformer = StandardTransformerBaseline(vocab_size, dim, num_heads=4, num_layers=1)

    cart_params = count_parameters(cart_lm)
    transformer_params = count_parameters(transformer)

    assert cart_params > 0, "CART should have parameters"
    assert transformer_params > 0, "Transformer should have parameters"
    assert isinstance(cart_params, int), "Parameter count should be int"
    assert isinstance(transformer_params, int), "Parameter count should be int"


def test_lm_training_step():
    """Test single training step for language model."""
    batch, seq_len, ctx_len, vocab_size, dim = 4, 10, 8, 256, 64

    lm = CartLanguageModel(vocab_size, dim, head_dim=32, prelude_layers=2, num_iterations=2)
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    context_ids = torch.randint(0, vocab_size, (batch, ctx_len))
    target_ids = torch.randint(0, vocab_size, (batch, seq_len))

    loss = train_lm_step(lm, input_ids, context_ids, target_ids, optimizer, loss_fn, is_cart=True)

    assert isinstance(loss, float), "Loss should be a float"
    assert loss > 0, "Loss should be positive"


def test_lm_evaluation():
    """Test language model evaluation."""
    num_samples, seq_len, ctx_len, vocab_size, dim = 20, 10, 8, 256, 64

    lm = CartLanguageModel(vocab_size, dim, head_dim=32, prelude_layers=2, num_iterations=2)

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    context_ids = torch.randint(0, vocab_size, (num_samples, ctx_len))
    target_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    loss = evaluate_lm(lm, input_ids, context_ids, target_ids, batch_size=4, is_cart=True)

    assert isinstance(loss, float), "Loss should be a float"
    assert loss > 0, "Loss should be positive"


def test_lm_end_to_end_training():
    """End-to-end test: create dataset, train CART LM, and evaluate."""
    vocab_size, dim = 128, 32
    num_train, num_val = 30, 10

    # Create dataset
    train_input_ids, train_context_ids, train_target_ids = create_synthetic_language_dataset(
        num_train, 8, 4, vocab_size
    )
    val_input_ids, val_context_ids, val_target_ids = create_synthetic_language_dataset(
        num_val, 8, 4, vocab_size
    )

    # Create model
    lm = CartLanguageModel(vocab_size, dim, head_dim=16, prelude_layers=1, num_iterations=1)
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Quick training
    batch_size = 4
    for epoch in range(2):
        for i in range(0, num_train, batch_size):
            batch_end = min(i + batch_size, num_train)
            batch_input = train_input_ids[i:batch_end]
            batch_context = train_context_ids[i:batch_end]
            batch_target = train_target_ids[i:batch_end]

            train_lm_step(lm, batch_input, batch_context, batch_target, optimizer, loss_fn, is_cart=True)

    # Evaluate
    val_loss = evaluate_lm(lm, val_input_ids, val_context_ids, val_target_ids, is_cart=True)

    assert not torch.isnan(torch.tensor(val_loss)), "Validation loss is NaN"
    assert val_loss > 0, "Validation loss should be positive"


if __name__ == "__main__":
    # Run Pass 1 tests
    test_mla_block_forward_shape()
    print("✓ test_mla_block_forward_shape")

    test_mla_block_gradient_flow()
    print("✓ test_mla_block_gradient_flow")

    test_lti_gate_spectral_radius()
    print("✓ test_lti_gate_spectral_radius")

    test_lti_gate_clipping()
    print("✓ test_lti_gate_clipping")

    test_lti_gate_trainable()
    print("✓ test_lti_gate_trainable")

    test_recurrent_core_single_iteration()
    print("✓ test_recurrent_core_single_iteration")

    test_recurrent_core_multiple_iterations()
    print("✓ test_recurrent_core_multiple_iterations")

    test_recurrent_core_stability()
    print("✓ test_recurrent_core_stability")

    # Run Pass 2 tests
    test_prelude_forward_shape()
    print("✓ test_prelude_forward_shape")

    test_prelude_gradient_flow()
    print("✓ test_prelude_gradient_flow")

    test_prelude_multiple_layers()
    print("✓ test_prelude_multiple_layers")

    test_cart_forward_shape()
    print("✓ test_cart_forward_shape")

    test_cart_gradient_flow()
    print("✓ test_cart_gradient_flow")

    test_cart_multiple_iterations()
    print("✓ test_cart_multiple_iterations")

    test_cart_end_to_end()
    print("✓ test_cart_end_to_end")

    # Run Pass 3 tests
    test_recurrent_core_layer_norm()
    print("✓ test_recurrent_core_layer_norm")

    test_recurrent_core_residual_stability()
    print("✓ test_recurrent_core_residual_stability")

    test_sequence_classifier_forward_shape()
    print("✓ test_sequence_classifier_forward_shape")

    test_sequence_classifier_gradient_flow()
    print("✓ test_sequence_classifier_gradient_flow")

    test_synthetic_dataset_creation()
    print("✓ test_synthetic_dataset_creation")

    test_classifier_training_step()
    print("✓ test_classifier_training_step")

    test_classifier_evaluation()
    print("✓ test_classifier_evaluation")

    test_classifier_full_pipeline()
    print("✓ test_classifier_full_pipeline")

    # Run Pass 4 tests
    test_cart_language_model_forward_shape()
    print("✓ test_cart_language_model_forward_shape")

    test_cart_language_model_gradient_flow()
    print("✓ test_cart_language_model_gradient_flow")

    test_transformer_baseline_forward_shape()
    print("✓ test_transformer_baseline_forward_shape")

    test_transformer_baseline_gradient_flow()
    print("✓ test_transformer_baseline_gradient_flow")

    test_synthetic_language_dataset()
    print("✓ test_synthetic_language_dataset")

    test_count_parameters()
    print("✓ test_count_parameters")

    test_lm_training_step()
    print("✓ test_lm_training_step")

    test_lm_evaluation()
    print("✓ test_lm_evaluation")

    test_lm_end_to_end_training()
    print("✓ test_lm_end_to_end_training")

    print("\nAll tests passed!")
