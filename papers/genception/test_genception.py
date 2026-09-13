import torch
from genception import GenCeption, SimpleTextEmbedding, SimpleUNetBackbone, InstructionConditioner


def test_text_embedding():
    """Test that text embedding produces consistent output shapes."""
    vocab_size = 100
    embed_dim = 64
    batch_size = 4
    seq_len = 10

    embedding = SimpleTextEmbedding(vocab_size, embed_dim)

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding(token_ids)

    assert output.shape == (batch_size, embed_dim), f"Expected {(batch_size, embed_dim)}, got {output.shape}"
    print("✓ Text embedding shape is correct")


def test_unet_backbone():
    """Test that UNet backbone processes images correctly."""
    batch_size = 2
    height, width = 64, 64
    base_channels = 16
    num_blocks = 3

    backbone = SimpleUNetBackbone(in_channels=3, base_channels=base_channels, num_blocks=num_blocks)

    image = torch.randn(batch_size, 3, height, width)
    features = backbone(image)

    assert features.shape == (batch_size, base_channels, height, width), \
        f"Expected {(batch_size, base_channels, height, width)}, got {features.shape}"
    print("✓ UNet backbone output shape is correct")


def test_instruction_conditioner():
    """Test that instruction conditioning works correctly."""
    batch_size = 2
    channels = 32
    height, width = 64, 64
    instruction_dim = 128

    conditioner = InstructionConditioner(channels, instruction_dim)

    image_features = torch.randn(batch_size, channels, height, width)
    instruction_vec = torch.randn(batch_size, instruction_dim)

    output = conditioner(image_features, instruction_vec)

    assert output.shape == image_features.shape, \
        f"Expected {image_features.shape}, got {output.shape}"
    print("✓ Instruction conditioner output shape is correct")

    # Verify that output differs from input (instruction had an effect)
    assert not torch.allclose(output, image_features), \
        "Conditioned features should differ from input features"
    print("✓ Instruction conditioning modulates features")


def test_genception_forward():
    """Test full GenCeption model forward pass."""
    batch_size = 2
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3)

    image = torch.randn(batch_size, 3, height, width)
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))

    features = model(image, token_ids)

    assert features.shape == (batch_size, base_channels, height, width), \
        f"Expected {(batch_size, base_channels, height, width)}, got {features.shape}"
    print("✓ GenCeption forward pass output shape is correct")


def test_genception_instruction_effects():
    """Test that different instructions produce different outputs."""
    batch_size = 2
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3)
    model.eval()

    # Use same image but different instructions
    image = torch.randn(batch_size, 3, height, width)
    token_ids_1 = torch.randint(0, vocab_size, (batch_size, 10))
    token_ids_2 = torch.randint(0, vocab_size, (batch_size, 10))

    with torch.no_grad():
        features_1 = model(image, token_ids_1)
        features_2 = model(image, token_ids_2)

    # Different instructions should (almost certainly) produce different features
    assert not torch.allclose(features_1, features_2, rtol=1e-3), \
        "Different instructions should produce different features"
    print("✓ Different instructions produce different conditioned features")


def test_genception_determinism():
    """Test that same inputs produce same outputs (determinism)."""
    batch_size = 2
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3)
    model.eval()

    image = torch.randn(batch_size, 3, height, width)
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))

    with torch.no_grad():
        features_1 = model(image, token_ids)
        features_2 = model(image, token_ids)

    assert torch.allclose(features_1, features_2), \
        "Same inputs should produce identical outputs in eval mode"
    print("✓ Model is deterministic in eval mode")


if __name__ == "__main__":
    print("Running GenCeption Pass 1 tests...\n")

    test_text_embedding()
    test_unet_backbone()
    test_instruction_conditioner()
    test_genception_forward()
    test_genception_instruction_effects()
    test_genception_determinism()

    print("\n✓ All tests passed!")
