import torch
from genception import (
    GenCeption, SimpleTextEmbedding, SimpleUNetBackbone, InstructionConditioner,
    DepthHead, NormalsHead, SegmentationHead, TaskSelector
)


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


def test_task_selector():
    """Test that task selector produces correct output shapes."""
    batch_size = 4
    instruction_dim = 128
    num_tasks = 3

    selector = TaskSelector(instruction_dim, num_tasks)

    instruction_vec = torch.randn(batch_size, instruction_dim)
    task_logits, task_ids = selector(instruction_vec)

    assert task_logits.shape == (batch_size, num_tasks), \
        f"Expected task_logits shape {(batch_size, num_tasks)}, got {task_logits.shape}"
    assert task_ids.shape == (batch_size,), \
        f"Expected task_ids shape {(batch_size,)}, got {task_ids.shape}"
    assert (task_ids >= 0).all() and (task_ids < num_tasks).all(), \
        f"Task IDs should be in range [0, {num_tasks})"
    print("✓ Task selector output shapes are correct")


def test_depth_head():
    """Test that depth head produces correct output shape."""
    batch_size = 2
    in_channels = 32
    height, width = 64, 64

    depth_head = DepthHead(in_channels)

    features = torch.randn(batch_size, in_channels, height, width)
    depth_out = depth_head(features)

    assert depth_out.shape == (batch_size, 1, height, width), \
        f"Expected depth output shape {(batch_size, 1, height, width)}, got {depth_out.shape}"
    print("✓ Depth head output shape is correct")


def test_normals_head():
    """Test that normals head produces correct output shape."""
    batch_size = 2
    in_channels = 32
    height, width = 64, 64

    normals_head = NormalsHead(in_channels)

    features = torch.randn(batch_size, in_channels, height, width)
    normals_out = normals_head(features)

    assert normals_out.shape == (batch_size, 3, height, width), \
        f"Expected normals output shape {(batch_size, 3, height, width)}, got {normals_out.shape}"
    print("✓ Normals head output shape is correct")


def test_segmentation_head():
    """Test that segmentation head produces correct output shape."""
    batch_size = 2
    in_channels = 32
    num_classes = 10
    height, width = 64, 64

    seg_head = SegmentationHead(in_channels, num_classes=num_classes)

    features = torch.randn(batch_size, in_channels, height, width)
    seg_out = seg_head(features)

    assert seg_out.shape == (batch_size, num_classes, height, width), \
        f"Expected segmentation output shape {(batch_size, num_classes, height, width)}, got {seg_out.shape}"
    print("✓ Segmentation head output shape is correct")


def test_genception_multi_task():
    """Test that GenCeption produces correct multi-task outputs."""
    batch_size = 2
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16
    num_classes = 10

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3, num_classes=num_classes)
    model.eval()

    image = torch.randn(batch_size, 3, height, width)
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))

    with torch.no_grad():
        output_dict = model(image, token_ids, return_all_tasks=True)

    # Check output dictionary structure
    assert "task_id" in output_dict, "Output should have task_id"
    assert "task_logits" in output_dict, "Output should have task_logits"
    assert "depth" in output_dict, "Output should have depth"
    assert "normals" in output_dict, "Output should have normals"
    assert "segmentation" in output_dict, "Output should have segmentation"

    # Check shapes
    assert output_dict["task_id"].shape == (batch_size,), \
        f"Expected task_id shape {(batch_size,)}, got {output_dict['task_id'].shape}"
    assert output_dict["task_logits"].shape == (batch_size, 3), \
        f"Expected task_logits shape {(batch_size, 3)}, got {output_dict['task_logits'].shape}"
    assert output_dict["depth"].shape == (batch_size, 1, height, width), \
        f"Expected depth shape {(batch_size, 1, height, width)}, got {output_dict['depth'].shape}"
    assert output_dict["normals"].shape == (batch_size, 3, height, width), \
        f"Expected normals shape {(batch_size, 3, height, width)}, got {output_dict['normals'].shape}"
    assert output_dict["segmentation"].shape == (batch_size, num_classes, height, width), \
        f"Expected segmentation shape {(batch_size, num_classes, height, width)}, got {output_dict['segmentation'].shape}"

    print("✓ GenCeption multi-task outputs have correct shapes")


def test_genception_task_selection():
    """Test that different instructions select different tasks."""
    batch_size = 8
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16
    num_classes = 10

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3, num_classes=num_classes)
    model.eval()

    image = torch.randn(batch_size, 3, height, width)

    # Create several different instruction sets
    token_ids_list = [
        torch.randint(0, vocab_size, (batch_size, 10))
        for _ in range(3)
    ]

    task_ids_per_instruction = []

    with torch.no_grad():
        for token_ids in token_ids_list:
            output_dict = model(image, token_ids, return_all_tasks=True)
            task_ids = output_dict["task_id"]
            task_ids_per_instruction.append(task_ids)

    # Verify that task selection is based on instruction
    # (different instructions should produce some variation in task selection across batch)
    all_task_ids = torch.cat(task_ids_per_instruction, dim=0)
    unique_tasks = torch.unique(all_task_ids)
    assert len(unique_tasks) > 1, "Task selection should vary across different instructions"
    print(f"✓ Task selection varies: {len(unique_tasks)} different tasks selected across instructions")


def test_genception_determinism_pass2():
    """Test that same inputs produce same outputs in Pass 2."""
    batch_size = 2
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16
    num_classes = 10

    model = GenCeption(vocab_size=vocab_size, embed_dim=embed_dim,
                       base_channels=base_channels, num_blocks=3, num_classes=num_classes)
    model.eval()

    image = torch.randn(batch_size, 3, height, width)
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))

    with torch.no_grad():
        output_dict_1 = model(image, token_ids, return_all_tasks=True)
        output_dict_2 = model(image, token_ids, return_all_tasks=True)

    # Check determinism
    assert torch.allclose(output_dict_1["depth"], output_dict_2["depth"]), \
        "Depth outputs should be deterministic"
    assert torch.allclose(output_dict_1["normals"], output_dict_2["normals"]), \
        "Normals outputs should be deterministic"
    assert torch.allclose(output_dict_1["segmentation"], output_dict_2["segmentation"]), \
        "Segmentation outputs should be deterministic"
    assert torch.allclose(output_dict_1["task_logits"], output_dict_2["task_logits"]), \
        "Task logits should be deterministic"

    print("✓ GenCeption Pass 2 outputs are deterministic in eval mode")


if __name__ == "__main__":
    print("Running GenCeption Pass 1 tests...\n")

    test_text_embedding()
    test_unet_backbone()
    test_instruction_conditioner()
    test_genception_forward()
    test_genception_instruction_effects()
    test_genception_determinism()

    print("\nRunning GenCeption Pass 2 tests...\n")

    test_task_selector()
    test_depth_head()
    test_normals_head()
    test_segmentation_head()
    test_genception_multi_task()
    test_genception_task_selection()
    test_genception_determinism_pass2()

    print("\n✓ All tests passed!")
