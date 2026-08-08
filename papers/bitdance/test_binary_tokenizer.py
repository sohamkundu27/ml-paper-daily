"""Test the binary tokenizer."""

import torch
import torch.optim as optim
from binary_tokenizer import BinaryTokenizer, BinaryTokenizerLoss


def test_tokenizer_shapes():
    """Test that encoder/decoder produce expected tensor shapes."""
    model = BinaryTokenizer(in_channels=3, token_dim=32, num_tokens_spatial=8)
    model.eval()

    # Create dummy image batch: 2 images of 32x32
    batch_size = 2
    x = torch.rand(batch_size, 3, 32, 32)

    with torch.no_grad():
        tokens, binary_tokens = model.encode(x)
        reconstructed = model.decode(binary_tokens)

    # Check shapes
    assert tokens.shape == (batch_size, 32, 4, 4), f"Expected (2, 32, 4, 4), got {tokens.shape}"
    assert binary_tokens.shape == (batch_size, 32, 4, 4), f"Expected (2, 32, 4, 4), got {binary_tokens.shape}"
    assert reconstructed.shape == (batch_size, 3, 32, 32), f"Expected (2, 3, 32, 32), got {reconstructed.shape}"

    # Check that reconstructed is in [0, 1]
    assert reconstructed.min() >= 0.0, f"Min value: {reconstructed.min()}"
    assert reconstructed.max() <= 1.0, f"Max value: {reconstructed.max()}"

    # Check that binary tokens are actually binary (0 or 1)
    assert torch.all((binary_tokens == 0) | (binary_tokens == 1)), "Binary tokens should be 0 or 1"

    print("✓ Shape test passed")


def test_tokenizer_reconstruction():
    """Test that tokenizer can be trained to reconstruct images."""
    model = BinaryTokenizer(in_channels=3, token_dim=32, num_tokens_spatial=8)
    loss_fn = BinaryTokenizerLoss(reconstruction_weight=1.0, perplexity_weight=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create simple synthetic image: solid red
    batch_size = 4
    x = torch.zeros(batch_size, 3, 32, 32)
    x[:, 0, :, :] = 0.8  # Red channel

    initial_losses = []
    final_losses = []

    # Train for a few steps
    for step in range(20):
        optimizer.zero_grad()

        # Forward pass
        tokens, binary_tokens = model.encode(x)
        x_recon = model.decode(binary_tokens)

        # Compute loss
        loss, recon_loss, binary_loss = loss_fn(x, x_recon, tokens)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_losses.append(float(recon_loss.detach()))
        if step == 19:
            final_losses.append(float(recon_loss.detach()))

    # Check that loss decreased
    assert final_losses[0] < initial_losses[0], (
        f"Loss should decrease after training. Initial: {initial_losses[0]:.6f}, Final: {final_losses[0]:.6f}"
    )

    print(f"✓ Reconstruction test passed (loss: {initial_losses[0]:.6f} -> {final_losses[0]:.6f})")


def test_binary_tokens_are_binary():
    """Test that encoding produces actual binary tokens."""
    model = BinaryTokenizer(in_channels=3, token_dim=16, num_tokens_spatial=4)
    model.eval()

    # Create a simple gradient image
    x = torch.linspace(0, 1, 16 * 16).view(1, 1, 16, 16).expand(1, 3, 16, 16)

    with torch.no_grad():
        tokens, binary_tokens = model.encode(x)

    # All values should be exactly 0 or 1
    unique_vals = torch.unique(binary_tokens)
    assert len(unique_vals) <= 2, f"Binary tokens should have at most 2 unique values, got {len(unique_vals)}"
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"Binary tokens should be 0 or 1, got {unique_vals}"

    print("✓ Binary tokens test passed")


def test_encode_decode_consistency():
    """Test that encoding and then decoding maintains rough consistency."""
    model = BinaryTokenizer(in_channels=3, token_dim=32, num_tokens_spatial=8)
    model.eval()

    # Create a test image with distinct regions
    x = torch.zeros(1, 3, 32, 32)
    x[:, 0, :16, :] = 0.9  # Top half red
    x[:, 1, 16:, :] = 0.9  # Bottom half green

    with torch.no_grad():
        tokens, binary_tokens = model.encode(x)
        x_recon = model.decode(binary_tokens)

    # Reconstruction should be somewhat similar to original (not exact, but reasonable)
    mse = torch.mean((x - x_recon) ** 2)
    assert mse < 0.5, f"Reconstruction MSE too high: {mse}"

    print(f"✓ Encode-decode consistency test passed (MSE: {mse:.6f})")


if __name__ == "__main__":
    test_tokenizer_shapes()
    test_binary_tokens_are_binary()
    test_encode_decode_consistency()
    test_tokenizer_reconstruction()
    print("\n✅ All tests passed!")
