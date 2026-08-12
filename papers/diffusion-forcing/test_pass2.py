import torch
import numpy as np
from diffusion_forcing import DiffusionForcing, SyntheticDataGenerator


def test_training_loop_simple():
    """Test that the training loop runs without errors."""
    token_dim = 8
    batch_size = 4
    seq_len = 6

    df = DiffusionForcing(token_dim)

    # Create a single batch of data
    x_0 = torch.randn(batch_size, seq_len, token_dim)
    data_loader = [x_0]

    # Train for 1 epoch
    losses = df.train(data_loader, num_epochs=1, learning_rate=1e-3)

    assert len(losses) == 1, f"Expected 1 epoch, got {len(losses)}"
    assert losses[0] > 0, "Loss should be positive"
    assert not np.isnan(losses[0]), "Loss contains NaN"
    print(f"✓ Training loop test passed, loss: {losses[0]:.6f}")


def test_loss_decreases():
    """Test that loss decreases over training epochs."""
    token_dim = 8
    batch_size = 4
    seq_len = 6

    df = DiffusionForcing(token_dim)

    # Create dataset with repeating batches
    x_0 = torch.randn(batch_size, seq_len, token_dim)
    data_loader = [x_0] * 5  # Repeat same batch 5 times to ensure learning signal

    # Train for multiple epochs
    losses = df.train(data_loader, num_epochs=10, learning_rate=1e-2)

    assert len(losses) == 10, f"Expected 10 epochs, got {len(losses)}"

    # Check that loss generally decreases
    final_loss = losses[-1]
    initial_loss = losses[0]

    assert final_loss < initial_loss, \
        f"Final loss {final_loss} should be less than initial loss {initial_loss}"

    print(f"✓ Loss decreases test passed")
    print(f"  Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
    print(f"  Improvement: {100 * (1 - final_loss / initial_loss):.1f}%")


def test_random_sequence_training():
    """Test training on random sequences."""
    token_dim = 16
    batch_size = 8
    seq_len = 10
    num_samples = 50

    df = DiffusionForcing(token_dim)

    # Generate random sequences
    data = SyntheticDataGenerator.random_sequences(num_samples, seq_len, token_dim)

    # Create data loader (batches)
    data_loader = [data[i:i+batch_size] for i in range(0, num_samples, batch_size)]

    # Train
    losses = df.train(data_loader, num_epochs=5, learning_rate=1e-3)

    assert len(losses) == 5, f"Expected 5 epochs, got {len(losses)}"
    assert all(l > 0 for l in losses), "All losses should be positive"
    assert not any(np.isnan(l) for l in losses), "No NaN values should appear"

    print(f"✓ Random sequence training test passed")
    print(f"  Losses over epochs: {[f'{l:.4f}' for l in losses]}")


def test_repeating_pattern_training():
    """Test training on repeating pattern sequences."""
    token_dim = 8
    seq_len = 12
    num_samples = 30

    df = DiffusionForcing(token_dim)

    # Generate repeating pattern sequences
    data = SyntheticDataGenerator.repeating_pattern_sequences(
        num_samples, seq_len, token_dim, pattern_len=4
    )

    # Create data loader
    data_loader = [data[i:i+8] for i in range(0, num_samples, 8)]

    # Train
    losses = df.train(data_loader, num_epochs=8, learning_rate=1e-3)

    assert len(losses) == 8, f"Expected 8 epochs, got {len(losses)}"
    print(f"✓ Repeating pattern training test passed")
    print(f"  First epoch loss: {losses[0]:.6f}, Last epoch loss: {losses[-1]:.6f}")


def test_sine_wave_training():
    """Test training on sine wave sequences."""
    token_dim = 8
    seq_len = 20
    num_samples = 40

    df = DiffusionForcing(token_dim)

    # Generate sine wave sequences
    data = SyntheticDataGenerator.sine_wave_sequences(
        num_samples, seq_len, token_dim, freq=2.0
    )

    # Create data loader
    data_loader = [data[i:i+8] for i in range(0, num_samples, 8)]

    # Train
    losses = df.train(data_loader, num_epochs=6, learning_rate=1e-3)

    assert len(losses) == 6, f"Expected 6 epochs, got {len(losses)}"
    print(f"✓ Sine wave training test passed")
    print(f"  Loss curve: {[f'{l:.4f}' for l in losses]}")


def test_denoiser_learns():
    """Test that the denoiser actually learns to denoise better after training."""
    token_dim = 8
    batch_size = 4
    seq_len = 6

    # Create two identical models
    df_before = DiffusionForcing(token_dim)
    df_after = DiffusionForcing(token_dim)

    # Copy weights so they start the same
    df_after.denoiser.load_state_dict(df_before.denoiser.state_dict())

    # Create data
    x_0 = torch.randn(batch_size, seq_len, token_dim)

    # Measure error before training
    t = torch.ones(batch_size) * 0.5
    x_t, _, _ = df_before.forward_diffusion(x_0, t)
    pred_before = df_before.denoiser(x_t, t)
    error_before = torch.nn.functional.mse_loss(pred_before, x_0).item()

    # Train the after model
    data_loader = [x_0] * 10
    df_after.train(data_loader, num_epochs=20, learning_rate=1e-2)

    # Measure error after training
    with torch.no_grad():
        x_t_after, _, _ = df_after.forward_diffusion(x_0, t)
        pred_after = df_after.denoiser(x_t_after, t)
        error_after = torch.nn.functional.mse_loss(pred_after, x_0).item()

    assert error_after < error_before, \
        f"Error should decrease: before={error_before:.6f}, after={error_after:.6f}"

    print(f"✓ Denoiser learns test passed")
    print(f"  Error before training: {error_before:.6f}")
    print(f"  Error after training: {error_after:.6f}")
    print(f"  Improvement: {100 * (1 - error_after / error_before):.1f}%")


def test_compute_loss():
    """Test the loss computation method."""
    token_dim = 8
    batch_size = 4
    seq_len = 6

    df = DiffusionForcing(token_dim)

    # Create clean data
    x_0 = torch.randn(batch_size, seq_len, token_dim)
    t = torch.ones(batch_size) * 0.5

    # Compute loss
    loss = df.compute_loss(x_0, t)

    assert loss.item() > 0, "Loss should be positive"
    assert not np.isnan(loss.item()), "Loss should not be NaN"
    print(f"✓ Compute loss test passed, loss: {loss.item():.6f}")


if __name__ == "__main__":
    test_training_loop_simple()
    test_loss_decreases()
    test_random_sequence_training()
    test_repeating_pattern_training()
    test_sine_wave_training()
    test_denoiser_learns()
    test_compute_loss()
    print("\n✅ All pass 2 tests passed!")
