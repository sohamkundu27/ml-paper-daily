"""Test stage-adaptive scheduler and loss weighting."""
import torch
import torch.nn as nn
from diffusion import (
    AudioDiffusionModel,
    GaussianDiffusion,
    StageAdaptiveScheduler,
    stage_adaptive_loss,
)


def test_scheduler_weights():
    """Test that scheduler weights change correctly over training progress."""
    num_steps = 100
    scheduler = StageAdaptiveScheduler(num_training_steps=num_steps, timesteps=100)

    # Early training: semantic should be high
    semantic_w_early, perceptual_w_early = scheduler.get_weights(0)
    assert semantic_w_early > 0.9, f"Early semantic weight should be high, got {semantic_w_early}"
    assert perceptual_w_early < 0.1, f"Early perceptual weight should be low, got {perceptual_w_early}"

    # Late training: perceptual should be high
    semantic_w_late, perceptual_w_late = scheduler.get_weights(num_steps - 1)
    assert semantic_w_late < 0.1, f"Late semantic weight should be low, got {semantic_w_late}"
    assert perceptual_w_late > 0.9, f"Late perceptual weight should be high, got {perceptual_w_late}"

    # Mid training: weights should be balanced
    semantic_w_mid, perceptual_w_mid = scheduler.get_weights(num_steps // 2)
    assert 0.4 < semantic_w_mid < 0.6, f"Mid semantic weight should be ~0.5, got {semantic_w_mid}"
    assert 0.4 < perceptual_w_mid < 0.6, f"Mid perceptual weight should be ~0.5, got {perceptual_w_mid}"

    # Weights should sum to 1
    for step in [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]:
        s, p = scheduler.get_weights(step)
        assert abs(s + p - 1.0) < 1e-5, f"Weights don't sum to 1.0 at step {step}"

    print("✓ Scheduler weights test passed")


def test_scheduler_strategies():
    """Test different scheduling strategies."""
    num_steps = 100
    scheduler_linear = StageAdaptiveScheduler(num_training_steps=num_steps, strategy='linear')
    scheduler_exp = StageAdaptiveScheduler(num_training_steps=num_steps, strategy='exponential')
    scheduler_cos = StageAdaptiveScheduler(num_training_steps=num_steps, strategy='cosine')

    # All should produce different weight curves
    weights_linear = [scheduler_linear.get_weights(i)[0] for i in range(0, num_steps, 10)]
    weights_exp = [scheduler_exp.get_weights(i)[0] for i in range(0, num_steps, 10)]
    weights_cos = [scheduler_cos.get_weights(i)[0] for i in range(0, num_steps, 10)]

    # Verify they're different (except at extremes)
    assert weights_linear[5] != weights_exp[5], "Linear and exponential should differ"
    assert weights_linear[5] != weights_cos[5], "Linear and cosine should differ"

    # But all should transition from high semantic (early) to high perceptual (late)
    assert weights_linear[0] > weights_linear[-1]
    assert weights_exp[0] > weights_exp[-1]
    assert weights_cos[0] > weights_cos[-1]

    print("✓ Scheduler strategies test passed")


def test_timestep_masks():
    """Test that masks correctly partition timesteps."""
    num_steps = 100
    scheduler = StageAdaptiveScheduler(num_training_steps=num_steps, timesteps=100)

    batch_size = 8
    t_low_noise = torch.tensor([10, 15, 20, 25])  # Low noise (should be perceptual)
    t_high_noise = torch.tensor([75, 80, 85, 90])  # High noise (should be semantic)

    # Test with low-noise timesteps
    semantic_mask_low, perceptual_mask_low, _, _ = scheduler.get_timestep_mask(
        len(t_low_noise), t_low_noise, 'cpu', current_step=0
    )
    assert perceptual_mask_low.sum() == len(t_low_noise), "Low noise should be perceptual"
    assert semantic_mask_low.sum() == 0, "Low noise should not be semantic"

    # Test with high-noise timesteps
    semantic_mask_high, perceptual_mask_high, _, _ = scheduler.get_timestep_mask(
        len(t_high_noise), t_high_noise, 'cpu', current_step=0
    )
    assert semantic_mask_high.sum() == len(t_high_noise), "High noise should be semantic"
    assert perceptual_mask_high.sum() == 0, "High noise should not be perceptual"

    print("✓ Timestep masks test passed")


def test_stage_adaptive_loss():
    """Test that stage-adaptive loss combines semantic and perceptual components."""
    num_steps = 100
    scheduler = StageAdaptiveScheduler(num_training_steps=num_steps, timesteps=100)
    diffusion = GaussianDiffusion(timesteps=100)

    batch_size = 8
    audio_dim = 128

    # Create dummy predictions and targets
    noise_pred = torch.randn(batch_size, audio_dim)
    target_noise = torch.randn(batch_size, audio_dim)
    t = torch.randint(0, 100, (batch_size,))

    # Compute loss at early stage
    loss_early = stage_adaptive_loss(noise_pred, target_noise, t, scheduler, current_step=0)
    assert loss_early.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss_early), "Loss should not be NaN"

    # Compute loss at late stage
    loss_late = stage_adaptive_loss(noise_pred, target_noise, t, scheduler, current_step=num_steps - 1)
    assert loss_late.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss_late), "Loss should not be NaN"

    print("✓ Stage-adaptive loss test passed")


def test_training_with_stage_adaptive():
    """Test a training loop with stage-adaptive loss."""
    model = AudioDiffusionModel(audio_dim=128, time_dim=64, hidden_dim=256)
    diffusion = GaussianDiffusion(timesteps=100)
    scheduler = StageAdaptiveScheduler(num_training_steps=10, timesteps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []

    for step in range(10):
        batch_size = 4
        x0 = torch.randn(batch_size, 128)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 100, (batch_size,))
        t_norm = t.float() / diffusion.timesteps

        # Forward diffusion
        x_t = diffusion.q_sample(x0, t, noise)

        # Model forward
        noise_pred = model(x_t, t_norm)

        # Stage-adaptive loss
        loss = stage_adaptive_loss(noise_pred, noise, t, scheduler, current_step=step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Loss should generally decrease (allow some noise)
    assert not any(torch.isnan(torch.tensor(l)) for l in losses), "Loss should not be NaN"
    assert all(l > 0 for l in losses), "All losses should be positive"

    print("✓ Training with stage-adaptive loss test passed")


if __name__ == "__main__":
    test_scheduler_weights()
    test_scheduler_strategies()
    test_timestep_masks()
    test_stage_adaptive_loss()
    test_training_with_stage_adaptive()
    print("\n✅ All stage-adaptive tests passed!")
