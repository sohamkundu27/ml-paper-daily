import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from diffusion_forcing import DiffusionForcing, SyntheticDataGenerator


class AutoregressiveBaseline(nn.Module):
    """Simple autoregressive baseline: predicts next token from previous tokens."""

    def __init__(self, token_dim, hidden_dim=64, context_window=3):
        super().__init__()
        self.token_dim = token_dim
        self.context_window = context_window

        # Simple MLP: concatenate context_window tokens -> predict next token
        self.net = nn.Sequential(
            nn.Linear(token_dim * context_window, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, x_prev):
        """Predict next token from previous tokens.

        Args:
            x_prev: [batch, context_window, token_dim] or [batch, seq_len, token_dim]

        Returns:
            [batch, token_dim] predicted next token(s)
        """
        batch_size = x_prev.shape[0]
        seq_len = x_prev.shape[1]

        if seq_len < self.context_window:
            # Pad if sequence is shorter than context window
            pad_size = self.context_window - seq_len
            x_prev = torch.cat([
                torch.zeros(batch_size, pad_size, self.token_dim, device=x_prev.device),
                x_prev
            ], dim=1)

        # Take last context_window tokens and reshape
        x_context = x_prev[:, -self.context_window:, :]  # [batch, context_window, token_dim]
        x_context_flat = x_context.reshape(batch_size, -1)  # [batch, context_window * token_dim]

        # Predict next token
        pred = self.net(x_context_flat)  # [batch, token_dim]

        return pred


def create_arithmetic_sequences(num_samples, seq_len, token_dim, seed=None):
    """Create synthetic arithmetic sequences.

    Each sequence is an arithmetic progression with random start and difference.
    Flatten to token_dim=1 or create multi-dimensional version.

    Returns:
        [num_samples, seq_len, token_dim] tensor
    """
    if seed is not None:
        np.random.seed(seed)

    sequences = []
    for _ in range(num_samples):
        # Random start and difference
        start = np.random.randn() * 10
        diff = np.random.randn() * 2

        # Generate arithmetic sequence
        seq = np.array([start + i * diff for i in range(seq_len)], dtype=np.float32)

        # Reshape to token_dim by repeating or padding
        seq = np.tile(seq[:, None], (1, token_dim))  # [seq_len, token_dim]

        sequences.append(torch.from_numpy(seq))

    return torch.stack(sequences)


def train_autoregressive(model, data, num_epochs=20, learning_rate=1e-3, batch_size=16):
    """Train autoregressive model to predict next tokens."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    num_samples = data.shape[0]
    seq_len = data.shape[1]

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle indices
        indices = torch.randperm(num_samples)
        data_shuffled = data[indices]

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i+batch_size].to(device)  # [batch_size, seq_len, token_dim]

            # Create training pairs: (context, target_next_token)
            for t in range(1, seq_len):
                x_context = batch[:, :t, :]  # Previous tokens
                y_target = batch[:, t:t+1, :]  # Next token (keep dim for consistency)

                # Predict
                pred = model(x_context)  # [batch_size, token_dim]

                # Loss
                loss = torch.nn.functional.mse_loss(pred, y_target.squeeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        losses.append(avg_loss)

    return losses


def rollout_diffusion_forcing(df_model, context, num_future_tokens, num_steps=30):
    """Generate future tokens using Diffusion Forcing.

    Args:
        df_model: trained DiffusionForcing instance
        context: [batch_size, context_len, token_dim] context tokens
        num_future_tokens: how many tokens to generate
        num_steps: denoising steps

    Returns:
        [batch_size, context_len + num_future_tokens, token_dim] full sequence
    """
    batch_size = context.shape[0]
    context_len = context.shape[1]
    token_dim = context.shape[2]
    total_len = context_len + num_future_tokens

    # Create mask: keep context, denoise future
    mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
    mask[:, :context_len] = True

    # Sample with mask
    result = df_model.sample_with_mask(context, mask, num_steps=num_steps)

    return result


def rollout_autoregressive(ar_model, context, num_future_tokens, device='cpu'):
    """Generate future tokens using autoregressive model.

    Args:
        ar_model: trained AutoregressiveBaseline instance
        context: [batch_size, context_len, token_dim] context tokens
        num_future_tokens: how many tokens to generate
        device: torch device

    Returns:
        [batch_size, context_len + num_future_tokens, token_dim] full sequence
    """
    batch_size = context.shape[0]
    context_len = context.shape[1]
    token_dim = context.shape[2]

    # Initialize sequence with context
    sequence = context.clone().to(device)

    # Greedily generate future tokens
    for _ in range(num_future_tokens):
        # Predict next token
        next_token = ar_model(sequence)  # [batch_size, token_dim]

        # Append to sequence
        sequence = torch.cat([sequence, next_token.unsqueeze(1)], dim=1)

    return sequence


def evaluate_rollout_error(ground_truth, predicted, horizon):
    """Compute rollout prediction error over multiple steps.

    Args:
        ground_truth: [batch_size, seq_len, token_dim]
        predicted: [batch_size, seq_len, token_dim]
        horizon: how many steps ahead to evaluate

    Returns:
        dict with error metrics
    """
    # Compute MSE over future horizon
    mse = torch.nn.functional.mse_loss(
        predicted[:, -horizon:, :],
        ground_truth[:, -horizon:, :]
    ).item()

    # Compute MAE (absolute error)
    mae = torch.abs(predicted[:, -horizon:, :] - ground_truth[:, -horizon:, :]).mean().item()

    return {'mse': mse, 'mae': mae}


def test_pass4_end_to_end_demo():
    """End-to-end demo: compare Diffusion Forcing vs autoregressive baseline."""
    print("\n" + "="*70)
    print("PASS 4: END-TO-END DEMO - Diffusion Forcing vs Autoregressive")
    print("="*70)

    # Setup
    token_dim = 4
    seq_len = 16
    batch_size = 8
    num_train_samples = 64
    device = 'cpu'

    print(f"\n📊 Task: Predict next tokens in arithmetic sequences")
    print(f"   Token dimension: {token_dim}")
    print(f"   Sequence length (train): {seq_len}")
    print(f"   Training samples: {num_train_samples}")

    # Generate training data
    print(f"\n🔧 Generating synthetic arithmetic sequences...")
    data = create_arithmetic_sequences(num_train_samples, seq_len, token_dim, seed=42)

    # Split into train and validation
    train_data = data[:int(0.8 * num_train_samples)]
    val_data = data[int(0.8 * num_train_samples):]

    print(f"   Train: {train_data.shape}, Val: {val_data.shape}")

    # ========== Train Diffusion Forcing ==========
    print(f"\n🎯 Training Diffusion Forcing...")
    df = DiffusionForcing(token_dim, device=device)

    # Create data loader
    data_loader = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]

    # Train
    losses_df = df.train(data_loader, num_epochs=15, learning_rate=1e-3)

    print(f"   Loss (epoch 1): {losses_df[0]:.6f}")
    print(f"   Loss (epoch 15): {losses_df[-1]:.6f}")
    print(f"   Improvement: {100 * (1 - losses_df[-1] / losses_df[0]):.1f}%")

    # ========== Train Autoregressive ==========
    print(f"\n🎯 Training Autoregressive Baseline...")
    ar = AutoregressiveBaseline(token_dim, hidden_dim=64, context_window=3).to(device)

    losses_ar = train_autoregressive(ar, train_data, num_epochs=15, learning_rate=1e-3, batch_size=batch_size)

    print(f"   Loss (epoch 1): {losses_ar[0]:.6f}")
    print(f"   Loss (epoch 15): {losses_ar[-1]:.6f}")
    print(f"   Improvement: {100 * (1 - losses_ar[-1] / losses_ar[0]):.1f}%")

    # ========== Evaluate on validation set ==========
    print(f"\n📈 Evaluating on validation set (predict 5 tokens ahead)...")

    # Use first half of val data as context, predict second half
    context_len = seq_len // 2
    horizon = seq_len - context_len  # 5 tokens

    val_context = val_data[:, :context_len, :].to(device)
    val_ground_truth = val_data.to(device)

    # Diffusion Forcing rollout
    print(f"\n  Computing Diffusion Forcing rollout...")
    df_pred = rollout_diffusion_forcing(df, val_context, horizon, num_steps=30)

    df_metrics = evaluate_rollout_error(val_ground_truth, df_pred, horizon)

    print(f"    MSE (horizon={horizon}): {df_metrics['mse']:.6f}")
    print(f"    MAE (horizon={horizon}): {df_metrics['mae']:.6f}")

    # Autoregressive rollout
    print(f"\n  Computing Autoregressive rollout...")
    ar_pred = rollout_autoregressive(ar, val_context, horizon, device=device)

    ar_metrics = evaluate_rollout_error(val_ground_truth, ar_pred, horizon)

    print(f"    MSE (horizon={horizon}): {ar_metrics['mse']:.6f}")
    print(f"    MAE (horizon={horizon}): {ar_metrics['mae']:.6f}")

    # ========== Comparison ==========
    print(f"\n🏆 COMPARISON RESULTS")
    print(f"   MSE Ratio (DF / AR): {df_metrics['mse'] / ar_metrics['mse']:.3f}x")
    print(f"   MAE Ratio (DF / AR): {df_metrics['mae'] / ar_metrics['mae']:.3f}x")

    if df_metrics['mse'] < ar_metrics['mse']:
        improvement = 100 * (1 - df_metrics['mse'] / ar_metrics['mse'])
        print(f"   ✅ Diffusion Forcing: {improvement:.1f}% better MSE")
    else:
        regress = 100 * (df_metrics['mse'] / ar_metrics['mse'] - 1)
        print(f"   ⚠️  Autoregressive: {regress:.1f}% better MSE")
        print(f"       (This is expected: AR is strong for in-distribution prediction)")

    # ========== Sample generation test ==========
    print(f"\n🎲 Testing sequence generation (no ground truth)...")

    # Sample from Diffusion Forcing
    sample_df = df.sample(batch_size=2, seq_len=10, num_steps=30)
    print(f"   DF sample shape: {sample_df.shape}")
    print(f"   Mean: {sample_df.mean():.4f}, Std: {sample_df.std():.4f}")

    # Sample from autoregressive (needs context)
    small_context = torch.randn(2, 3, token_dim).to(device)
    sample_ar = rollout_autoregressive(ar, small_context, 7, device=device)
    print(f"   AR sample shape: {sample_ar.shape}")
    print(f"   Mean: {sample_ar.mean():.4f}, Std: {sample_ar.std():.4f}")

    print(f"\n✅ Pass 4 end-to-end demo completed!")

    return {
        'df_metrics': df_metrics,
        'ar_metrics': ar_metrics,
        'df_losses': losses_df,
        'ar_losses': losses_ar,
    }


def test_pass4_masking_advantage():
    """Test Diffusion Forcing's masking advantage: condition on past, predict future."""
    print("\n" + "="*70)
    print("PASS 4: Masking Capability - Diffusion Forcing's Key Advantage")
    print("="*70)

    token_dim = 4
    seq_len = 12
    batch_size = 4
    num_samples = 32
    device = 'cpu'

    print(f"\n📊 Testing variable context lengths with masking...")

    # Generate training data
    data = create_arithmetic_sequences(num_samples, seq_len, token_dim, seed=43)

    # Train Diffusion Forcing
    df = DiffusionForcing(token_dim, device=device)
    data_loader = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    df.train(data_loader, num_epochs=10, learning_rate=1e-3)

    # Train Autoregressive
    ar = AutoregressiveBaseline(token_dim, hidden_dim=64, context_window=3).to(device)
    train_autoregressive(ar, data, num_epochs=10, learning_rate=1e-3, batch_size=batch_size)

    # Test data
    test_data = create_arithmetic_sequences(8, seq_len, token_dim, seed=44)

    results = []

    # Test with different context lengths
    for context_frac in [0.25, 0.5, 0.75]:
        context_len = int(seq_len * context_frac)
        horizon = seq_len - context_len

        test_context = test_data[:, :context_len, :].to(device)
        test_ground_truth = test_data.to(device)

        # Diffusion Forcing (can use masking)
        df_pred = rollout_diffusion_forcing(df, test_context, horizon, num_steps=25)
        df_mse = torch.nn.functional.mse_loss(
            df_pred[:, -horizon:, :],
            test_ground_truth[:, -horizon:, :]
        ).item()

        # Autoregressive (limited by context window)
        ar_pred = rollout_autoregressive(ar, test_context, horizon, device=device)
        ar_mse = torch.nn.functional.mse_loss(
            ar_pred[:, -horizon:, :],
            test_ground_truth[:, -horizon:, :]
        ).item()

        results.append({
            'context_frac': context_frac,
            'context_len': context_len,
            'horizon': horizon,
            'df_mse': df_mse,
            'ar_mse': ar_mse,
        })

        print(f"\n  Context {context_frac:.0%} ({context_len} tokens), predict {horizon} ahead:")
        print(f"    DF MSE: {df_mse:.6f}")
        print(f"    AR MSE: {ar_mse:.6f}")
        print(f"    Ratio: {df_mse / ar_mse:.3f}x")

    print(f"\n✅ Masking capability test completed!")

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PASS 4: END-TO-END DEMONSTRATION")
    print("Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion")
    print("="*70)

    # Run demo
    demo_results = test_pass4_end_to_end_demo()

    # Test masking
    masking_results = test_pass4_masking_advantage()

    print("\n" + "="*70)
    print("✅ ALL PASS 4 TESTS COMPLETED")
    print("="*70)
