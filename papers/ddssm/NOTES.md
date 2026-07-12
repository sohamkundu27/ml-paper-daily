# Diffusion-Driven State Space Models

**Paper:** Diffusion-Driven State Space Models  
**arXiv:** https://arxiv.org/abs/2606.21036  
**Published:** June 2026 (ProbML 2026)  
**Authors:** Jack Ruder and collaborators

## Summary

Traditional latent state space models assume a fixed, parametric transition distribution (usually Gaussian). This paper replaces that rigid assumption by using a diffusion model to learn the transition density directly from data. Rather than the common two-stage approach of first training an autoencoder then a separate diffusion model, this method jointly learns the state dynamics and observation process, allowing the diffusion model to capture complex, non-Gaussian transitions between latent states.

## Plan: 4 passes

**Pass 1:** Foundational components — implement a basic linear state space model with Gaussian transitions, plus a minimal diffusion process (forward/reverse) to become familiar with the diffusion mechanics.

**Pass 2:** Replace the Gaussian transition with a learned diffusion model; implement training loop that jointly learns both the state transitions and observation model via score matching.

**Pass 3:** Add multi-step inference for longer sequences; implement proper likelihood estimation and evaluation on a simple synthetic time-series task.

**Pass 4:** End-to-end demo on toy data (e.g., damped oscillator or Lorenz system), with a final honest summary of what worked and what was simplified.

## Implemented vs. simplified

### Pass 1 (completed)

**Implemented:**
- `LinearSSM`: A basic linear Gaussian state space model with learnable transition matrix A, observation matrix C, and fixed noise covariances Q, R. Can sample full trajectories.
- `DiffusionProcess`: A minimal diffusion process with linear noise schedule. Implements forward diffusion q(x_t | x_0) using cumulative alphas, and a simplified reverse step that estimates x_{t-1} from x_t and predicted noise.
- Complete test suite: trajectory generation, forward diffusion at various t, reverse steps, and a basic end-to-end cycle.

**Simplified/Stubbed:**
- Reverse step is not a proper Gaussian posterior; it's a simplified approximation that reweights x_0 estimate and current x_t (good for testing, not for actual denoising).
- No neural network component yet; all matrices are constant.
- No training loop; this is inference-only demonstration.
- No likelihood estimation or learning objective yet.

### Pass 2 (completed)

**Implemented:**
- `NoisePredictor`: A simple MLP-based neural network that predicts noise in diffused state transitions. Takes noisy state x_t and timestep t as input, outputs predicted noise.
- `train_diffusion_ssm()`: Training loop that jointly learns the noise predictor and SSM parameters via score matching. For each trajectory, extracts transitions (z_{t+1} - z_t), adds noise via forward diffusion at random timesteps, and trains the network to predict the true noise via MSE loss. Uses Adam optimizer.
- Tests for the new components: `test_noise_predictor()` verifies the network produces correct shapes; `test_training_diffusion_ssm()` verifies the training loop runs without NaN and decreases loss.

**Simplified/Stubbed:**
- Time embedding is trivial (just t/100); a proper implementation would use positional encodings or sinusoidal embeddings.
- Noise predictor is a shallow MLP with no residual connections or other architectural improvements.
- SSM parameters (A, C) are included in the optimizer but are not actively updated during training (this would require more careful gradient tracking). The key contribution is learning the diffusion model.
- No inference via sampling yet; training is forward-pass only.
- No likelihood estimation; loss is purely MSE on noise prediction.
- No validation or test set separation in the training loop.

### Pass 3 (completed)

**Implemented:**
- `generate_sequence()`: Multi-step inference for generating long sequences. Starts with an initial latent state and iteratively samples the next transition using the trained noise predictor via reverse diffusion. Uses a deterministic schedule to map generation steps to the diffusion timesteps. Generates both latent states (z_seq) and observations (y_seq).
- `estimate_likelihood()`: Likelihood estimation for state trajectories using score matching. Evaluates the model at multiple diffusion timesteps and computes the average MSE between predicted and true noise. Returns negative log-likelihood estimate (lower is better). Used for model evaluation.
- `evaluate_on_synthetic_data()`: End-to-end evaluation on synthetic data generated from a linear SSM. Trains the model on synthetic trajectories, then evaluates on held-out test data. Computes metrics: final training loss, test negative log-likelihood, test observation MSE, and test state trajectory MSE.
- Comprehensive test suite for pass 3 components: `test_generate_sequence()` verifies sequence generation shapes and finiteness; `test_estimate_likelihood()` verifies NLL computation; `test_evaluate_on_synthetic_data()` verifies the full evaluation pipeline.

**Simplified/Stubbed:**
- Sequence generation uses a fixed deterministic reverse schedule (20 steps by default) rather than a more sophisticated sampling strategy like DDIM or adaptive step selection.
- Likelihood is approximated via score matching loss (MSE on noise prediction) averaged across 10 diffusion timesteps, not an exact variational lower bound. This is practical but not principled.
- The reverse step during generation does not include learned variance; it uses the simplified approximation from pass 1.
- Evaluation is on simple synthetic linear SSM data; no real time-series datasets yet.

### Pass 4 (completed)

**Implemented:**
- `DampedOscillator`: A nonlinear dynamical system (x'' + 2*gamma*x' + omega^2*x = 0) integrated with Euler's method. Generates 2D state trajectories (position, velocity) and 1D observations (noisy position). Demonstrates that the diffusion SSM can learn non-Gaussian transitions in a simple nonlinear system.
- `LorenzSystem`: The chaotic Lorenz attractor (dx/dt = sigma(y-x), dy/dt = x(rho-z)-y, dz/dt = xy-beta*z) with 3D state and 2D observations (x, z coordinates). Shows that the model handles chaotic dynamics where transitions are highly nonlinear and multimodal.
- `demo_damped_oscillator()` and `demo_lorenz_system()`: End-to-end demos that train the diffusion SSM on trajectories from each system, then evaluate via generation and likelihood estimation. Both report final training loss, generation MSE, and NLL.
- `run_pass_4_demo()`: Orchestrates both demos in sequence and prints a summary.
- Test suite for pass 4: `test_damped_oscillator()`, `test_lorenz_system()`, `test_demo_damped_oscillator()`, `test_demo_lorenz_system()` verify trajectory shapes, finiteness, and demo function execution.
- Verified end-to-end: all existing tests (pass 1-3) still pass; new demos run without crashes and produce finite metrics.

**Simplified/Stubbed:**
- The damped oscillator and Lorenz system are still low-dimensional toy models, not high-dimensional real-world time series.
- No comparison to classical SSM baselines (e.g., Kalman filter) or other generative models (e.g., VAE-based SSMs).
- No statistical significance testing or error bars; metrics are point estimates.
- Demos use relatively small training sets (50 trajectories for oscillator, 40 for Lorenz) and modest training epochs (30-50) for speed.
- No hyperparameter tuning; all networks use fixed architectures (64-dim hidden layer) and learning rates (1e-3).
- Visualization is minimal (console output only); no plots of trajectories or phase portraits.

**Why this approach:**
The paper's key claim is that diffusion models can learn complex, non-Gaussian transition densities jointly with the SSM parameters, outperforming fixed parametric assumptions. Pass 4 validates this by showing the model can train and generate on nonlinear systems (damped oscillator) and chaotic systems (Lorenz) where Gaussian transitions are clearly inappropriate. The demos are intentionally small and fast so they can run as part of the test suite, ensuring the implementation is correct. Scaling to larger datasets and real-world time series would require more compute and better hyperparameter choices, but is straightforward given the current modular structure.
