# ReMix: Reinforcement routing for mixtures of LoRAs

**arXiv:** https://arxiv.org/abs/2603.10160

**Published:** March 2026 (LLA Workshop at ICLR 2026)

**Summary:**

Low-rank adapters (LoRAs) enable efficient fine-tuning by injecting trainable low-rank weight matrices into frozen pretrained models, dramatically reducing the number of parameters that need to be updated. A natural extension is to use multiple LoRAs and route different inputs to different subsets of them, creating a mixture-of-LoRAs. However, existing approaches suffer from routing weight collapse: only one or two LoRAs end up dominating the routing decisions, severely limiting the effective capacity of the ensemble.

ReMix addresses this by using discrete, non-learnable routing weights that ensure all LoRAs are equally selected, combined with a reinforcement learning-based training procedure. The router is treated as a policy that learns to select which LoRAs to activate, with the supervision loss as the reward. An unbiased gradient estimator using the RLOO (leave-one-out) technique enables stable training while keeping all LoRAs active and contributing.

## Plan: 4 passes

**Pass 1 (Foundational):** Implement basic LoRA layer and a simple fixed router. A LoRA layer injects trainable low-rank matrices A and B into a pretrained linear layer (W), computing output as W*x + α*(A*B*x). Include a minimal router that uniformly selects which LoRAs to activate.

**Pass 2 (Core mechanism):** Implement the reinforcement learning router using the RLOO gradient estimator. The router learns a policy to select LoRAs based on input, with load balancing constraints to prevent collapse.

**Pass 3 (Integration):** Combine multiple LoRAs in a mixture architecture and add monitoring for routing statistics to verify that load is balanced across all LoRAs.

**Pass 4 (End-to-end demo):** Small demonstration on toy fine-tuning task (e.g., adapting a small language model for a specific domain or task) showing that mixture-of-LoRAs outperforms single LoRA under the same parameter budget.

## Implemented vs. simplified

### Pass 1 implementation:
- ✅ `LoRALayer`: Low-rank adapter computing α·(x @ A @ B), where A ∈ ℝ^{in×rank}, B ∈ ℝ^{rank×out}
- ✅ `LoRALinear`: Combines frozen base linear layer with injected LoRA (output = W·x + LoRA(x))
- ✅ `SimpleRouter`: Uniform random router that independently selects k LoRAs per sample
- ✅ `MixtureOfLoRAs`: Stacks multiple LoRAs with routing, averages contributions from selected LoRAs
- ✅ Comprehensive test suite: forward pass shapes, gradient flow, routing statistics, end-to-end training

### Simplified/stubbed in Pass 1:
- **Router is non-learnable**: Fixed uniform random selection (Pass 2 adds learned routing)
- **No gradient estimation**: SimpleRouter outputs hard binary masks (no RLOO or variance reduction yet)
- **No load balancing**: Routing can collapse in principle (Pass 2 enforces balanced activation)
- **No reward signal or RL**: Router selection is not optimized via reinforcement learning
- **Uniform averaging**: All selected LoRAs contribute equally (no learned weighting)
- **No language model integration**: Works on generic linear layer adapter, not real LLM weights
- **Fixed rank**: All LoRAs use the same rank (future work could use adaptive ranks)

### Pass 2 implementation:
- ✅ `LearnedRouter`: Policy network that learns to score each LoRA based on input features
- ✅ **RLOO-style gradient estimation**: Computes policy loss with baseline to reduce variance
- ✅ **Load balancing loss**: KL divergence penalty to encourage uniform activation across all LoRAs
- ✅ **Policy gradient training**: Uses REINFORCE-style update where negative loss serves as reward signal
- ✅ `MixtureOfLoRAsRL`: Full integration with learned routing and end-to-end training
- ✅ Load tracking and statistics: Monitor routing imbalance throughout training
- ✅ Comprehensive test suite: Policy loss gradient flow, load balancing, training convergence

### Simplified/stubbed in Pass 2:
- **Deterministic top-k selection**: Uses argmax for inference rather than sampling (simplification for stability)
- **Per-batch baseline**: Baseline is mean loss over batch (full RLOO would compute per-sample leave-one-out baselines)
- **No sophisticated RL**: Uses simple policy gradient rather than more complex RL algorithms (PPO, A3C, etc.)
- **No language model weights**: Still works on synthetic/toy tasks, not real LLM fine-tuning

### Pass 3 implementation:
- ✅ `RoutingMonitor`: Tracks activation statistics across batches for each LoRA
  - Per-LoRA activation rates: fraction of samples each LoRA is selected for
  - Imbalance ratio: max load / min load (1.0 = perfect balance, ∞ = collapse)
  - Shannon entropy: measures uniformity of activation distribution (higher = more uniform)
  - Snapshot history: records statistics at specific training steps
- ✅ `MixtureOfLoRAsMonitored`: Full mixture with integrated monitoring
  - Automatically tracks routing decisions during forward/backward passes
  - Provides `get_routing_statistics()` for querying current load distribution
  - Can reset monitoring counters to measure statistics over specific windows
  - Delegates model parameters to underlying mixture for optimizer compatibility
- ✅ Comprehensive monitoring tests (9 new tests):
  - Basic initialization and updates
  - Statistic computation (rates, imbalance, entropy)
  - Imbalance detection (verifies monitoring catches collapse patterns)
  - Monitor reset functionality
  - Integration with monitored mixture
  - Training with real load-balancing verification
  - Consistency with unmonitored version (same outputs)
  - History snapshots for tracking training progression

### Simplified/stubbed in Pass 3:
- **Limited to top-k routing**: Still using deterministic top-k rather than sampling-based routing
- **No visualization**: Monitoring data is numeric; visualization tools not yet implemented
- **No real LLM adaptation**: Monitoring works on synthetic tasks, not real language model fine-tuning
