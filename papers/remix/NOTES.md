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
