"""
CART: Context-Anchored Recurrent Transformer
Pass 1: Core MLA block with learned LTI gate for stability
Pass 2: Multi-layer prelude network for context encoding
Pass 3: Stacked recurrent iterations with residual + layer norm, plus sequence classification task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CartMLABlock(nn.Module):
    """
    Multi-head latent attention block with context reuse.

    In pass 1, this is simplified: no multi-head projection, single query transform.
    Context (K,V) is provided as input and reused throughout recurrence.
    """

    def __init__(self, dim: int, head_dim: int = 64, dropout: float = 0.0):
        """
        Args:
            dim: Feature dimension
            head_dim: Dimension per attention head
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Query projection: transform input for cross-attention over fixed context
        self.q_proj = nn.Linear(dim, head_dim)

        # Context is provided externally (K, V), no projection needed in this block
        # In Pass 2, prelude will generate K,V

        # Output projection
        self.out_proj = nn.Linear(head_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context_k: torch.Tensor, context_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Query input, shape (batch, seq_len, dim)
            context_k: Context keys, shape (batch, ctx_len, head_dim)
            context_v: Context values, shape (batch, ctx_len, head_dim)

        Returns:
            Output, shape (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        ctx_len = context_k.size(1)

        # Project query
        q = self.q_proj(x)  # (batch, seq_len, head_dim)

        # Attention scores: q @ k^T / sqrt(d)
        scores = torch.matmul(q, context_k.transpose(-2, -1)) * self.scale  # (batch, seq_len, ctx_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, context_v)  # (batch, seq_len, head_dim)

        # Output projection
        output = self.out_proj(attn_output)  # (batch, seq_len, dim)

        return output


class LearnedLTIGate(nn.Module):
    """
    Learned Linear Time-Invariant (LTI) gate for recurrent stability.

    Maintains spectral radius in a stable range via learnable scalar parameter.
    The recurrence x_{t+1} = α * x_t + y_t, where α controls the spectral radius.
    """

    def __init__(self, init_alpha: float = 0.8):
        """
        Args:
            init_alpha: Initial value for the gate parameter (spectral radius).
                       Should be in [0.7, 0.9] for stable recurrence.
        """
        super().__init__()
        # Store alpha as parameter, clipped to [0, 1) to ensure stability
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Recurrent update: x_new = clipped_alpha * x + y

        Args:
            x: Previous state, shape (batch, seq_len, dim)
            y: New information, shape (batch, seq_len, dim)

        Returns:
            Updated state, shape (batch, seq_len, dim)
        """
        # Clip alpha to [0, 0.99) to ensure spectral radius < 1 (stability)
        alpha_stable = torch.clamp(self.alpha, 0.0, 0.99)

        # Recurrent update
        x_new = alpha_stable * x + y

        return x_new

    def get_spectral_radius(self) -> torch.Tensor:
        """Return the current spectral radius (clamped alpha)."""
        return torch.clamp(self.alpha, 0.0, 0.99)


class CartPrelude(nn.Module):
    """
    Multi-layer prelude network that encodes context into reusable K,V representations.

    The prelude computes K and V once from raw context, which are then reused
    across multiple recurrent iterations in the core. This separation of context
    encoding from iterative refinement is key to CART's parameter efficiency.

    Pass 2: multi-layer feedforward encoder with separate K,V projections.
    """

    def __init__(self, dim: int, head_dim: int, num_layers: int = 2, dropout: float = 0.0):
        """
        Args:
            dim: Input context dimension
            head_dim: Dimension per attention head (output dimension)
            num_layers: Number of layers in the encoder (default 2)
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        # Build multi-layer feedforward encoder
        layers = []
        for i in range(num_layers):
            in_d = dim if i == 0 else head_dim
            out_d = head_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

        # Separate learnable projections for K and V
        self.k_proj = nn.Linear(head_dim, head_dim)
        self.v_proj = nn.Linear(head_dim, head_dim)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context into K,V representations.

        Args:
            context: Raw context, shape (batch, ctx_len, dim)

        Returns:
            Tuple of (K, V), each with shape (batch, ctx_len, head_dim)
        """
        # Encode context through multi-layer feedforward
        encoded = self.encoder(context)  # (batch, ctx_len, head_dim)

        # Project to K and V (independent projections for expressiveness)
        k = self.k_proj(encoded)  # (batch, ctx_len, head_dim)
        v = self.v_proj(encoded)  # (batch, ctx_len, head_dim)

        return k, v


class CartRecurrentCore(nn.Module):
    """
    Recurrent core of CART: iteratively refines representation via MLA.

    Pass 1 simplified: single recurrent iteration over a fixed context.
    Pass 3: adds layer normalization and residual connections.
    """

    def __init__(self, dim: int, head_dim: int = 64, num_iterations: int = 1, dropout: float = 0.0):
        """
        Args:
            dim: Feature dimension
            head_dim: Dimension per attention head
            num_iterations: Number of recurrent iterations
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations

        # MLA block
        self.mla = CartMLABlock(dim, head_dim, dropout)

        # LTI gate for stability
        self.lti_gate = LearnedLTIGate(init_alpha=0.8)

        # Layer normalization for stable stacking (Pass 3)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x_init: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_init: Initial state, shape (batch, seq_len, dim)
            context_k: Context keys, shape (batch, ctx_len, head_dim)
            context_v: Context values, shape (batch, ctx_len, head_dim)

        Returns:
            Final state after recurrent iterations, shape (batch, seq_len, dim)
        """
        x = x_init

        for _ in range(self.num_iterations):
            # Apply MLA block with residual connection and layer norm
            residual = x
            y = self.mla(x, context_k, context_v)

            # Recurrent update via LTI gate
            x = self.lti_gate(residual, y)

            # Apply layer normalization (Pass 3)
            x = self.norm(x)

        return x


class Cart(nn.Module):
    """
    Full CART model: Prelude + RecurrentCore

    Demonstrates the separation of context encoding from iterative refinement.
    The prelude encodes raw context into K,V once; the core reuses them across
    multiple recurrent iterations, reducing per-iteration computation.

    Pass 2: integrates CartPrelude with CartRecurrentCore.
    Pass 3: adds residual connections and layer norm to core.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        prelude_layers: int = 2,
        num_iterations: int = 1,
        dropout: float = 0.0
    ):
        """
        Args:
            dim: Feature dimension (input and output)
            head_dim: Dimension per attention head
            prelude_layers: Number of layers in prelude encoder
            num_iterations: Number of recurrent iterations
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        # Prelude: encodes raw context into K,V
        self.prelude = CartPrelude(dim, head_dim, prelude_layers, dropout)

        # Recurrent core: refines input using encoded K,V
        self.core = CartRecurrentCore(dim, head_dim, num_iterations, dropout)

    def forward(self, x_init: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: prelude encodes context, core refines input.

        Args:
            x_init: Initial representation, shape (batch, seq_len, dim)
            context: Raw context, shape (batch, ctx_len, dim)

        Returns:
            Refined representation, shape (batch, seq_len, dim)
        """
        # Prelude: encode context into K,V (computed once, reused across iterations)
        context_k, context_v = self.prelude(context)

        # Core: refine x_init using fixed K,V across recurrent iterations
        output = self.core(x_init, context_k, context_v)

        return output


class CartSequenceClassifier(nn.Module):
    """
    Sequence-level classification task on top of CART.

    Pass 3: Simple task to demonstrate sequence-level reasoning.
    Task: predict if a sequence length is above a threshold.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        prelude_layers: int = 2,
        num_iterations: int = 1,
        dropout: float = 0.0,
        num_classes: int = 2
    ):
        """
        Args:
            dim: Feature dimension
            head_dim: Dimension per attention head
            prelude_layers: Number of layers in prelude
            num_iterations: Number of recurrent iterations
            dropout: Dropout rate
            num_classes: Number of output classes (default: binary classification)
        """
        super().__init__()
        self.cart = Cart(dim, head_dim, prelude_layers, num_iterations, dropout)

        # Simple pooling + classification head
        self.pool_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x_init: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence classification.

        Args:
            x_init: Initial sequence representation, shape (batch, seq_len, dim)
            context: Context, shape (batch, ctx_len, dim)

        Returns:
            Logits for classification, shape (batch, num_classes)
        """
        # Get refined representation
        refined = self.cart(x_init, context)

        # Mean pooling over sequence dimension
        pooled = refined.mean(dim=1)  # (batch, dim)

        # Normalize before classification
        pooled = self.pool_norm(pooled)

        # Classification
        logits = self.classifier(pooled)

        return logits


def create_dummy_context(batch: int, seq_len: int, ctx_len: int, head_dim: int, device: str = 'cpu'):
    """
    Create dummy context K,V for testing.

    Args:
        batch: Batch size
        seq_len: Sequence length
        ctx_len: Context length
        head_dim: Dimension per head
        device: Device to place tensors on

    Returns:
        Tuple of (K, V)
    """
    context_k = torch.randn(batch, ctx_len, head_dim, device=device)
    context_v = torch.randn(batch, ctx_len, head_dim, device=device)
    return context_k, context_v


def create_dummy_raw_context(batch: int, ctx_len: int, dim: int, device: str = 'cpu'):
    """
    Create dummy raw context for prelude processing.

    Args:
        batch: Batch size
        ctx_len: Context length
        dim: Feature dimension
        device: Device to place tensors on

    Returns:
        Context tensor of shape (batch, ctx_len, dim)
    """
    return torch.randn(batch, ctx_len, dim, device=device)


def create_synthetic_length_classification_dataset(
    num_samples: int,
    seq_len_range: Tuple[int, int],
    ctx_len: int,
    dim: int,
    threshold: int = 8,
    device: str = 'cpu'
):
    """
    Create synthetic dataset for sequence length classification task.

    Task: Predict if sequence length is >= threshold.

    Args:
        num_samples: Number of samples
        seq_len_range: (min_seq_len, max_seq_len)
        ctx_len: Context length
        dim: Feature dimension
        threshold: Length threshold for classification
        device: Device to place tensors on

    Returns:
        Tuple of (inputs, contexts, labels)
        - inputs: shape (num_samples, seq_len, dim)
        - contexts: shape (num_samples, ctx_len, dim)
        - labels: shape (num_samples,) with values in {0, 1}
    """
    min_len, max_len = seq_len_range

    inputs = []
    contexts = []
    labels = []

    for _ in range(num_samples):
        # Random sequence length
        seq_len = torch.randint(min_len, max_len + 1, (1,)).item()

        # Random input sequence
        x = torch.randn(seq_len, dim, device=device)
        inputs.append(x)

        # Random context
        ctx = torch.randn(ctx_len, dim, device=device)
        contexts.append(ctx)

        # Label: 1 if seq_len >= threshold else 0
        label = torch.tensor(1 if seq_len >= threshold else 0, device=device)
        labels.append(label)

    # Pad inputs and contexts to fixed lengths for batching
    max_seq_len = max_len
    padded_inputs = torch.zeros(num_samples, max_seq_len, dim, device=device)
    padded_contexts = torch.zeros(num_samples, ctx_len, dim, device=device)

    for i, (inp, ctx) in enumerate(zip(inputs, contexts)):
        padded_inputs[i, :inp.shape[0], :] = inp
        padded_contexts[i, :, :] = ctx

    labels = torch.stack(labels)

    return padded_inputs, padded_contexts, labels


def train_classifier_step(
    model: CartSequenceClassifier,
    batch_inputs: torch.Tensor,
    batch_contexts: torch.Tensor,
    batch_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module
) -> float:
    """
    Single training step for CartSequenceClassifier.

    Args:
        model: Classifier model
        batch_inputs: Input batch, shape (batch, seq_len, dim)
        batch_contexts: Context batch, shape (batch, ctx_len, dim)
        batch_labels: Label batch, shape (batch,)
        optimizer: Optimizer
        loss_fn: Loss function

    Returns:
        Loss value
    """
    optimizer.zero_grad()

    # Forward pass
    logits = model(batch_inputs, batch_contexts)

    # Compute loss
    loss = loss_fn(logits, batch_labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_classifier(
    model: CartSequenceClassifier,
    inputs: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate classifier on a dataset.

    Args:
        model: Classifier model
        inputs: Input data, shape (num_samples, seq_len, dim)
        contexts: Context data, shape (num_samples, ctx_len, dim)
        labels: Label data, shape (num_samples,)
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_end = min(i + batch_size, len(inputs))
            batch_inputs = inputs[i:batch_end]
            batch_contexts = contexts[i:batch_end]
            batch_labels = labels[i:batch_end]

            logits = model(batch_inputs, batch_contexts)
            loss = loss_fn(logits, batch_labels)

            total_loss += loss.item() * (batch_end - i)
            total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total_samples += batch_end - i

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    model.train()

    return avg_loss, accuracy


class CartLanguageModel(nn.Module):
    """
    Simple language model using CART for token prediction.

    Pass 4: Demonstrates CART in a language modeling task.
    Given a context (previous tokens), predicts the next token.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        head_dim: int = 64,
        prelude_layers: int = 2,
        num_iterations: int = 2,
        dropout: float = 0.0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            dim: Feature dimension
            head_dim: Dimension per attention head
            prelude_layers: Number of layers in prelude
            num_iterations: Number of recurrent iterations
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, dim)

        # CART model: context encoding + refinement
        self.cart = Cart(dim, head_dim, prelude_layers, num_iterations, dropout)

        # Output projection to vocab
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        context_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for language model.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            context_ids: Context token IDs, shape (batch, ctx_len)

        Returns:
            Logits for next token prediction, shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x_init = self.embed(input_ids)  # (batch, seq_len, dim)
        context = self.embed(context_ids)  # (batch, ctx_len, dim)

        # Apply CART
        refined = self.cart(x_init, context)  # (batch, seq_len, dim)

        # Project to vocabulary
        logits = self.output_proj(refined)  # (batch, seq_len, vocab_size)

        return logits


class StandardTransformerBaseline(nn.Module):
    """
    Standard transformer encoder for comparison with CART.

    This is a simple stack of transformer encoder layers for fair parameter comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            dim: Feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, dim)

        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to vocab
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer baseline.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)

        Returns:
            Logits for token prediction, shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embed(input_ids)  # (batch, seq_len, dim)

        # Apply transformer
        refined = self.encoder(x)  # (batch, seq_len, dim)

        # Project to vocabulary
        logits = self.output_proj(refined)  # (batch, seq_len, vocab_size)

        return logits


def create_synthetic_language_dataset(
    num_samples: int,
    seq_len: int,
    ctx_len: int,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic dataset for language modeling.

    Args:
        num_samples: Number of samples
        seq_len: Input sequence length
        ctx_len: Context length
        vocab_size: Vocabulary size
        device: Device to place tensors on

    Returns:
        Tuple of (input_ids, context_ids, target_ids)
        - input_ids: shape (num_samples, seq_len)
        - context_ids: shape (num_samples, ctx_len)
        - target_ids: shape (num_samples, seq_len) -- next token in each position
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    context_ids = torch.randint(0, vocab_size, (num_samples, ctx_len), device=device)
    # Target: simple next-token prediction (shift input by 1, last position is random)
    target_ids = torch.cat([
        input_ids[:, 1:],
        torch.randint(0, vocab_size, (num_samples, 1), device=device)
    ], dim=1)

    return input_ids, context_ids, target_ids


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_lm_step(
    model: nn.Module,
    batch_input_ids: torch.Tensor,
    batch_context_ids: torch.Tensor,
    batch_target_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    is_cart: bool = True
) -> float:
    """
    Single training step for language model.

    Args:
        model: Language model
        batch_input_ids: Input token IDs, shape (batch, seq_len)
        batch_context_ids: Context token IDs, shape (batch, ctx_len) [only for CART]
        batch_target_ids: Target token IDs, shape (batch, seq_len)
        optimizer: Optimizer
        loss_fn: Loss function
        is_cart: Whether model is CART (True) or standard transformer (False)

    Returns:
        Loss value
    """
    optimizer.zero_grad()

    # Forward pass
    if is_cart:
        logits = model(batch_input_ids, batch_context_ids)
    else:
        logits = model(batch_input_ids)

    # Reshape for loss
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = batch_target_ids.reshape(-1)

    # Compute loss
    loss = loss_fn(logits_flat, targets_flat)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_lm(
    model: nn.Module,
    input_ids: torch.Tensor,
    context_ids: torch.Tensor,
    target_ids: torch.Tensor,
    batch_size: int = 32,
    is_cart: bool = True
) -> float:
    """
    Evaluate language model on a dataset.

    Args:
        model: Language model
        input_ids: Input token IDs, shape (num_samples, seq_len)
        context_ids: Context token IDs, shape (num_samples, ctx_len) [only for CART]
        target_ids: Target token IDs, shape (num_samples, seq_len)
        batch_size: Batch size for evaluation
        is_cart: Whether model is CART (True) or standard transformer (False)

    Returns:
        Average loss
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_end = min(i + batch_size, len(input_ids))
            batch_input = input_ids[i:batch_end]
            batch_target = target_ids[i:batch_end]

            if is_cart:
                batch_context = context_ids[i:batch_end]
                logits = model(batch_input, batch_context)
            else:
                logits = model(batch_input)

            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = batch_target.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item() * (batch_end - i)
            total_samples += batch_end - i

    avg_loss = total_loss / total_samples
    model.train()

    return avg_loss
