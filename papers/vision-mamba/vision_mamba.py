import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for image patches."""
    
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
    
    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pos_embed


class SSMBlock(nn.Module):
    """Simple linear state space model for sequence processing.
    
    This is a simplified SSM, not the full Mamba algorithm.
    It learns to transform sequences through learnable state transitions.
    """
    
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Learnable state transition matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, state_dim) * 0.01)
        
        # Scale factor for numerical stability
        self.scale = 1.0 / np.sqrt(state_dim)
    
    def forward(self, x):
        """
        Process sequence through SSM.
        
        Args:
            x: (B, T, D) tensor of patch embeddings
        
        Returns:
            y: (B, T, D) tensor of output features
        """
        B, T, D = x.shape
        
        # Initialize hidden state
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        
        # Forward pass through sequence
        for t in range(T):
            x_t = x[:, t, :]  # (B, D)
            
            # Update state: h_{t+1} = A h_t + B x_t
            h = torch.matmul(h, self.A.t()) + torch.matmul(x_t, self.B.t())
            h = h * self.scale
            
            # Output: y_t = C h_t
            y_t = torch.matmul(h, self.C.t())  # (B, D)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (B, T, D)


class BidirectionalSSMBlock(nn.Module):
    """Bidirectional SSM block: scan forward and backward, then combine."""
    
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.forward_ssm = SSMBlock(dim, state_dim)
        self.backward_ssm = SSMBlock(dim, state_dim)
        
        # Learnable gating to combine forward and backward outputs
        self.gate = nn.Linear(dim * 2, dim)
    
    def forward(self, x):
        """
        Apply bidirectional SSM scanning.
        
        Args:
            x: (B, T, D)
        
        Returns:
            y: (B, T, D)
        """
        # Forward pass
        y_fwd = self.forward_ssm(x)  # (B, T, D)
        
        # Backward pass (reverse sequence)
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])  # Reverse back to original order
        
        # Combine forward and backward outputs
        combined = torch.cat([y_fwd, y_bwd], dim=-1)  # (B, T, 2D)
        y = self.gate(combined)  # (B, T, D)
        
        return y


class VisionMambaBlock(nn.Module):
    """Complete Vision Mamba block for processing image patches."""
    
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.bidirectional_ssm = BidirectionalSSMBlock(dim, state_dim)
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D) where T is number of patches, D is patch dimension
        
        Returns:
            y: (B, T, D)
        """
        # Apply SSM with residual connection
        y = self.bidirectional_ssm(x)
        y = y + x  # Residual
        y = self.ln(y)
        return y
