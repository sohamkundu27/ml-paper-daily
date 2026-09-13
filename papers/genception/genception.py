import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTextEmbedding(nn.Module):
    """Simple text instruction embedding using word-level averaging."""

    def __init__(self, vocab_size=1000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch, seq_len) tensor of token indices
        Returns:
            embedded: (batch, embed_dim) tensor averaging word embeddings
        """
        embedded = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        instruction_vec = embedded.mean(dim=1)  # (batch, embed_dim) average pooling
        return instruction_vec


class ConvBlock(nn.Module):
    """Simple convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleUNetBackbone(nn.Module):
    """Simplified UNet-like feature extractor for images."""

    def __init__(self, in_channels=3, base_channels=32, num_blocks=3):
        super().__init__()
        self.num_blocks = num_blocks
        self.base_channels = base_channels

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            self.down_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** (num_blocks - 1))
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch * 2)

        # Upsampling path (mirror of downsampling)
        self.up_blocks = nn.ModuleList()
        current_ch = bottleneck_ch * 2
        for i in range(num_blocks - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            self.up_blocks.append(ConvBlock(current_ch, out_ch))
            current_ch = out_ch

    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W) input image
        Returns:
            features: (batch, base_channels, H, W) feature map
        """
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for up_block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = up_block(x)

        return x


class InstructionConditioner(nn.Module):
    """Fuses image features with instruction embeddings."""

    def __init__(self, feature_channels, instruction_dim):
        super().__init__()
        self.feature_channels = feature_channels
        self.instruction_dim = instruction_dim

        self.proj = nn.Linear(instruction_dim, feature_channels)

    def forward(self, image_features, instruction_vec):
        """
        Args:
            image_features: (batch, channels, H, W)
            instruction_vec: (batch, instruction_dim)
        Returns:
            conditioned: (batch, channels, H, W) instruction-modulated features
        """
        # Project instruction to feature space
        instr_proj = self.proj(instruction_vec)  # (batch, channels)

        # Reshape for broadcasting: (batch, channels, 1, 1)
        instr_proj = instr_proj.unsqueeze(-1).unsqueeze(-1)

        # Multiply features by instruction (modulation)
        conditioned = image_features * (1.0 + instr_proj)

        return conditioned


class GenCeption(nn.Module):
    """GenCeption: instruction-conditioned vision backbone."""

    def __init__(self, vocab_size=1000, embed_dim=128, base_channels=32, num_blocks=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.base_channels = base_channels

        self.text_embedding = SimpleTextEmbedding(vocab_size, embed_dim)
        self.image_backbone = SimpleUNetBackbone(in_channels=3,
                                                  base_channels=base_channels,
                                                  num_blocks=num_blocks)
        self.conditioner = InstructionConditioner(base_channels, embed_dim)

    def forward(self, image, token_ids):
        """
        Args:
            image: (batch, 3, H, W) input image
            token_ids: (batch, seq_len) token indices for instruction
        Returns:
            features: (batch, base_channels, H, W) instruction-conditioned features
        """
        # Extract instruction embedding
        instruction_vec = self.text_embedding(token_ids)

        # Extract image features
        image_features = self.image_backbone(image)

        # Condition features on instruction
        conditioned_features = self.conditioner(image_features, instruction_vec)

        return conditioned_features
