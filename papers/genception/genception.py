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


class DepthHead(nn.Module):
    """Decoder head for depth estimation."""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels)
        self.depth_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: (batch, in_channels, H, W)
        Returns:
            depth: (batch, 1, H, W) depth predictions
        """
        x = self.conv1(features)
        x = self.conv2(x)
        depth = self.depth_pred(x)
        return depth


class NormalsHead(nn.Module):
    """Decoder head for surface normals estimation."""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels)
        self.normals_pred = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: (batch, in_channels, H, W)
        Returns:
            normals: (batch, 3, H, W) surface normal predictions
        """
        x = self.conv1(features)
        x = self.conv2(x)
        normals = self.normals_pred(x)
        return normals


class SegmentationHead(nn.Module):
    """Decoder head for semantic segmentation."""

    def __init__(self, in_channels, num_classes=10, hidden_channels=64):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels)
        self.seg_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: (batch, in_channels, H, W)
        Returns:
            segmentation: (batch, num_classes, H, W) segmentation logits
        """
        x = self.conv1(features)
        x = self.conv2(x)
        seg = self.seg_pred(x)
        return seg


class TaskSelector(nn.Module):
    """Selects task based on instruction embedding."""

    def __init__(self, instruction_dim, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_projector = nn.Linear(instruction_dim, num_tasks)

    def forward(self, instruction_vec):
        """
        Args:
            instruction_vec: (batch, instruction_dim)
        Returns:
            task_logits: (batch, num_tasks) logits for each task
            task_ids: (batch,) selected task index for each sample
        """
        task_logits = self.task_projector(instruction_vec)
        task_ids = task_logits.argmax(dim=1)
        return task_logits, task_ids


class GenCeption(nn.Module):
    """GenCeption: instruction-conditioned vision backbone with multi-task heads."""

    def __init__(self, vocab_size=1000, embed_dim=128, base_channels=32, num_blocks=3, num_classes=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.num_classes = num_classes

        self.text_embedding = SimpleTextEmbedding(vocab_size, embed_dim)
        self.image_backbone = SimpleUNetBackbone(in_channels=3,
                                                  base_channels=base_channels,
                                                  num_blocks=num_blocks)
        self.conditioner = InstructionConditioner(base_channels, embed_dim)

        # Task selection and decoder heads
        self.task_selector = TaskSelector(embed_dim, num_tasks=3)
        self.depth_head = DepthHead(base_channels, hidden_channels=64)
        self.normals_head = NormalsHead(base_channels, hidden_channels=64)
        self.segmentation_head = SegmentationHead(base_channels, num_classes=num_classes, hidden_channels=64)

        self.task_names = ["depth", "normals", "segmentation"]

    def forward(self, image, token_ids, return_all_tasks=False):
        """
        Args:
            image: (batch, 3, H, W) input image
            token_ids: (batch, seq_len) token indices for instruction
            return_all_tasks: if True, return dict with all task outputs; if False, return conditioned features
        Returns:
            If return_all_tasks=False (default): features (batch, base_channels, H, W) for backward compatibility
            If return_all_tasks=True: dict with task_id, task_logits, depth, normals, segmentation
        """
        # Extract instruction embedding
        instruction_vec = self.text_embedding(token_ids)

        # Extract image features
        image_features = self.image_backbone(image)

        # Condition features on instruction
        conditioned_features = self.conditioner(image_features, instruction_vec)

        if not return_all_tasks:
            # Pass 1 backward compatibility: return conditioned features
            return conditioned_features

        # Pass 2 and beyond: compute task-specific outputs
        task_logits, task_ids = self.task_selector(instruction_vec)

        # Compute outputs for all tasks
        depth_out = self.depth_head(conditioned_features)  # (batch, 1, H, W)
        normals_out = self.normals_head(conditioned_features)  # (batch, 3, H, W)
        seg_out = self.segmentation_head(conditioned_features)  # (batch, num_classes, H, W)

        output_dict = {
            "task_id": task_ids,
            "task_logits": task_logits,
            "depth": depth_out,
            "normals": normals_out,
            "segmentation": seg_out
        }

        return output_dict
