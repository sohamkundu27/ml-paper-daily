"""End-to-end image generation demo for BitDance.

Demonstrates the full pipeline:
1. Encode random/toy images to binary tokens
2. Mask some tokens
3. Regenerate masked tokens using autoregressive model
4. Decode back to image space
5. Compute quality metrics (reconstruction error, diversity)
"""

import torch
import torch.nn.functional as F
from binary_tokenizer import BinaryTokenizer
from binary_diffusion import BinaryDiffusionHead, BinaryDiffusionSampler
from autoregressive_predictor import AutoregressiveTokenPredictor, MultiScaleTokenGenerator


class EndToEndDemo:
    """End-to-end image generation pipeline for BitDance."""

    def __init__(
        self,
        tokenizer=None,
        diffusion_head=None,
        autoregressive_predictor=None,
        device="cpu",
    ):
        """
        Initialize the demo pipeline.

        Args:
            tokenizer: BinaryTokenizer model (created if None)
            diffusion_head: BinaryDiffusionHead model (created if None)
            autoregressive_predictor: AutoregressiveTokenPredictor (created if None)
            device: device to run on
        """
        self.device = device

        # Initialize components if not provided
        if tokenizer is None:
            self.tokenizer = BinaryTokenizer(in_channels=3, token_dim=32).to(device)
        else:
            self.tokenizer = tokenizer.to(device)

        if diffusion_head is None:
            self.diffusion_head = BinaryDiffusionHead(
                token_dim=32, hidden_dim=128, num_timesteps=1000
            ).to(device)
        else:
            self.diffusion_head = diffusion_head.to(device)

        if autoregressive_predictor is None:
            self.autoregressive_predictor = AutoregressiveTokenPredictor(
                token_dim=32, embed_dim=128, num_heads=4, depth=2, num_timesteps=1000
            ).to(device)
        else:
            self.autoregressive_predictor = autoregressive_predictor.to(device)

        self.diffusion_sampler = BinaryDiffusionSampler(
            self.diffusion_head, num_timesteps=1000, noise_scale=1.0
        )
        self.token_generator = MultiScaleTokenGenerator(
            self.autoregressive_predictor, diffusion_sampler=self.diffusion_sampler, num_scales=1
        )

    def create_toy_images(self, batch_size=4, height=32, width=32):
        """
        Create synthetic toy images for testing.

        Args:
            batch_size: number of images
            height: image height
            width: image width

        Returns:
            images: (B, 3, H, W) tensor in [0, 1]
        """
        images = []

        for i in range(batch_size):
            # Create simple patterns
            img = torch.zeros(3, height, width)

            if i == 0:
                # Solid color
                img[0, :, :] = 0.7  # Red channel
            elif i == 1:
                # Vertical stripes
                img[1, :, ::2] = 0.7  # Green channel
            elif i == 2:
                # Horizontal stripes
                img[2, ::2, :] = 0.7  # Blue channel
            else:
                # Checkerboard
                img[0, ::2, ::2] = 0.7
                img[1, 1::2, 1::2] = 0.7

            images.append(img)

        return torch.stack(images).to(self.device)

    def encode_images(self, images):
        """
        Encode images to binary tokens.

        Args:
            images: (B, 3, H, W) image tensor in [0, 1]

        Returns:
            tokens: continuous-valued token embeddings (B, token_dim, h, w)
            binary_tokens: binarized tokens (B, token_dim, h, w) in {0, 1}
        """
        self.tokenizer.eval()
        with torch.no_grad():
            tokens, binary_tokens = self.tokenizer.encode(images)
        return tokens, binary_tokens

    def decode_tokens(self, tokens):
        """
        Decode tokens back to images.

        Args:
            tokens: (B, token_dim, h, w) token tensor

        Returns:
            images: (B, 3, H, W) reconstructed images in [0, 1]
        """
        self.tokenizer.eval()
        with torch.no_grad():
            images = self.tokenizer.decode(tokens)
        return images

    def mask_tokens(self, binary_tokens, mask_fraction=0.5):
        """
        Mask random tokens by setting them to 0.

        Args:
            binary_tokens: (B, token_dim, h, w) binary tokens
            mask_fraction: fraction of tokens to mask [0, 1]

        Returns:
            masked_tokens: tokens with random portion set to 0
            mask: boolean mask indicating which tokens were masked
        """
        batch_size, token_dim, h, w = binary_tokens.shape
        mask = torch.bernoulli(
            torch.full((batch_size, token_dim, h, w), mask_fraction, device=self.device)
        )
        masked_tokens = binary_tokens * (1 - mask)
        return masked_tokens, mask.bool()

    def regenerate_masked_tokens(self, masked_tokens, num_diffusion_steps=20):
        """
        Regenerate masked tokens using the autoregressive model.

        Args:
            masked_tokens: (B, token_dim, h, w) tokens with some positions masked
            num_diffusion_steps: number of diffusion steps for refinement

        Returns:
            regenerated_tokens: (B, token_dim, h, w) full token grid
        """
        self.autoregressive_predictor.eval()
        self.diffusion_head.eval()

        batch_size, token_dim, h, w = masked_tokens.shape

        # Flatten tokens to sequence for autoregressive model
        flattened = masked_tokens.permute(0, 2, 3, 1).reshape(batch_size, h * w, token_dim)

        with torch.no_grad():
            # Get predictions from autoregressive model
            pred_tokens = self.autoregressive_predictor(flattened)

        # Reshape back to spatial
        pred_tokens = pred_tokens.reshape(batch_size, h, w, token_dim).permute(0, 3, 1, 2)

        # Use diffusion sampler for refinement
        with torch.no_grad():
            refined = self.diffusion_sampler.sample(
                pred_tokens.shape, num_steps=num_diffusion_steps, device=self.device
            )

        return refined

    def generate_image_completion(self, images, mask_fraction=0.5, num_diffusion_steps=20):
        """
        Full end-to-end pipeline: encode -> mask -> regenerate -> decode.

        Args:
            images: (B, 3, H, W) original images
            mask_fraction: fraction of tokens to mask
            num_diffusion_steps: diffusion steps for refinement

        Returns:
            reconstructed: (B, 3, H, W) reconstructed images
            original: (B, 3, H, W) original images
            mask: (B, token_dim, h, w) boolean mask
        """
        # Encode to tokens
        tokens, binary_tokens = self.encode_images(images)

        # Mask some tokens
        masked_tokens, mask = self.mask_tokens(binary_tokens, mask_fraction=mask_fraction)

        # Regenerate masked tokens
        regenerated_tokens = self.regenerate_masked_tokens(
            masked_tokens, num_diffusion_steps=num_diffusion_steps
        )

        # Decode back to images
        reconstructed = self.decode_tokens(regenerated_tokens)

        return reconstructed, images, mask

    def compute_metrics(self, original_images, reconstructed_images, mask=None):
        """
        Compute quality metrics for reconstruction.

        Args:
            original_images: (B, 3, H, W) original images
            reconstructed_images: (B, 3, H, W) reconstructed images
            mask: optional (B, C, h, w) mask indicating regenerated regions

        Returns:
            metrics: dict with computed metrics
        """
        # MSE reconstruction error
        mse = F.mse_loss(original_images, reconstructed_images).item()
        psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item()

        # L1 error
        l1 = F.l1_loss(original_images, reconstructed_images).item()

        # If mask provided, compute metrics on masked regions separately
        metrics = {"mse": mse, "l1": l1, "psnr": psnr}

        if mask is not None:
            # Mask is on token grid, skip direct region comparison
            # Just record fraction of tokens masked
            mask_fraction = mask.float().mean().item()
            metrics["mask_fraction"] = mask_fraction

        # Diversity: variance of reconstructed samples across batch
        if original_images.shape[0] > 1:
            variance = original_images.var(dim=0).mean().item()
            recon_variance = reconstructed_images.var(dim=0).mean().item()
            metrics["original_variance"] = variance
            metrics["reconstructed_variance"] = recon_variance

        return metrics

    def run_demo(self, batch_size=4, mask_fraction=0.5, num_diffusion_steps=20):
        """
        Run the full demo pipeline.

        Args:
            batch_size: number of images to process
            mask_fraction: fraction of tokens to mask
            num_diffusion_steps: number of diffusion refinement steps

        Returns:
            results: dict with results and metrics
        """
        # Create toy images
        original_images = self.create_toy_images(batch_size=batch_size, height=32, width=32)

        # Run end-to-end pipeline
        reconstructed_images, _, mask = self.generate_image_completion(
            original_images, mask_fraction=mask_fraction, num_diffusion_steps=num_diffusion_steps
        )

        # Compute metrics
        metrics = self.compute_metrics(original_images, reconstructed_images, mask=mask)

        results = {
            "original_images": original_images,
            "reconstructed_images": reconstructed_images,
            "mask": mask,
            "metrics": metrics,
        }

        return results
