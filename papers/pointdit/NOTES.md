# PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation

**arXiv**: https://arxiv.org/abs/2607.02515  
**Submitted**: July 2, 2026  
**Accepted to**: ICML 2026

## Summary

PointDiT proposes a minimalist diffusion-transformer approach to single-image 3D geometry reconstruction that operates directly on pixel-space point map patches. Rather than using complex latent-space encodings or hybrid architectures, the method trains a Vision Transformer-based diffusion model from scratch to denoise point maps, conditioned on image features from a pre-trained vision model. By working in pixel space and avoiding VAE bottlenecks, the model achieves sharper geometric boundaries and is more robust to ambiguous regions like transparent surfaces, surpassing more complex alternatives.

## Plan: 4 passes

**Pass 1**: Build a basic Vision Transformer-based diffusion backbone that can process 3D point map patches. Implement the core denoising network (ViT encoder + MLP head), simple Gaussian noise scheduling, and a minimal diffusion forward/reverse process. Test on random noise with fixed timesteps. No conditioning yet; no real training.

**Pass 2**: Add image feature conditioning via pre-computed visual embeddings. Integrate a condition injection mechanism into the ViT (cross-attention or embedding concatenation). Demonstrate conditioning on synthetic image features with a simple forward pass.

**Pass 3**: Implement a basic training loop with noise prediction loss on synthetic point maps. Add timestep embedding. Show that the model learns to denoise random noise toward simple geometric shapes (e.g., sphere, cube).

**Pass 4**: End-to-end demo: load a pre-trained image encoder stub, apply the diffusion model to generate 3D points from synthetic image + noise, and visualize the generated point cloud. Include an honest summary in NOTES.md of what was built vs. simplified (e.g., no actual DINO encoder, no real image data, no metrics).

## Implemented vs. simplified

**Pass 1**:
- ✅ Basic ViT-based diffusion backbone (simplified: plain 4-layer ViT, no advanced architectural details)
- ✅ Gaussian noise schedule (linear schedule, no complex variance schedules)
- ✅ Diffusion forward/reverse process skeleton
- ✅ Minimal test: denoising shapes with asserts on output dimensions
- ❌ No real training
- ❌ No image conditioning
- ❌ No pre-trained encoders
- ❌ No real 3D data or metrics

**Pass 2 (this commit)**:
- ✅ Cross-attention mechanism for image feature conditioning (added CrossAttentionBlock)
- ✅ Flexible condition injection into DiffusionTransformer (accepts variable-dimension embeddings)
- ✅ Support for both patched and global image embeddings
- ✅ Backward compatibility: model works without conditioning (use_cross_attention=False)
- ✅ Tests demonstrate conditioning with synthetic image features (different condition dimensions)
- ✅ Verified that conditioning affects model output (different conditions → different predictions)
- ❌ No real image encoder integration yet (that's pass 3 onwards)
- ❌ No actual training or loss computation
- ❌ No real 3D data or evaluation metrics
