import torch
from diffusion import GaussianDiffusion

diffusion = GaussianDiffusion(timesteps=100)
print('Alpha at t=0:', diffusion.alphas_cumprod[0].item())
print('Alpha at t=1:', diffusion.alphas_cumprod[1].item())
print('Alpha at t=99:', diffusion.alphas_cumprod[99].item())

# Test q_sample
x0 = torch.ones(1, 10)
noise = torch.zeros_like(x0)
x_t_0 = diffusion.q_sample(x0, 0, noise)

sqrt_alpha_0 = diffusion.sqrt_alphas_cumprod[0]
print('\nFor x0=1, noise=0:')
print('sqrt_alpha_0:', sqrt_alpha_0.item())
print('x_t at t=0 (should be close to 1):', x_t_0[0, 0].item())
print('Expected:', sqrt_alpha_0.item())
