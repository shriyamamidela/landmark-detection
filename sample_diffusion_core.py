"""
Minimal sampler for ATLAS-FLOW-DIFF
Uses: F (backbone features), E (edge maps), T (topology tokens)
Produces: Low-resolution DT map for SVF warping
"""

import torch
import torch.nn.functional as F

# ----------------------------------------------------------
# Beta Schedule (same as training)
# ----------------------------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


# ----------------------------------------------------------
# p_sample: one reverse diffusion step
# ----------------------------------------------------------
@torch.no_grad()
def p_sample(model, x_t, t, T_tok, extra_cond, betas, alphas, alphas_cumprod):
    """
    model: ConditionalUNet
    x_t: (B,1,H,W) noise sample
    t: scalar timestep tensor
    T_tok: (B,243) topology tokens
    extra_cond: (B, 512+3, Hf, Wf) [F || E]
    """

    # Run model on *latent resolution* input
    eps_theta = model(x_t, t.float(), T_tok, extra_cond=extra_cond)

    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]

    sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
    sqrt_beta = torch.sqrt(betas[t])

    # noise (0 if last step)
    z = torch.randn_like(x_t) if t > 0 else 0

    # Reverse DDPM equation
    x_prev = sqrt_recip_alpha * (
        x_t - ((1 - alpha_t) / sqrt_one_minus) * eps_theta
    ) + sqrt_beta * z

    return x_prev


# ----------------------------------------------------------
# sample_diffusion_core: main loop
# ----------------------------------------------------------
@torch.no_grad()
def sample_diffusion_core(
    model,
    F_cond,
    E_cond,
    T_tok,
    steps=20,
    device="cuda"
):
    """
    model: ConditionalUNet
    F_cond: (1,512,Hf,Wf)
    E_cond: (1,3,Hf,Wf)
    T_tok:  (1,243)
    returns:
        DT_lowres: (1,1,Hf,Wf)
    """

    # Build concatenated conditioning
    extra_cond = torch.cat([F_cond, E_cond], dim=1)  # (1, 515, Hf, Wf)

    # Infer latent resolution from F (C5 resolution)
    Hf, Wf = F_cond.shape[2:]
    shape = (1, 1, Hf, Wf)

    # DDPM noise parameters
    betas = get_beta_schedule(steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # start from pure noise at latent res
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(steps)):
        step = torch.tensor([t], device=device)
        x_t = p_sample(
            model, x_t, step, T_tok,
            extra_cond, betas, alphas, alphas_cumprod
        )

    return x_t  # (1,1,Hf,Wf)
