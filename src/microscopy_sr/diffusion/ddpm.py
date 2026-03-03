from __future__ import annotations

import torch
import torch.nn.functional as F


class SRDiffusion:
    def __init__(self, model, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device) -> "SRDiffusion":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def p_losses(self, x0: torch.Tensor, cond_lr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, cond_lr, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond_lr: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
        device = cond_lr.device
        x = torch.randn(shape, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = self.model(x, cond_lr, t)

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            beta_t = self.betas[step]

            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.clamp(-1.0, 1.0)
