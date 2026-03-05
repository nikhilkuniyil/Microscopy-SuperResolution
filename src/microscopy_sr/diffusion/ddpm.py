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
    def sample(
        self,
        cond_lr: torch.Tensor,
        shape: tuple[int, int, int, int],
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample using DDIM (Song et al. 2021) for fast inference.

        Args:
            ddim_steps: Number of denoising steps (default 50, vs 1000 for DDPM).
                        Lower = faster but slightly lower quality.
            eta: Stochasticity. 0.0 = deterministic DDIM, 1.0 = full DDPM noise.
                 Use eta > 0 to preserve sample diversity for uncertainty estimation.
        """
        device = cond_lr.device
        x = torch.randn(shape, device=device)

        # Select evenly-spaced subset of timesteps
        step_indices = torch.linspace(0, self.timesteps - 1, ddim_steps + 1).long()
        timesteps = step_indices.flip(0)  # T -> 0

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i].item()
            t_prev = timesteps[i + 1].item()

            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self.model(x, cond_lr, t)

            alpha_bar_cur = self.alpha_bars[t_cur]
            alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Predicted x0
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_cur) * eps) / torch.sqrt(alpha_bar_cur)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_cur)) * torch.sqrt(1 - alpha_bar_cur / alpha_bar_prev)
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps + sigma * noise

        return x.clamp(-1.0, 1.0)
