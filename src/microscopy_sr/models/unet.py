from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        # (B, C, H, W) -> (B, H*W, C)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        qkv = self.qkv(h)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, C)

        head_dim = C // self.num_heads

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, H * W, self.num_heads, head_dim).permute(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        attn = attn.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.out_proj(attn)  # (B, N, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out  # residual


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, use_attn: bool = False) -> None:
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, t_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x, t_emb)
        skip = h                   # save skip at full (pre-downsample) resolution
        h = self.down(h)
        h = self.attn(h)           # attention AFTER downsampling (at lower resolution)
        return h, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int, use_attn: bool = False) -> None:
        super().__init__()
        self.attn = SelfAttention(in_ch) if use_attn else nn.Identity()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ResBlock(out_ch + skip_ch, out_ch, t_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)           # attention BEFORE upsampling (at lower resolution)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class ConditionalUNet(nn.Module):
    """
    4-level conditional U-Net for diffusion-based super-resolution.

    Architecture (for patch_size=128 HR input):
      Encoder: 128->64->32->16->8  (channel mults [1,2,4,8])
      Attention at 16x16 (down3 output) and 8x8 (down4 output + bottleneck)
      Decoder: 8->16->32->64->128  (symmetric attention)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        t_dim: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.t_dim = t_dim
        ch = [base_channels * m for m in channel_mults]  # [64, 128, 256, 512]

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoder: attention at ch[2] (16x16) and ch[3] (8x8)
        self.down1 = Down(ch[0], ch[0], t_dim, use_attn=False)  # 128->64, no attn
        self.down2 = Down(ch[0], ch[1], t_dim, use_attn=False)  # 64->32, no attn
        self.down3 = Down(ch[1], ch[2], t_dim, use_attn=True)   # 32->16, attn at 16x16
        self.down4 = Down(ch[2], ch[3], t_dim, use_attn=True)   # 16->8,  attn at 8x8

        # Bottleneck (8x8)
        self.mid1 = ResBlock(ch[3], ch[3], t_dim)
        self.mid_attn = SelfAttention(ch[3])
        self.mid2 = ResBlock(ch[3], ch[3], t_dim)

        # Decoder: symmetric attention at 8x8 (up4 input) and 16x16 (up3 input)
        # skip channel counts: s4=ch[3], s3=ch[2], s2=ch[1], s1=ch[0]
        self.up4 = Up(ch[3], ch[3], ch[2], t_dim, use_attn=True)   # attn at 8x8 input
        self.up3 = Up(ch[2], ch[2], ch[1], t_dim, use_attn=True)   # attn at 16x16 input
        self.up2 = Up(ch[1], ch[1], ch[0], t_dim, use_attn=False)
        self.up1 = Up(ch[0], ch[0], ch[0], t_dim, use_attn=False)

        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    def forward(self, x_noisy: torch.Tensor, cond_lr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = F.interpolate(cond_lr, size=x_noisy.shape[-2:], mode="bicubic", align_corners=False)
        x = torch.cat([x_noisy, cond], dim=1)

        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)

        x = self.in_conv(x)

        x, s1 = self.down1(x, t_emb)   # s1: (B, ch[0], 64, 64)
        x, s2 = self.down2(x, t_emb)   # s2: (B, ch[1], 32, 32)
        x, s3 = self.down3(x, t_emb)   # s3: (B, ch[2], 16, 16)
        x, s4 = self.down4(x, t_emb)   # s4: (B, ch[3], 8,  8)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up4(x, s4, t_emb)
        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        x = F.silu(self.out_norm(x))
        return self.out(x)
