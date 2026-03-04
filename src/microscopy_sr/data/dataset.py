from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MicroscopySuperResolutionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        patch_size: int = 128,
        upscale_factor: int = 4,
        few_shot_k: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.to_tensor = T.ToTensor()
        self.crop = (
            T.RandomCrop((patch_size, patch_size))
            if split == "train"
            else T.CenterCrop((patch_size, patch_size))
        )

        image_paths = []
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.dib", "*.DIB"):
            image_paths.extend(
                p for p in self.root_dir.rglob(ext)
                if "masks" not in p.parts
            )
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No images found in {self.root_dir}")

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(image_paths), generator=g).tolist()
        cutoff = int(len(image_paths) * train_ratio)

        if split == "train":
            idxs = perm[:cutoff]
        elif split == "val":
            idxs = perm[cutoff:]
        else:
            raise ValueError("split must be one of {'train', 'val'}")

        selected = [image_paths[i] for i in idxs]
        if few_shot_k is not None:
            selected = selected[:few_shot_k]

        if not selected:
            raise ValueError(f"No samples selected for split={split}")

        self.image_paths = selected

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        hr = self.to_tensor(self.crop(img))  # [1, H, W], 0..1

        h, w = hr.shape[-2:]
        lr_h, lr_w = h // self.upscale_factor, w // self.upscale_factor
        if lr_h < 1 or lr_w < 1:
            raise ValueError(
                f"Patch size {self.patch_size} too small for factor {self.upscale_factor}"
            )

        from torchvision.transforms.functional import gaussian_blur
        hr_blurred = gaussian_blur(hr, kernel_size=[3, 3], sigma=[1.0, 1.0])
        lr = F.interpolate(
            hr_blurred.unsqueeze(0), size=(lr_h, lr_w), mode="bicubic", align_corners=False
        ).squeeze(0)

        return {"hr": hr * 2.0 - 1.0, "lr": lr * 2.0 - 1.0, "path": str(path)}
