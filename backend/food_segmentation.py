"""Food Segmentation using Residual U-Net.

This module provides food outline detection from meal images using a
pre-trained Residual U-Net model. The model segments food regions from
the background, producing a binary mask that can be used for:
- Visual food outline overlay on the original image
- Portion size estimation
- Calorie prediction assistance

Model Architecture:
- Encoder: 3 stages (32→64→128 channels) with SE attention
- Bottleneck: 128 channels with SE attention (no extra pooling)
- Decoder: 2 stages (128→64→32 channels) with skip connections
- Output: 1 channel (food mask)
- Input: 256x256 RGB image
- Output: 256x256 binary mask
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import MODEL_DIR, settings

log = logging.getLogger("nutriai.segmentation")


class SEBlock:
    """Squeeze-and-Excitation block for channel attention."""
    pass


class ResidualBlock:
    """Residual block with SE attention."""
    pass


class FoodSegmentationModel:
    """Singleton wrapper around Residual U-Net for food segmentation."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self.backend = "unloaded"
        self.version = "n/a"

    def load(self) -> bool:
        """Load segmentation model. Returns True if loaded successfully."""
        if self._model is not None:
            return True

        checkpoint_path = Path(settings.segmentation_checkpoint)
        if not checkpoint_path.is_file():
            log.info("No segmentation checkpoint at %s — segmentation disabled", checkpoint_path)
            return False

        try:
            import torch
            import torch.nn as nn

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Define the model architecture
            model = self._build_model()

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            model.load_state_dict(checkpoint)
            model.to(self._device)
            model.eval()

            self._model = model
            self.backend = "residual_unet"
            self.version = f"residual_unet-{checkpoint_path.stem}"
            log.info("Segmentation model loaded from %s", checkpoint_path)
            return True

        except Exception as exc:
            log.warning("Segmentation model failed to load (%s)", exc)
            self.backend = "unloaded"
            return False

    def _build_model(self):
        """Build the Residual U-Net architecture matching the checkpoint.

        The checkpoint has 66 keys with this structure per block:
        - net.0: Conv2d (weight + bias)
        - net.1: BatchNorm (weight + bias, NO running stats)
        - net.3: Conv2d (weight + bias)
        - net.4: BatchNorm (weight + bias, NO running stats)
        - se.fc.2: Linear (weight only, no bias)
        - se.fc.4: Linear (weight only, no bias)

        No shortcut layers — identity residual via addition is NOT used.
        Forward pass uses 2 pools (not 3), no pool before mid,
        skip connections: e2→dec2, e1→dec1.
        """
        import torch.nn as nn

        class SEBlock(nn.Module):
            def __init__(self, channels, reduction):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.ReLU(),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
                return x * w

        class ResBlock(nn.Module):
            def __init__(self, in_ch, out_ch, se_reduction):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch, affine=True, track_running_stats=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch, affine=True, track_running_stats=False),
                )
                self.se = SEBlock(out_ch, se_reduction)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.se(self.net(x)))

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = ResBlock(3, 32, se_reduction=8)
                self.enc2 = ResBlock(32, 64, se_reduction=16)
                self.enc3 = ResBlock(64, 128, se_reduction=16)
                self.mid = ResBlock(128, 128, se_reduction=16)
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = ResBlock(128, 64, se_reduction=16)
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = ResBlock(64, 32, se_reduction=8)
                self.residual = nn.Conv2d(32, 1, 1)

            def forward(self, x):
                import torch
                import torch.nn.functional as F
                e1 = self.enc1(x)
                e2 = self.enc2(F.max_pool2d(e1, 2))
                e3 = self.enc3(F.max_pool2d(e2, 2))
                m = self.mid(e3)
                d2 = self.up2(m)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                d1 = self.dec1(torch.cat([d1, e1], dim=1))
                return torch.sigmoid(self.residual(d1))

        return UNet()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Predict food segmentation mask from image.

        Returns dict with 'mask', 'outline_overlay', 'food_area_ratio',
        'latency_ms', and 'engine' keys.
        """
        if self._model is None:
            return {
                "mask": None,
                "outline_overlay": None,
                "food_area_ratio": 0.0,
                "latency_ms": 0.0,
                "engine": "none",
            }

        import torch
        import torchvision.transforms as transforms

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            mask = self._model(input_tensor)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Convert to numpy
        mask_np = mask.squeeze().cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        # Calculate food area ratio
        food_area_ratio = float(mask_binary.sum() / (255 * 256 * 256))

        # Create outline overlay
        overlay = self._create_outline_overlay(image, mask_np)

        return {
            "mask": mask_binary,
            "outline_overlay": overlay,
            "food_area_ratio": round(food_area_ratio, 4),
            "latency_ms": round(latency_ms, 1),
            "engine": self.backend,
        }

    def _create_outline_overlay(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Create an overlay showing food outline on original image."""
        # Resize mask to original image size
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            image.size, Image.Resampling.NEAREST
        ))

        # Create outline by finding edges
        from scipy import ndimage
        outline = ndimage.binary_dilation(mask_resized > 127, iterations=3)
        outline = outline ^ (mask_resized > 127)

        # Create RGBA overlay
        overlay = image.convert("RGBA")
        outline_array = np.array(overlay)

        # Set outline pixels to green with transparency
        outline_array[outline, 0] = 143  # Green
        outline_array[outline, 1] = 208  # Green
        outline_array[outline, 2] = 63   # Green
        outline_array[outline, 3] = 200  # Alpha

        return Image.fromarray(outline_array, "RGBA")


# Module-level singleton
segmentation = FoodSegmentationModel()
