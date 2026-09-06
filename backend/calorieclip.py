"""CalorieCLIP integration — direct calorie prediction from food images.

This module wraps the CalorieCLIP model (CLIP ViT-B/32 + regression head)
to provide direct calorie estimates without the depth → volume → weight chain.

The model is trained on Nutrition5k + Food-101 (13K images) and achieves:
- MAE: 51.4 calories
- 67.6% within ±50 cal
- 90.5% within ±100 cal

At inference: Image → CLIP encoder → regression head → calories (single float).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import MODEL_DIR, settings

log = logging.getLogger("nutriai.calorieclip")


class CalorieCLIPModel:
    """Singleton wrapper around CalorieCLIP for the pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self.backend = "unloaded"
        self.version = "n/a"

    def load(self) -> bool:
        """Load CalorieCLIP model. Returns True if loaded successfully."""
        if self._model is not None:
            return True

        checkpoint_path = Path(settings.calorieclip_checkpoint)
        if not checkpoint_path.is_file():
            log.info("No CalorieCLIP checkpoint at %s — calorie estimation disabled", checkpoint_path)
            return False

        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load CLIP
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )

            # Load regression head
            from models.calorie_clip import RegressionHead

            head = RegressionHead(input_dim=512)

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

            if "clip_state" in checkpoint:
                clip_model.load_state_dict(checkpoint["clip_state"], strict=False)
            if "regressor_state" in checkpoint:
                head.load_state_dict(checkpoint["regressor_state"])

            clip_model.to(self._device)
            clip_model.eval()
            head.to(self._device)
            head.eval()

            self._model = {
                "clip": clip_model,
                "head": head,
                "preprocess": preprocess,
            }

            epoch = checkpoint.get("epoch", "?")
            mae = checkpoint.get("mae", "?")
            self.version = f"calorieclip-epoch{epoch}-mae{mae}"
            self.backend = "calorieclip"
            log.info(
                "CalorieCLIP loaded from %s (epoch=%s, mae=%s)",
                checkpoint_path,
                epoch,
                mae,
            )
            return True

        except Exception as exc:
            log.warning("CalorieCLIP failed to load (%s)", exc)
            self.backend = "unloaded"
            return False

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Predict calories from a food image.

        Returns dict with 'calories', 'latency_ms', and 'engine' keys.
        """
        if self._model is None:
            return {"calories": 0.0, "latency_ms": 0.0, "engine": "none"}

        import torch

        clip_model = self._model["clip"]
        head = self._model["head"]
        preprocess = self._model["preprocess"]

        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            features = clip_model.encode_image(tensor).float()
            calories = head(features).item()
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "calories": round(max(0.0, calories), 1),
            "latency_ms": round(latency_ms, 1),
            "engine": self.backend,
        }

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Predict calories for a batch of images."""
        if self._model is None or not images:
            return [{"calories": 0.0, "latency_ms": 0.0, "engine": "none"} for _ in images]

        import torch

        clip_model = self._model["clip"]
        head = self._model["head"]
        preprocess = self._model["preprocess"]

        tensors = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            features = clip_model.encode_image(tensors).float()
            calories = head(features).squeeze(-1)
        latency_ms = (time.perf_counter() - t0) * 1000

        per_image_ms = latency_ms / len(images)
        return [
            {
                "calories": round(max(0.0, cal.item()), 1),
                "latency_ms": round(per_image_ms, 1),
                "engine": self.backend,
            }
            for cal in calories
        ]


# Module-level singleton
calorieclip = CalorieCLIPModel()
