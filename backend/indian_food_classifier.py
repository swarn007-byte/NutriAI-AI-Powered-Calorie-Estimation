"""Indian Food Classification using EfficientNet.

This module provides food classification for 24 Indian dishes using a
pre-trained EfficientNet model. The model identifies the type of food
from a meal image and provides confidence scores.

Model Details:
- Architecture: EfficientNet (custom trained)
- Input: 300x300 RGB image
- Output: 24 Indian food classes
- Validation Top-1 Accuracy: 77.8%

Supported Indian Dishes:
- Aloo Gobi, Bhindi Masala, Butter Chicken, Chicken Curry
- Chole Masala, Dal Tadka, Dosa, Fish Curry
- French Fries, Grilled Chicken, Gulab Jamun, Idli
- Kheer, Mixed Veg Curry, Naan, Palak Paneer
- Paneer Butter Masala, Pasta Red Sauce, Pav Bhaji
- Pizza Slice, Poha, Poori, Roti Chapati, Samosa
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import MODEL_DIR, settings

log = logging.getLogger("nutriai.classification")


# Calorie lookup table for 24 Indian dishes (kcal per average serving)
INDIAN_FOOD_CALORIES = {
    "aloo_gobi": {"calories": 250, "serving_g": 200, "protein_g": 8, "carbs_g": 35, "fat_g": 10},
    "bhindi_masala": {"calories": 180, "serving_g": 180, "protein_g": 5, "carbs_g": 25, "fat_g": 8},
    "butter_chicken": {"calories": 450, "serving_g": 250, "protein_g": 35, "carbs_g": 15, "fat_g": 30},
    "chicken_curry": {"calories": 380, "serving_g": 250, "protein_g": 30, "carbs_g": 12, "fat_g": 25},
    "chole_masala": {"calories": 280, "serving_g": 200, "protein_g": 12, "carbs_g": 40, "fat_g": 8},
    "dal_tadka": {"calories": 220, "serving_g": 200, "protein_g": 15, "carbs_g": 35, "fat_g": 5},
    "dosa": {"calories": 180, "serving_g": 120, "protein_g": 5, "carbs_g": 30, "fat_g": 5},
    "fish_curry": {"calories": 320, "serving_g": 250, "protein_g": 28, "carbs_g": 10, "fat_g": 20},
    "french_fries": {"calories": 320, "serving_g": 150, "protein_g": 4, "carbs_g": 45, "fat_g": 15},
    "grilled_chicken": {"calories": 280, "serving_g": 200, "protein_g": 40, "carbs_g": 0, "fat_g": 12},
    "gulab_jamun": {"calories": 350, "serving_g": 100, "protein_g": 5, "carbs_g": 50, "fat_g": 15},
    "idli": {"calories": 120, "serving_g": 100, "protein_g": 4, "carbs_g": 25, "fat_g": 1},
    "kheer": {"calories": 280, "serving_g": 150, "protein_g": 8, "carbs_g": 40, "fat_g": 10},
    "mixed_veg_curry": {"calories": 200, "serving_g": 200, "protein_g": 6, "carbs_g": 30, "fat_g": 8},
    "naan": {"calories": 250, "serving_g": 100, "protein_g": 8, "carbs_g": 45, "fat_g": 5},
    "palak_paneer": {"calories": 350, "serving_g": 200, "protein_g": 18, "carbs_g": 15, "fat_g": 25},
    "paneer_butter_masala": {"calories": 400, "serving_g": 250, "protein_g": 20, "carbs_g": 20, "fat_g": 30},
    "pasta_red_sauce": {"calories": 350, "serving_g": 250, "protein_g": 12, "carbs_g": 50, "fat_g": 12},
    "pav_bhaji": {"calories": 400, "serving_g": 300, "protein_g": 10, "carbs_g": 55, "fat_g": 18},
    "pizza_slice": {"calories": 300, "serving_g": 150, "protein_g": 12, "carbs_g": 35, "fat_g": 12},
    "poha": {"calories": 180, "serving_g": 150, "protein_g": 4, "carbs_g": 35, "fat_g": 4},
    "poori": {"calories": 200, "serving_g": 80, "protein_g": 4, "carbs_g": 30, "fat_g": 8},
    "roti_chapati": {"calories": 120, "serving_g": 60, "protein_g": 4, "carbs_g": 22, "fat_g": 2},
    "samosa": {"calories": 250, "serving_g": 100, "protein_g": 5, "carbs_g": 30, "fat_g": 12},
}

INDIAN_FOOD_CLASSES = list(INDIAN_FOOD_CALORIES.keys())


class IndianFoodClassifier:
    """Singleton wrapper around EfficientNet for Indian food classification."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self.backend = "unloaded"
        self.version = "n/a"
        self._classes = INDIAN_FOOD_CLASSES

    def load(self) -> bool:
        """Load classification model. Returns True if loaded successfully."""
        if self._model is not None:
            return True

        checkpoint_path = Path(settings.classification_checkpoint)
        if not checkpoint_path.is_file():
            log.info("No classification checkpoint at %s — classification disabled", checkpoint_path)
            return False

        try:
            import torch
            import torchvision.models as models

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

            # Extract model info
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                self._classes = checkpoint.get("classes", INDIAN_FOOD_CLASSES)
                input_resolution = checkpoint.get("input_resolution", 300)
                self.version = checkpoint.get("version", "v1")
            else:
                state_dict = checkpoint
                input_resolution = 300

            # Build EfficientNet model
            model = models.efficientnet_b3(pretrained=False)
            num_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_features, len(self._classes))

            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.to(self._device)
            model.eval()

            self._model = {
                "model": model,
                "input_resolution": input_resolution,
            }
            self.backend = "efficientnet_b3"
            log.info("Classification model loaded from %s (classes=%d, resolution=%d)",
                     checkpoint_path, len(self._classes), input_resolution)
            return True

        except Exception as exc:
            log.warning("Classification model failed to load (%s)", exc)
            self.backend = "unloaded"
            return False

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Predict food class from image.

        Returns dict with 'class', 'confidence', 'all_predictions',
        'nutrition', 'latency_ms', and 'engine' keys.
        """
        if self._model is None:
            return {
                "class": "unknown",
                "confidence": 0.0,
                "all_predictions": [],
                "nutrition": None,
                "latency_ms": 0.0,
                "engine": "none",
            }

        import torch
        import torchvision.transforms as transforms

        model = self._model["model"]
        input_resolution = self._model["input_resolution"]

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Get top predictions
        probs, indices = torch.topk(probabilities, k=min(5, len(self._classes)))
        probs = probs.squeeze().cpu().numpy()
        indices = indices.squeeze().cpu().numpy()

        # Build predictions list
        all_predictions = []
        for prob, idx in zip(probs, indices):
            class_name = self._classes[idx]
            nutrition = INDIAN_FOOD_CALORIES.get(class_name, {})
            all_predictions.append({
                "class": class_name,
                "confidence": round(float(prob), 4),
                "nutrition": nutrition,
            })

        # Get top prediction
        top_class = self._classes[indices[0]]
        top_confidence = round(float(probs[0]), 4)
        nutrition = INDIAN_FOOD_CALORIES.get(top_class, {})

        return {
            "class": top_class,
            "confidence": top_confidence,
            "all_predictions": all_predictions,
            "nutrition": nutrition,
            "latency_ms": round(latency_ms, 1),
            "engine": self.backend,
        }


# Module-level singleton
classifier = IndianFoodClassifier()
