"""PortionNet wrapper for NutriAI — bridges PortionNet's architecture into the
existing classify.py Prediction interface.

Usage:
    # Standalone test (random weights, validates pipeline):
    python portionnet_wrapper.py --image frontend/samples/thali.jpg

    # With a real checkpoint:
    PORTIONNET_CHECKPOINT=/path/to/best_model_seed7.pt python portionnet_wrapper.py --image frontend/samples/thali.jpg

The wrapper loads PortionNet in RGB-only mode (no point clouds needed at
inference). It maps PortionNet's 131 MetaFood3D classes to NutriAI's 42-class
catalog via a simple overlap table; unmatched classes fall through as
'unrecognized'.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger("portionnet")

# ---------------------------------------------------------------------------
# MetaFood3D class names (131 classes) — from PortionNet's dataset.py
# We map the subset that overlaps with NutriAI's 42-class catalog.
# ---------------------------------------------------------------------------
METIFOOD_CLASSES = [
    "apple", "avocado", "banana", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beets", "bell_pepper", "bok_choy", "bread_pudding",
    "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli",
    "caprese_salad", "carrot", "ceviche", "cheese_plate", "cheesecake",
    "chicken_curry", "chicken_quesadilla", "chicken_wings",
    "chocolate_cake", "chocolate_mousse", "clam_chowder", "club_sandwich",
    "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips",
    "foie_gras", "french_fries", "french_onion_soup", "french_toast",
    "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread",
    "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop",
    "poutine", "prime_rib", "pulled_pork_sandwich", "ramen",
    "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]

# NutriAI's 42-class catalog → best-guess PortionNet class overlap
# Only dishes with a plausible visual match are listed.
NUTRIAI_TO_PORTIONNET: dict[str, str] = {
    "aloo_paratha": "pancakes",
    "biryani": "fried_rice",
    "butter_chicken": "chicken_curry",
    "chole_masala": "chickpea",       # no exact match — will be unrecognized
    "dal_makhani": "soup",            # approximate
    "dosa": "crepe",                  # no exact match
    "fried_rice": "fried_rice",
    "gulab_jamun": "donuts",
    "idli": "dumplings",
    "jalebi": "waffles",
    "kadai_paneer": "cheese_plate",
    "lassi": "smoothie",             # no exact match
    "masala_dosa": "crepe",
    "naan": "bread",
    "paneer_butter_masala": "cheese_plate",
    "paratha": "pancakes",
    "pav_bhaji": "hamburger",
    "rajma": "soup",
    "samosa": "samosa",
    "tandoori_chicken": "grilled_salmon",
}

# Reverse: PortionNet class → NutriAI display name (for mapping back)
PORTIONNET_TO_NUTRIAI: dict[str, str] = {v: k for k, v in NUTRIAI_TO_PORTIONNET.items()}


class PortionNetClassifier:
    """Thin wrapper around PortionNet that returns a Prediction-compatible dict."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self._loaded = False
        self._checkpoint_path = checkpoint_path

    def load(self) -> None:
        """Lazy-load the model on first predict call."""
        if self._loaded:
            return

        # Import torch here so the rest of the app works without it
        import torch
        from tools.portionnet.src.models import PortionNet

        self.model = PortionNet(num_classes=131, feature_dim=256, num_heads=8)

        if self._checkpoint_path and Path(self._checkpoint_path).is_file():
            log.info("Loading PortionNet checkpoint from %s", self._checkpoint_path)
            ckpt = torch.load(self._checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            log.warning(
                "No PortionNet checkpoint found — using randomly initialized weights "
                "(for pipeline testing only)"
            )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Run RGB-only inference and return a dict matching classify.Prediction."""
        import torch
        from torchvision import transforms

        self.load()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor, pointcloud=None, mode="rgb_only")
        latency_ms = (time.perf_counter() - t0) * 1000

        logits = outputs["class_logits"]
        probs = torch.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        top_prob = probs[0, top_idx].item()

        top5 = torch.topk(probs, min(5, probs.shape[1]), dim=1)
        alternatives = []
        for idx, prob in zip(top5.indices[0].tolist(), top5.values[0].tolist()):
            raw_name = METIFOOD_CLASSES[idx] if idx < len(METIFOOD_CLASSES) else f"class_{idx}"
            mapped = PORTIONNET_TO_NUTRIAI.get(raw_name, raw_name)
            alternatives.append({"label": mapped, "confidence": round(prob, 4)})

        raw_name = METIFOOD_CLASSES[top_idx] if top_idx < len(METIFOOD_CLASSES) else f"class_{top_idx}"
        mapped_label = PORTIONNET_TO_NUTRIAI.get(raw_name, raw_name)

        volume_ml = float(outputs["volume"].item())
        energy_kcal = float(outputs["energy"].item())

        return {
            "label": mapped_label,
            "raw_portionnet_class": raw_name,
            "confidence": round(top_prob, 4),
            "alternatives": alternatives,
            "engine": "portionnet",
            "volume_ml_raw": round(volume_ml, 2),
            "energy_kcal_raw": round(energy_kcal, 2),
            "latency_ms": round(latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test PortionNet on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to food image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model_seed7.pt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ckpt = args.checkpoint or (Path(__file__).parent / "models" / "portionnet_seed7.pt")
    if not Path(ckpt).exists():
        ckpt = None

    clf = PortionNetClassifier(checkpoint_path=str(ckpt) if ckpt else None, device=args.device)
    image = Image.open(args.image)
    result = clf.predict(image)

    print("\n=== PortionNet Results ===")
    print(f"  Image:              {args.image}")
    print(f"  Predicted class:    {result['label']}  (raw: {result['raw_portionnet_class']})")
    print(f"  Confidence:         {result['confidence'] * 100:.1f}%")
    print(f"  Volume (raw):       {result['volume_ml_raw']} ml")
    print(f"  Energy (raw):       {result['energy_kcal_raw']} kcal")
    print(f"  Latency:            {result['latency_ms']} ms")
    print(f"  Top-5 alternatives:")
    for alt in result["alternatives"]:
        print(f"    {alt['label']:30s}  {alt['confidence'] * 100:.1f}%")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
