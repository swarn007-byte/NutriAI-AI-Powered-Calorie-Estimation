"""Stage 4 — Fine-grained dish classification (design.md §7.4 / §9).

**This is the one component in the project that is actually trained.** Detection,
depth and nutrition are pretrained models or lookups; the exact-dish decision —
dal vs sambhar, paneer vs tofu — is the EfficientNet-B3 head fine-tuned here.

Run `python classify.py --train --data ../data/processed` to train — or
`tools/train_kaggle.py` to do it on a Kaggle GPU, which is where the shipped
checkpoint comes from.

At inference time there are three engines behind one call, tried in this order
and loaded once at startup (design.md §12.1):

1. `RemoteClassifier` — a `model_api/` deployment named by `CLASSIFIER_URL`.
   Keeps torch out of this process entirely, so the API fits a 512 MB box.
2. A local `models/efficientnet_v*.pt` checkpoint.
3. `SignatureClassifier` — a transparent colour/texture/geometry prior over the
   same class list.

The prior is explicitly *not* presented as a trained model: every response
carries `engine: "signature"` so the UI can label it, and its confidence is
capped so §12.2's low-confidence review path stays active rather than showing a
confident wrong answer. The fallbacks exist so that a sleeping Space or a
half-finished deploy costs accuracy on one photo instead of returning an error.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from config import MODEL_DIR, PROJECT_DIR, settings
from imaging import rgb_to_lab, texture_energy

log = logging.getLogger("nutriai.classify")

INPUT_RESOLUTION = 300  # design.md §9
CONFIDENCE_CAP = 0.88  # the heuristic engine never claims near-certainty


@dataclass
class Prediction:
    label: str
    confidence: float
    alternatives: list[dict[str, Any]]
    engine: str


def _anchor(
    rgb: tuple[int, int, int],
    texture: float,
    area_frac: float,
    coarse: Sequence[str],
) -> dict[str, Any]:
    return {
        "rgb": rgb,
        "texture": texture,
        "area_frac": area_frac,
        "coarse": tuple(coarse),
    }


# Visual signature per class. Colours are representative sRGB anchors converted
# to CIE Lab at import; `texture` is the expected L*-standard-deviation of the
# region; `area_frac` is the typical share of the plate the item occupies.
SIGNATURES: dict[str, dict[str, Any]] = {
    "paneer_butter_masala": _anchor((198, 96, 52), 5.5, 0.17, ("gravy", "red_dish", "bowl")),
    "palak_paneer": _anchor((74, 98, 56), 6.5, 0.17, ("vegetable_green", "bowl")),
    "dal_tadka": _anchor((214, 168, 74), 5.0, 0.16, ("dal_or_yellow", "bowl")),
    "sambhar": _anchor((186, 110, 60), 5.0, 0.16, ("gravy", "dal_or_yellow", "bowl")),
    "chole_masala": _anchor((120, 76, 46), 8.5, 0.17, ("brown_dish", "gravy", "bowl")),
    "rajma_masala": _anchor((134, 62, 44), 8.0, 0.17, ("red_dish", "brown_dish", "bowl")),
    "aloo_gobi": _anchor((206, 176, 106), 10.0, 0.20, ("rice_or_bread", "dal_or_yellow")),
    "bhindi_masala": _anchor((88, 84, 46), 10.5, 0.18, ("vegetable_green", "dark_side")),
    "mixed_veg_curry": _anchor((176, 122, 76), 11.0, 0.20, ("mixed_dish", "gravy", "broccoli")),
    "butter_chicken": _anchor((206, 104, 62), 6.0, 0.18, ("gravy", "red_dish", "bowl")),
    "chicken_curry": _anchor((166, 100, 58), 8.0, 0.18, ("gravy", "brown_dish", "bowl")),
    "fish_curry": _anchor((196, 132, 70), 7.5, 0.17, ("gravy", "dal_or_yellow", "bowl")),
    "egg_curry": _anchor((190, 122, 78), 9.0, 0.17, ("gravy", "mixed_dish", "bowl")),
    "plain_rice": _anchor((238, 234, 222), 7.0, 0.30, ("rice_or_bread",)),
    "jeera_rice": _anchor((226, 214, 186), 8.0, 0.28, ("rice_or_bread",)),
    "veg_biryani": _anchor((214, 174, 114), 12.5, 0.30, ("rice_or_bread", "dal_or_yellow")),
    "chicken_biryani": _anchor((198, 152, 100), 13.0, 0.30, ("rice_or_bread", "brown_dish")),
    "roti_chapati": _anchor((214, 186, 146), 8.5, 0.26, ("rice_or_bread", "brown_dish")),
    "naan": _anchor((228, 202, 158), 9.0, 0.28, ("rice_or_bread",)),
    "paratha": _anchor((206, 168, 112), 10.5, 0.26, ("rice_or_bread", "brown_dish")),
    "poori": _anchor((208, 156, 92), 8.0, 0.18, ("brown_dish", "rice_or_bread")),
    "idli": _anchor((244, 242, 234), 4.5, 0.14, ("rice_or_bread",)),
    "dosa": _anchor((216, 170, 106), 9.5, 0.32, ("rice_or_bread", "brown_dish")),
    "medu_vada": _anchor((168, 116, 66), 9.0, 0.12, ("brown_dish", "donut")),
    "samosa": _anchor((196, 152, 96), 8.5, 0.14, ("brown_dish", "rice_or_bread")),
    "pav_bhaji": _anchor((168, 78, 48), 9.5, 0.20, ("red_dish", "gravy")),
    "upma": _anchor((222, 200, 148), 10.0, 0.22, ("rice_or_bread", "dal_or_yellow")),
    "poha": _anchor((228, 206, 148), 10.5, 0.22, ("rice_or_bread", "dal_or_yellow")),
    "curd_yogurt": _anchor((246, 245, 240), 3.5, 0.12, ("rice_or_bread", "cup", "bowl")),
    "raita": _anchor((238, 236, 224), 5.5, 0.12, ("rice_or_bread", "cup", "bowl")),
    "coconut_chutney": _anchor((226, 228, 206), 6.0, 0.08, ("rice_or_bread", "vegetable_green")),
    "green_salad": _anchor((124, 152, 84), 13.5, 0.16, ("vegetable_green", "carrot", "broccoli")),
    "papad": _anchor((226, 206, 168), 7.0, 0.16, ("rice_or_bread",)),
    "gulab_jamun": _anchor((128, 62, 32), 6.5, 0.10, ("brown_dish", "donut", "dark_side")),
    "kheer": _anchor((240, 228, 202), 5.0, 0.12, ("rice_or_bread", "cup")),
    "boiled_egg": _anchor((240, 232, 200), 6.0, 0.09, ("rice_or_bread",)),
    "grilled_chicken": _anchor((178, 130, 84), 9.5, 0.18, ("brown_dish", "hot dog")),
    "pasta_red_sauce": _anchor((182, 76, 52), 9.0, 0.24, ("red_dish", "gravy")),
    "pizza_slice": _anchor((196, 138, 84), 12.0, 0.30, ("pizza", "brown_dish")),
    "french_fries": _anchor((222, 178, 102), 11.5, 0.20, ("dal_or_yellow", "rice_or_bread")),
    "banana": _anchor((232, 208, 108), 5.0, 0.14, ("banana", "dal_or_yellow")),
    "apple": _anchor((190, 60, 52), 5.5, 0.12, ("apple", "red_dish")),
}

CLASS_LIST: tuple[str, ...] = tuple(SIGNATURES.keys())

_SIG_LAB = {
    key: rgb_to_lab(np.array([[[c / 255.0 for c in value["rgb"]]]], dtype=np.float32))[0, 0]
    for key, value in SIGNATURES.items()
}


class SignatureClassifier:
    """Transparent colour/texture/geometry prior over `CLASS_LIST`."""

    engine = "signature"
    version = "signature-v1"

    def predict(
        self,
        *,
        mean_lab: np.ndarray,
        texture: float,
        area_frac: float,
        coarse_label: str,
        top_k: int = 4,
    ) -> Prediction:
        scores: list[tuple[str, float]] = []
        for label, signature in SIGNATURES.items():
            sig_lab = _SIG_LAB[label]
            colour_distance = float(np.linalg.norm(mean_lab - sig_lab)) / 26.0
            texture_distance = abs(texture - float(signature["texture"])) / 7.0
            area_ratio = max(area_frac, 1e-3) / max(float(signature["area_frac"]), 1e-3)
            area_distance = abs(math.log(area_ratio)) * 0.9
            affinity = 1.05 if coarse_label in signature["coarse"] else 0.0
            scores.append(
                (label, -(colour_distance + 0.34 * texture_distance + 0.26 * area_distance) + affinity)
            )

        scores.sort(key=lambda row: row[1], reverse=True)
        # Softmax sharpness. With no labelled data available to calibrate
        # against, this is chosen rather than fitted: sharp enough that an
        # unambiguous match (colour, texture, footprint and coarse group all
        # agreeing) clears `unrecognized_threshold` and names a dish, flat
        # enough that everything else stays below `low_confidence_threshold`
        # and surfaces its alternatives instead. Reported confidences from this
        # engine are therefore ordinal, not calibrated probabilities — which is
        # why `engine` is reported as "heuristic" alongside them.
        raw = np.array([value for _, value in scores], dtype=np.float64) * 2.6
        raw -= raw.max()
        probabilities = np.exp(raw)
        probabilities /= probabilities.sum()

        # Honest ceiling: a hand-built prior should never report near-certainty.
        scale = min(1.0, CONFIDENCE_CAP / float(probabilities[0]))
        probabilities = probabilities * scale

        ranked = [
            {"label": label, "confidence": round(float(probability), 4)}
            for (label, _), probability in zip(scores, probabilities)
        ]
        return Prediction(
            label=ranked[0]["label"],
            confidence=float(ranked[0]["confidence"]),
            alternatives=ranked[1:top_k],
            engine=self.engine,
        )


class PortionNetClassifier:
    """PortionNet RGB-only classifier — bridges PortionNet into Prediction format.

    Uses the same architecture as PortionNet (ViT-B/16 + ResNet-18 dual encoder
    with cross-modal attention), but runs RGB-only at inference (no point clouds).
    Maps PortionNet's 131 MetaFood3D classes to NutriAI's 42-class catalog via
    an overlap table; unmatched classes fall through as 'unrecognized'.
    """

    engine = "portionnet"

    # MetaFood3D class names (131 classes) — from PortionNet's dataset
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

    # NutriAI → PortionNet class mapping (best-guess visual overlap)
    NUTRIAI_TO_PORTIONNET: dict[str, str] = {
        "aloo_paratha": "pancakes",
        "biryani": "fried_rice",
        "butter_chicken": "chicken_curry",
        "dal_makhani": "soup",
        "dosa": "crepe",
        "fried_rice": "fried_rice",
        "gulab_jamun": "donuts",
        "idli": "dumplings",
        "jalebi": "waffles",
        "kadai_paneer": "cheese_plate",
        "masala_dosa": "crepe",
        "naan": "bread",
        "paneer_butter_masala": "cheese_plate",
        "paratha": "pancakes",
        "pav_bhaji": "hamburger",
        "rajma": "soup",
        "samosa": "samosa",
        "tandoori_chicken": "grilled_salmon",
    }

    # Reverse mapping: PortionNet class → NutriAI label
    PORTIONNET_TO_NUTRIAI: dict[str, str] = {
        v: k for k, v in NUTRIAI_TO_PORTIONNET.items()
    }

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._loaded = False

    def load(self, checkpoint_path: str | None = None) -> bool:
        """Load PortionNet model. Returns True if loaded successfully."""
        if self._loaded:
            return True

        try:
            import torch
            from tools.portionnet.src.models import PortionNet

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = PortionNet(num_classes=131, feature_dim=256, num_heads=8)

            if checkpoint_path and Path(checkpoint_path).is_file():
                log.info("Loading PortionNet checkpoint from %s", checkpoint_path)
                ckpt = torch.load(checkpoint_path, map_location=self._device)
                self._model.load_state_dict(ckpt["model_state_dict"])
            else:
                log.warning(
                    "No PortionNet checkpoint — using randomly initialized weights "
                    "(pipeline testing only)"
                )

            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            return True

        except Exception as exc:
            log.warning("PortionNet failed to load (%s)", exc)
            return False

    def predict(self, image: Image.Image, *, top_k: int = 4) -> Prediction:
        """Run RGB-only inference and return a Prediction."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor, pointcloud=None, mode="rgb_only")

        logits = outputs["class_logits"]
        probs = torch.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        top_prob = probs[0, top_idx].item()

        # Map to NutriAI class
        raw_name = self.METIFOOD_CLASSES[top_idx] if top_idx < len(self.METIFOOD_CLASSES) else f"class_{top_idx}"
        mapped_label = self.PORTIONNET_TO_NUTRIAI.get(raw_name, "unrecognized")

        # Top-k alternatives
        top5 = torch.topk(probs, min(top_k, probs.shape[1]), dim=1)
        alternatives = []
        for idx, prob in zip(top5.indices[0].tolist(), top5.values[0].tolist()):
            raw = self.METIFOOD_CLASSES[idx] if idx < len(self.METIFOOD_CLASSES) else f"class_{idx}"
            mapped = self.PORTIONNET_TO_NUTRIAI.get(raw, "unrecognized")
            alternatives.append({"label": mapped, "confidence": round(prob, 4)})

        return Prediction(
            label=mapped_label,
            confidence=round(top_prob, 4),
            alternatives=alternatives[1:],  # exclude top-1 from alternatives
            engine=self.engine,
        )


class RemoteUnavailable(RuntimeError):
    """Raised inside `RemoteClassifier`; never allowed to escape `DishClassifier`."""


class RemoteClassifier:
    """Stage 4 against a `model_api/` deployment instead of local weights.

    The point is what this process *doesn't* need: torch is ~2.5 GB of wheels and
    a few hundred MB resident, for one stage out of five. Hosting the classifier
    separately lets the API run on a 512 MB instance (config.classifier_url).

    Every failure here is non-fatal by construction — `DishClassifier` keeps a
    local checkpoint or the signature prior behind this one and drops to it, so
    an asleep Space costs accuracy on one analysis rather than returning an
    error. The failure that *would* be user-visible is paying the timeout on
    every request while the service is down: six crops against a dead host is
    indistinguishable from an outage even though the fallback works fine. Hence
    the breaker below.
    """

    engine = "efficientnet_b3"

    # A host that failed once will usually fail again, so stop asking for a
    # while and answer from the fallback immediately.
    FAILURE_THRESHOLD = 3
    COOLDOWN_S = 120.0
    # Crops go over the wire as JPEG at the model's own input resolution. Both
    # halves matter: a crop out of a 1600px photo can be 500 KB as PNG and is
    # ~20 KB this way, and the service resizes to this exact size anyway — PIL
    # short-circuits a same-size resize, so pre-resizing changes the payload
    # without changing the arithmetic. JPEG quantisation at q92 is far below
    # the model's own error.
    JPEG_QUALITY = 92

    def __init__(self, url: str, *, token: str = "", timeout_s: float = 25.0) -> None:
        # Accept either the base URL or the /classify URL, because both are
        # things a person reasonably pastes out of a browser.
        base = url.rstrip("/")
        if base.endswith("/classify"):
            base = base[: -len("/classify")]
        self.base_url = base
        self.classify_url = f"{base}/classify"
        self.health_url = f"{base}/health"
        self.timeout_s = timeout_s
        self.version = "remote"
        self.classes: tuple[str, ...] = ()
        self.input_resolution = INPUT_RESOLUTION
        self.last_error: str | None = None
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._failures = 0
        self._cooldown_until = 0.0
        self._client = None

    # -- plumbing --------------------------------------------------------
    def _http(self):
        """One pooled client for the process.

        A new connection per request means a fresh TLS handshake to the Space
        every time, which on a cross-region hop costs more than the inference.
        """
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s, follow_redirects=True)
        return self._client

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._cooldown_until

    def _succeeded(self) -> None:
        self._failures = 0
        self._cooldown_until = 0.0
        self.last_error = None

    def _failed(self, reason: str) -> None:
        self.last_error = reason
        self._failures += 1
        if self._failures >= self.FAILURE_THRESHOLD:
            self._cooldown_until = time.monotonic() + self.COOLDOWN_S
            log.warning(
                "Remote classifier failed %d times (%s) — not calling it for %.0fs",
                self._failures,
                reason,
                self.COOLDOWN_S,
            )

    # -- lifecycle -------------------------------------------------------
    def _identify(self) -> tuple[dict[str, Any] | None, str | None]:
        """Read /health for the version, class list and input size.

        Returns `(body, error)` and deliberately touches neither the breaker nor
        `last_error`: this is bookkeeping for what `/api/health` reports, not a
        judgement about whether the remote works. Conflating the two made a
        successful prediction followed by a slow /health look like a failure.
        """
        try:
            response = self._http().get(self.health_url, headers=self._headers)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

        self.version = str(body.get("version") or "remote")
        self.classes = tuple(body.get("classes") or ())
        self.input_resolution = int(body.get("input_resolution") or INPUT_RESOLUTION)
        return body, None

    def probe(self) -> bool:
        """Identify the service and report whether it can actually serve.

        A False here is not a reason to stop using the remote: on a free Space
        this call is what *wakes* the container, so the expected first result is
        a timeout followed by a working service a minute later.
        """
        body, error = self._identify()
        if body is None:
            self.last_error = error
            log.warning("Remote classifier %s did not answer /health (%s)", self.health_url, error)
            return False

        if not body.get("ready"):
            # Reachable but unweighted. Reported rather than retried: this is a
            # configuration mistake on the far end, and it will not fix itself.
            self.last_error = str(body.get("error") or "remote model is not loaded")
            log.warning("Remote classifier is up but not ready: %s", self.last_error)
            return False
        log.info(
            "Remote classifier ready at %s — %s, %d classes, %dpx",
            self.base_url,
            self.version,
            len(self.classes),
            self.input_resolution,
        )
        self._succeeded()
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- inference -------------------------------------------------------
    def _encode(self, crop: Image.Image) -> bytes:
        buffer = io.BytesIO()
        resized = crop.convert("RGB").resize(
            (self.input_resolution, self.input_resolution), Image.BICUBIC
        )
        resized.save(buffer, "JPEG", quality=self.JPEG_QUALITY)
        return buffer.getvalue()

    def predict(self, crops: Sequence[Image.Image], *, top_k: int = 4) -> list[Prediction]:
        """One request for the whole plate. Raises `RemoteUnavailable` on any problem.

        Batched because the alternative is up to `max_items_per_plate` round
        trips per photo, and the round trip — not the forward pass — is what
        costs the time when the model is on another host.
        """
        if not crops:
            return []
        if not self.available:
            raise RemoteUnavailable(f"in cooldown after {self._failures} failures")

        files = [
            ("images", (f"crop{index}.jpg", self._encode(crop), "image/jpeg"))
            for index, crop in enumerate(crops)
        ]
        try:
            response = self._http().post(
                self.classify_url,
                files=files,
                data={"top_k": str(max(1, top_k))},
                headers=self._headers,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._failed(reason)
            raise RemoteUnavailable(reason) from exc

        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(crops):
            # A shape mismatch would otherwise mean labels attached to the wrong
            # items, which is worse than no answer because it looks plausible.
            reason = f"expected {len(crops)} results, got {type(results).__name__} {len(results or [])}"
            self._failed(reason)
            raise RemoteUnavailable(reason)

        version = str(body.get("version") or self.version)
        if version != self.version and self.version != "remote":
            # Worth an INFO: it means a new checkpoint rolled out under a live
            # backend. Skipped when the old value is the "remote" placeholder,
            # which only means /health was never reached — not a change.
            log.info("Remote classifier version changed: %s -> %s", self.version, version)
        self.version = version

        predictions = []
        for entry in results:
            alternatives = [
                {"label": str(item["label"]), "confidence": float(item["confidence"])}
                for item in (entry.get("alternatives") or [])
                if isinstance(item, dict) and "label" in item and "confidence" in item
            ]
            predictions.append(
                Prediction(
                    label=str(entry["label"]),
                    confidence=float(entry["confidence"]),
                    alternatives=alternatives[: max(0, top_k - 1)],
                    # Marked so /api/health and model_versions say where the
                    # answer came from — the same weights, a different machine.
                    engine=f"{self.engine}@remote",
                )
            )
        self._succeeded()
        if not self.classes:
            # The startup probe never got through — the ordinary case for a Space
            # that was asleep then — so the class list is still unknown. It is
            # knowable now that the service is demonstrably up, and one request
            # buys `/api/health` an accurate class count for the rest of the
            # process instead of a stale guess. Metadata only: a failure here
            # must not mark a remote that just worked as broken.
            self._identify()
        return predictions


class DishClassifier:
    """Stage-4 singleton. Trained weights if they are reachable, else the prior.

    Three engines behind one call, tried most-capable first: a hosted
    `model_api/` deployment, a local checkpoint, and the signature prior. The
    first two run the same trained weights; the prior is a genuinely worse
    answer, so it is only correct as a last resort.
    """

    def __init__(self) -> None:
        self.backend = "unloaded"
        self._model = None
        self._remote: RemoteClassifier | None = None
        self._portionnet: PortionNetClassifier | None = None
        self._signature = SignatureClassifier()
        # Only what this process knows locally. What gets *reported* is the
        # properties below, which prefer the engine that actually answers.
        self._local_version = "n/a"
        self._local_classes: tuple[str, ...] = CLASS_LIST

    # -- what is being reported ------------------------------------------
    # Read through to the remote rather than snapshotted at load(). A free Space
    # is usually asleep when this process starts, so the startup probe fails and
    # the version and class list are not known yet — they are learned by the
    # first request that succeeds. Copying them once at load() left /api/health
    # reporting "remote" and all 42 signature classes forever, while real
    # predictions came back tagged with the checkpoint's actual version.

    @property
    def version(self) -> str:
        if self._remote is not None:
            return self._remote.version
        return self._local_version

    @property
    def classes(self) -> tuple[str, ...]:
        if self._remote is not None and self._remote.classes:
            return self._remote.classes
        return self._local_classes

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        primary: str | None = None

        if settings.classifier_url:
            remote = RemoteClassifier(
                settings.classifier_url,
                token=settings.classifier_token,
                timeout_s=settings.classifier_timeout_s,
            )
            # Kept even when the probe fails. A free Space sleeps after
            # inactivity and the probe is the request that wakes it, so
            # "unreachable at startup" is the normal state, not a fault. The
            # fallbacks below cover the wait, and the breaker covers a real
            # outage.
            remote.probe()
            self._remote = remote
            primary = f"{remote.engine}@remote"

        # --- PortionNet (opt-in via CLASSIFIER_ENGINE=portionnet) -----------
        if settings.classifier_engine == "portionnet":
            pn = PortionNetClassifier()
            if pn.load(settings.portionnet_checkpoint):
                self._portionnet = pn
                primary = primary or "portionnet"
                log.info("PortionNet loaded (RGB-only mode)")
            else:
                log.warning("PortionNet failed to load — falling back")

        checkpoint_path = Path(settings.classifier_checkpoint)
        if settings.enable_torch_models and checkpoint_path.is_file():
            try:
                self._load_checkpoint(checkpoint_path)
                # Only claims the primary slot if no remote took it; otherwise
                # it is the fallback, and reporting it as primary would be a lie
                # about which weights produced a given answer.
                primary = primary or "efficientnet_b3"
                log.info("Local EfficientNet-B3 checkpoint loaded (%s)", self._local_version)
            except Exception as exc:
                log.warning("Classifier checkpoint at %s failed to load (%s)", checkpoint_path, exc)
        elif primary is None:
            log.info("No classifier checkpoint at %s — using signature prior", checkpoint_path)

        if primary is None:
            self.backend = self._signature.engine
            self._local_version = self._signature.version
        else:
            self.backend = primary

    def _load_checkpoint(self, path: Path) -> None:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        classes = tuple(payload.get("classes") or CLASS_LIST)
        model = _build_backbone(len(classes), pretrained=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._model = model
        self._local_classes = classes
        self._local_version = str(payload.get("version") or path.stem)

    @property
    def ready(self) -> bool:
        return self.backend != "unloaded"

    @property
    def is_trained_model(self) -> bool:
        """True when answers come from fine-tuned weights, wherever they run."""
        return self._model is not None or self._remote is not None

    @property
    def fallbacks(self) -> list[str]:
        """Engines behind the primary, in the order they would be tried."""
        chain = []
        if self._remote is not None and self._model is not None:
            chain.append("efficientnet_b3")
        if self.backend != self._signature.engine:
            chain.append(self._signature.engine)
        return chain

    # -- inference -------------------------------------------------------
    def predict_crop(
        self,
        crop: Image.Image,
        *,
        coarse_label: str,
        area_frac: float,
        top_k: int = 4,
    ) -> Prediction:
        """Single-crop convenience wrapper. Prefer `predict_crops` for a plate."""
        return self.predict_crops(
            [crop], coarse_labels=[coarse_label], area_fracs=[area_frac], top_k=top_k
        )[0]

    def predict_crops(
        self,
        crops: Sequence[Image.Image],
        *,
        coarse_labels: Sequence[str],
        area_fracs: Sequence[float],
        top_k: int = 4,
    ) -> list[Prediction]:
        """Classify every crop from one plate together.

        Batched at this level rather than per item because both trained paths
        get materially cheaper for it — one HTTP round trip instead of six, one
        forward pass over a stacked tensor instead of six — and because the
        fallback decision belongs to the whole plate: a half-remote,
        half-signature result set would mix calibrated probabilities with
        ordinal scores and then compare them against one threshold.
        """
        if not crops:
            return []

        if self._remote is not None and self._remote.available:
            try:
                return self._remote.predict(crops, top_k=top_k)
            except RemoteUnavailable as exc:
                log.warning("Remote classifier unusable (%s) — falling back", exc)

        # PortionNet: one prediction per crop (no batching yet)
        if self._portionnet is not None:
            try:
                return [self._portionnet.predict(crop, top_k=top_k) for crop in crops]
            except Exception as exc:
                log.warning("PortionNet inference failed (%s) — falling back", exc)

        if self._model is not None:
            try:
                return self._predict_torch(crops, top_k=top_k)
            except Exception as exc:
                log.warning("Classifier inference failed (%s) — using signature prior", exc)

        return [
            self._predict_signature(crop, coarse_label=coarse, area_frac=area, top_k=top_k)
            for crop, coarse, area in zip(crops, coarse_labels, area_fracs)
        ]

    def _predict_signature(
        self, crop: Image.Image, *, coarse_label: str, area_frac: float, top_k: int
    ) -> Prediction:
        array = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
        lab = rgb_to_lab(array)
        return self._signature.predict(
            mean_lab=lab.reshape(-1, 3).mean(axis=0),
            texture=float(np.std(lab[..., 0])) or texture_energy(lab[..., 0]),
            area_frac=area_frac,
            coarse_label=coarse_label,
            top_k=top_k,
        )

    def _predict_torch(self, crops: Sequence[Image.Image], *, top_k: int) -> list[Prediction]:
        import torch

        # Average probabilities across deterministic views. A horizontal flip
        # preserves dish identity while reducing sensitivity to composition and
        # the direction a phone happened to be held.
        views = [crops]
        if settings.classifier_tta_passes >= 2:
            views.append([crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for crop in crops])
        batches = [torch.stack([_preprocess(crop) for crop in view]) for view in views]
        with torch.no_grad():
            probabilities = np.mean(
                [torch.softmax(self._model(batch), dim=1).cpu().numpy() for batch in batches],
                axis=0,
            )  # type: ignore[misc]

        predictions = []
        for row in probabilities:
            order = np.argsort(row)[::-1][: max(top_k, 1)]
            ranked = [
                {"label": self.classes[int(index)], "confidence": round(float(row[index]), 4)}
                for index in order
            ]
            predictions.append(
                Prediction(
                    label=ranked[0]["label"],
                    confidence=float(ranked[0]["confidence"]),
                    alternatives=ranked[1:top_k],
                    engine="efficientnet_b3",
                )
            )
        return predictions


# --------------------------------------------------------------------------
# Preprocessing / backbone (shared by training and inference)
# --------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image, size: int = INPUT_RESOLUTION):
    import torch

    resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
    array = (np.asarray(resized, dtype=np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _build_backbone(num_classes: int, *, pretrained: bool = True):
    """EfficientNet-B3 with a fresh head. Prefers timm, falls back to torchvision."""
    try:
        import timm

        return timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=num_classes)
    except ImportError:
        pass
    try:
        from torchvision.models import efficientnet_b3
        import torch.nn as nn

        weights = "IMAGENET1K_V1" if pretrained else None
        model = efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    except ImportError as exc:
        raise RuntimeError(
            "EfficientNet-B3 needs `timm` or `torchvision` installed:\n"
            "  pip install timm            # or: pip install torchvision"
        ) from exc


def _freeze_early_layers(model, keep_fraction: float = 0.30) -> tuple[int, int]:
    """Freeze roughly the first 70% of parameter groups (design.md §9)."""
    groups = [(name, param) for name, param in model.named_parameters()]
    cutoff = int(len(groups) * (1.0 - keep_fraction))
    frozen = 0
    for index, (name, param) in enumerate(groups):
        trainable = index >= cutoff or "classifier" in name or "head" in name or name.startswith("fc")
        param.requires_grad = trainable
        frozen += 0 if trainable else 1
    return frozen, len(groups) - frozen


# --------------------------------------------------------------------------
# Training (design.md §9)
# --------------------------------------------------------------------------

def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Randomised photometric and geometric jitter, driven by one seeded RNG.

    Deliberately still PIL and still fed from a `random.Random` the caller owns,
    rather than `torchvision.transforms`. Every draw comes from that one generator,
    which is what makes a batch reproducible no matter how many decode threads
    assembled it — `test_augmentation_is_independent_of_worker_count` holds this
    down, and a transform pipeline with its own global RNG would break it.

    Wider than v1's. Its jitter was mild enough that the model saw nearly the same
    photograph each epoch, and with a hundred-odd images per class that invites
    memorisation. The additions target how food photographs actually vary: white
    balance swings between kitchen and daylight (colour and contrast), phones are
    held at an angle (shear), and a plate is often partly occluded by a hand or
    another dish (erasing a block).
    """
    from PIL import ImageEnhance

    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-18.0, 18.0)
    image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    # A small shear, standing in for the camera not being straight above the
    # plate. Kept mild: food seen from a steep angle is a different problem, not
    # an augmentation of this one.
    if rng.random() < 0.3:
        width, height = image.size
        shear = rng.uniform(-0.12, 0.12)
        image = image.transform(
            (width, height),
            Image.AFFINE,
            (1.0, shear, -shear * height / 2.0, 0.0, 1.0, 0.0),
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255),
        )

    scale = rng.uniform(0.70, 1.0)
    width, height = image.size
    crop_w, crop_h = int(width * scale), int(height * scale)
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.70, 1.35))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.80, 1.25))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.60, 1.60))

    # Occlusion. Done here in PIL rather than as tensor-space random erasing so it
    # happens before the resize and normalise, and so it stays on this one RNG.
    if rng.random() < 0.25:
        width, height = image.size
        block_w = int(width * rng.uniform(0.10, 0.28))
        block_h = int(height * rng.uniform(0.10, 0.28))
        if block_w > 0 and block_h > 0:
            x0 = rng.randint(0, max(0, width - block_w))
            y0 = rng.randint(0, max(0, height - block_h))
            grey = rng.randint(96, 160)
            image.paste((grey, grey, grey), (x0, y0, x0 + block_w, y0 + block_h))
    return image


def _discover_dataset(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    """Read an ImageFolder-style tree: `<root>/<class_name>/*.jpg`."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_names:
        raise SystemExit(f"No class folders under {root}. Expected {root}/<dish_name>/*.jpg")
    samples: list[tuple[Path, int]] = []
    for index, name in enumerate(class_names):
        for file in sorted((root / name).rglob("*")):
            if file.suffix.lower() in extensions:
                samples.append((file, index))
    if not samples:
        raise SystemExit(f"Found {len(class_names)} class folders under {root} but no images.")
    return samples, class_names


def _stratified_split(
    samples: list[tuple[Path, int]], seed: int = 42
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """70/15/15 stratified split (design.md §9)."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []
    for rows in by_class.values():
        rng.shuffle(rows)
        n = len(rows)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15))) if n - n_train > 1 else 0
        train += rows[:n_train]
        val += rows[n_train : n_train + n_val]
        test += rows[n_train + n_val :]
    rng.shuffle(train)
    return train, val, test


def _device(prefer: str | None = None):
    """Pick the training device. CUDA when present — a Kaggle T4/P100 is the
    intended host — then Apple MPS, then CPU."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(
    rows: list[tuple[Path, int]],
    batch_size: int,
    *,
    augment: bool,
    seed: int,
    device=None,
    workers: int = 4,
):
    """Yield `(inputs, targets)` on `device`, decoding images ahead of the GPU.

    JPEG decode plus PIL augmentation costs more per image than the forward and
    backward pass does, so a synchronous loader leaves the GPU idle most of the
    wall clock. Two batches of images stay in flight in a thread pool — PIL
    releases the GIL during decode, so this parallelises for real — and the queue
    is refilled as it drains rather than per batch, which keeps the pool busy
    across the batch boundary instead of stalling on the slowest image in a chunk.
    """
    import torch

    def load(job: tuple[int, tuple[Path, int]]):
        index, (path, label) = job
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            log.warning("Skipping unreadable image %s (%s)", path, exc)
            return None
        if augment:
            # A generator per sample, seeded from the position: threads must not
            # share one `Random`, and this also makes a run reproducible whatever
            # order the pool happens to finish in.
            image = _augment(image, random.Random((seed * 1_000_003) ^ index))
        return _preprocess(image), label

    def collate(items):
        inputs = torch.stack([tensor for tensor, _ in items])
        targets = torch.tensor([label for _, label in items], dtype=torch.long)
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        return inputs, targets

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque = deque()
        cursor = 0
        depth = batch_size * 2
        while cursor < len(rows) and len(pending) < depth:
            pending.append(pool.submit(load, (cursor, rows[cursor])))
            cursor += 1

        while pending:
            items = []
            while pending and len(items) < batch_size:
                result = pending.popleft().result()
                if cursor < len(rows):
                    pending.append(pool.submit(load, (cursor, rows[cursor])))
                    cursor += 1
                if result is not None:
                    items.append(result)
            if items:
                yield collate(items)


def _evaluate(model, rows: list[tuple[Path, int]], class_names: list[str], batch_size: int, device=None):
    import torch

    if not rows:
        return 0.0, {}
    model.eval()
    correct = 0
    total = 0
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in class_names}
    autocast = _autocast(device)
    with torch.no_grad():
        for inputs, targets in _batches(rows, batch_size, augment=False, seed=0, device=device):
            with autocast():
                predictions = model(inputs).argmax(dim=1)
            for prediction, target in zip(predictions.tolist(), targets.tolist()):
                total += 1
                if prediction == target:
                    correct += 1
                    per_class[class_names[target]]["tp"] += 1
                else:
                    per_class[class_names[prediction]]["fp"] += 1
                    per_class[class_names[target]]["fn"] += 1
    f1: dict[str, float] = {}
    for name, counts in per_class.items():
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        f1[name] = round((2 * counts["tp"] / denominator) if denominator else 0.0, 4)
    return (correct / total if total else 0.0), f1


def _autocast(device):
    """Mixed precision on CUDA only. On CPU autocast to bfloat16 is slower than
    plain float32 for this model, and on MPS it is not reliable."""
    import contextlib

    import torch

    if device is not None and getattr(device, "type", None) == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext


def _warm_start(model, checkpoint: Path, class_names: list[str]) -> dict[str, Any]:
    """Start from an earlier checkpoint instead of bare ImageNet weights.

    Two things make this more than a `load_state_dict`. The class list moves
    between runs — widening the dataset adds labels — so the head's row count
    changes and a strict load would refuse the entire file over its last tensor.
    And the rows that *do* survive are worth keeping: a head row is what the model
    learned about one dish, so `samosa`'s row is still about samosas even after six
    new labels pushed its index along. Rows are therefore matched by class name,
    never by position; matching by position would quietly relabel most of the head
    and look like a mysteriously bad initialisation.

    The head is found by shape rather than by name — the last 2-D tensor whose
    first dimension is the old class count — because `classifier.weight`,
    `classifier.1.weight` and `head.fc.weight` are all the same thing under three
    architectures, and the next run may not use this one.

    Anything that does not line up keeps its freshly-initialised value and is
    reported. A silent shape mismatch here surfaces much later as an accuracy that
    will not move, which is an expensive way to find out.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incoming = payload.get("state_dict", payload)
    old_classes = list(payload.get("classes") or [])
    current = model.state_dict()

    head_keys: list[str] = []
    if old_classes:
        head_keys = [
            key
            for key, tensor in incoming.items()
            if getattr(tensor, "ndim", 0) == 2 and tensor.shape[0] == len(old_classes)
        ]
    head_weight = head_keys[-1] if head_keys else None
    head_bias = None
    if head_weight is not None:
        candidate = head_weight.rsplit(".", 1)[0] + ".bias"
        if candidate in incoming and getattr(incoming[candidate], "ndim", 0) == 1:
            head_bias = candidate

    loaded, reshaped, skipped = [], [], []
    for key, tensor in incoming.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape == tensor.shape:
            current[key] = tensor.clone()
            loaded.append(key)
        elif key in (head_weight, head_bias):
            reshaped.append(key)
        else:
            skipped.append(key)

    # Transplant the head row by row, for the labels both runs share.
    if reshaped and old_classes:
        index = {name: position for position, name in enumerate(class_names)}
        for key in reshaped:
            destination = current[key].clone()
            for old_position, name in enumerate(old_classes):
                new_position = index.get(name)
                if new_position is not None and old_position < incoming[key].shape[0]:
                    destination[new_position] = incoming[key][old_position]
            current[key] = destination

    model.load_state_dict(current)

    # An unchanged class list means the head kept its shape and arrived through
    # the verbatim branch, never the transplant one. Counting only transplants
    # would then report "0 rows reused" for the case where every row was reused,
    # which is precisely the log line a warm start is read to check.
    shared = [name for name in old_classes if name in set(class_names)]
    head_arrived = head_weight is not None and (head_weight in loaded or head_weight in reshaped)
    report = {
        "checkpoint": str(checkpoint),
        "from_version": payload.get("version"),
        "from_test_top1": payload.get("test_top1"),
        "tensors_loaded": len(loaded),
        "tensors_skipped": sorted(skipped),
        "head_rows_reused": len(shared) if head_arrived else 0,
        "head_rows_new": [name for name in class_names if name not in set(old_classes)],
    }
    log.info(
        "Warm start from %s (%s, test top-1 %s): %d tensors, %d/%d head rows reused",
        Path(checkpoint).name,
        report["from_version"] or "unversioned",
        report["from_test_top1"] if report["from_test_top1"] is not None else "n/a",
        len(loaded),
        report["head_rows_reused"],
        len(class_names),
    )
    if skipped:
        log.warning("Warm start ignored %d tensor(s): %s", len(skipped), ", ".join(sorted(skipped)[:6]))
    return report


def _param_groups(model, *, base_lr: float, weight_decay: float, layer_decay: float) -> list[dict]:
    """Shallow layers learn slower than deep ones, and norms/biases skip decay.

    v1 froze the first 70% of parameter groups, which is this same idea with the
    dial at zero — a frozen layer is one whose learning rate is exactly 0. It cost
    real accuracy. Across v1's last nine epochs the training loss moved 0.515 →
    0.498 and stopped 0.155 above the floor that label smoothing puts underneath
    it: that is a model which cannot fit its own training data, not one that has
    run out of things to learn. Food photographs sit far enough from ImageNet that
    the mid-level features have to move, and frozen ones cannot.

    Layer-wise decay is the same intuition with the dial turned up. Every layer
    trains, but the shallow ones — edges, colour, texture, largely transferable
    already — get a fraction of the head's rate, so they adapt without being torn
    apart by the early gradients of a freshly-initialised classifier. Set
    `layer_decay=1.0` to train everything at one rate.

    Depth is read off the order of `named_parameters()`, which follows the forward
    pass for every backbone this project might use, rather than from an
    architecture-specific layer map — the next run may not be an EfficientNet.

    Weight decay is skipped for anything one-dimensional. Norm scales and biases
    have no scale-invariance for decay to exploit, so shrinking them does not
    regularise, it just drags the normalisation statistics off centre.
    """
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("Nothing to train: every parameter is frozen.")

    buckets = min(12, len(trainable))
    groups: list[dict] = []
    for position, (name, param) in enumerate(trainable):
        depth = (position * buckets) // len(trainable)  # 0 = shallowest
        scale = layer_decay ** (buckets - 1 - depth) if layer_decay < 1.0 else 1.0
        groups.append(
            {
                "params": [param],
                "lr": base_lr * scale,
                # `param.ndim <= 1` catches norm scales, norm shifts and biases
                # without having to guess at their names.
                "weight_decay": 0.0 if param.ndim <= 1 else weight_decay,
                "name": name,
            }
        )
    return groups


def _mix_batch(inputs, targets, *, mixup_alpha: float, cutmix_alpha: float, mix_prob: float, rng):
    """Blend two examples and their labels. Returns `(inputs, y_a, y_b, lam)`.

    With a couple of hundred images per class a model can memorise a class before
    it has generalised it, and v1's confusions were between dishes that genuinely
    look alike — `mixed_veg_curry` against `pav_bhaji`, the whole brown-gravy
    family against itself. Mixing forces the decision boundary to stay roughly
    linear between examples instead of carving a tight pocket around each
    photograph.

    Mixup averages two whole images; CutMix pastes a rectangle of one over the
    other. They fail in opposite ways — mixup can produce something that looks
    like no real food, CutMix can cut away the very region that identified the
    dish — so alternating is steadier than committing to either.

    `lam` comes back for the caller to charge the loss against both labels rather
    than being folded into a soft target here. That keeps the inverse-frequency
    class weights and the label smoothing already configured on the criterion
    applying unchanged, which a hand-built target tensor would silently bypass.
    """
    import numpy as _np
    import torch

    if inputs.size(0) < 2 or rng.random() >= mix_prob:
        return inputs, targets, targets, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or rng.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    lam = float(_np.random.default_rng(rng.randrange(2**32)).beta(alpha, alpha))
    lam = min(max(lam, 0.0), 1.0)
    # Drawn from the injected `rng` rather than torch's global generator. Every
    # other choice in this function already comes from `rng`, and a run whose
    # mixing pattern cannot be reproduced from its seed is a run whose loss curve
    # cannot be compared against another one's — which is the entire point of
    # measuring a recipe change. The permutation can come back as the identity,
    # pairing samples with themselves and making the step a no-op; that is
    # harmless, and at a real batch size it is also vanishingly rare.
    positions = list(range(inputs.size(0)))
    rng.shuffle(positions)
    order = torch.tensor(positions, device=inputs.device)
    shuffled = targets[order]

    if not use_cutmix:
        return inputs.mul(lam).add_(inputs[order], alpha=1.0 - lam), targets, shuffled, lam

    height, width = inputs.shape[-2:]
    cut_h, cut_w = int(height * math.sqrt(1.0 - lam)), int(width * math.sqrt(1.0 - lam))
    if cut_h < 1 or cut_w < 1:
        return inputs, targets, targets, 1.0
    centre_y, centre_x = rng.randrange(height), rng.randrange(width)
    y0, y1 = max(centre_y - cut_h // 2, 0), min(centre_y + cut_h // 2, height)
    x0, x1 = max(centre_x - cut_w // 2, 0), min(centre_x + cut_w // 2, width)
    if y1 <= y0 or x1 <= x0:
        return inputs, targets, targets, 1.0
    mixed = inputs.clone()
    mixed[:, :, y0:y1, x0:x1] = inputs[order][:, :, y0:y1, x0:x1]
    # Recompute `lam` from the rectangle actually pasted: clipping at the edges
    # means the requested area and the delivered area differ, and the loss has to
    # be weighted by what the model can actually see.
    lam = 1.0 - ((y1 - y0) * (x1 - x0) / float(height * width))
    return mixed, targets, shuffled, lam


class _EMA:
    """A slowly-moving average of the weights, evaluated beside the live ones.

    Gradient descent near a minimum does not settle, it orbits; the average of the
    points on that orbit is usually a better model than any single point on it.
    The cost is one extra copy of the weights in memory and one extra evaluation
    per epoch. The gain needs nothing at inference time, because `train()` keeps
    whichever of the two evaluated better and writes only that one to the
    checkpoint.

    Only floating-point tensors are averaged. `num_batches_tracked` is an integer
    counter, and a decayed average of a counter is not a counter.
    """

    def __init__(self, model, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone().float()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in self.shadow:
                    self.shadow[key].mul_(self.decay).add_(
                        value.detach().float(), alpha=1.0 - self.decay
                    )

    def _merged(self, model) -> dict[str, Any]:
        merged = {key: value.detach().clone() for key, value in model.state_dict().items()}
        for key, value in self.shadow.items():
            if key in merged:
                merged[key] = value.to(merged[key].dtype)
        return merged

    @contextlib.contextmanager
    def applied_to(self, model):
        """`with ema.applied_to(model):` evaluates the average, then puts it back.

        The live weights are restored in a `finally`, because leaving the averaged
        weights installed would make the next epoch train from them — a subtle way
        to turn an evaluation aid into a second, unintended optimiser.
        """
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self._merged(model))
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    def state_dict(self, model) -> dict[str, Any]:
        return {key: value.cpu() for key, value in self._merged(model).items()}


def train(
    data_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 24,
    learning_rate: float = 3e-4,
    patience: int = 8,
    version: str | None = None,
    device: str | None = None,
    workers: int = 4,
    init_from: str | None = None,
    freeze_fraction: float = 0.0,
    layer_decay: float = 0.75,
    warmup_epochs: int = 2,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
    ema_decay: float = 0.999,
) -> dict[str, Any]:
    """Fine-tune the backbone and write a checkpoint + TRAINING_LOG entry.

    The defaults are v2's, not v1's, and they differ in ways worth naming because
    v1's own history says why. It froze 70% of the network and its training loss
    flattened 0.155 above the floor label smoothing puts under it while validation
    sat at 0.77 for nine epochs — both curves flat at once, which is underfitting,
    not overfitting. So: nothing is frozen by default, shallow layers get a decayed
    learning rate instead (`_param_groups`), the base rate is 3× higher because a
    full fine-tune with warmup can take it, and mixup/CutMix plus a weight average
    supply the regularisation that freezing was accidentally providing.
    """
    import torch
    import torch.nn as nn

    samples, class_names = _discover_dataset(data_dir)
    train_rows, val_rows, test_rows = _stratified_split(samples)
    log.info(
        "Dataset: %d images / %d classes (train %d, val %d, test %d)",
        len(samples), len(class_names), len(train_rows), len(val_rows), len(test_rows),
    )

    dev = _device(device)
    model = _build_backbone(len(class_names), pretrained=True).to(dev)
    warm_start = None
    if init_from:
        warm_start = _warm_start(model, Path(init_from), class_names)
    frozen, trainable = _freeze_early_layers(model, keep_fraction=1.0 - freeze_fraction)
    log.info("Device %s — froze %d parameter groups, training %d", dev, frozen, trainable)

    counts = np.bincount([label for _, label in train_rows], minlength=len(class_names))
    weights = torch.tensor(
        (counts.sum() / np.maximum(counts, 1)) / len(class_names), dtype=torch.float32
    ).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    groups = _param_groups(
        model, base_lr=learning_rate, weight_decay=weight_decay, layer_decay=layer_decay
    )
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    log.info(
        "AdamW over %d parameter groups — lr %.2e down to %.2e across the depth, weight decay %g",
        len(groups),
        max(group["lr"] for group in groups),
        min(group["lr"] for group in groups),
        weight_decay,
    )

    # Warmup then cosine, as one schedule over the *whole* run rather than a
    # per-epoch step. A full fine-tune at 3e-4 with a freshly-initialised head
    # will wreck pretrained features in its first few hundred steps if the rate
    # starts at full value; ramping from near zero is what makes the higher rate
    # safe to use at all. `SequentialLR` needs the switch point in epochs because
    # both children are stepped once per epoch.
    warmup = max(0, min(warmup_epochs, max(epochs - 1, 0)))
    if warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1)),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Half precision roughly doubles throughput on a T4 and halves activation
    # memory, which is what lets batch 24 at 300px fit at all on a 16 GB card.
    autocast = _autocast(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ema = _EMA(model, ema_decay) if ema_decay > 0 else None

    best_accuracy = 0.0
    best_state: dict[str, Any] | None = None
    best_source = "live"
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(epoch).shuffle(train_rows)
        mixer = random.Random(10_000 + epoch)
        running_loss = 0.0
        seen = 0
        for inputs, targets in _batches(
            train_rows, batch_size, augment=True, seed=epoch, device=dev, workers=workers
        ):
            inputs, target_a, target_b, lam = _mix_batch(
                inputs,
                targets,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mix_prob=mix_prob,
                rng=mixer,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(inputs)
                # Charging both labels separately, rather than building a blended
                # target, is what keeps the class weights and label smoothing on
                # `criterion` in force for a mixed batch.
                loss = criterion(logits, target_a)
                if lam < 1.0:
                    loss = lam * loss + (1.0 - lam) * criterion(logits, target_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            running_loss += float(loss.item()) * inputs.size(0)
            seen += inputs.size(0)
        scheduler.step()

        holdout = val_rows or test_rows
        val_accuracy, _ = _evaluate(model, holdout, class_names, batch_size, dev)
        ema_accuracy = None
        if ema is not None:
            with ema.applied_to(model):
                ema_accuracy, _ = _evaluate(model, holdout, class_names, batch_size, dev)
        train_loss = running_loss / max(seen, 1)
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_top1": round(val_accuracy, 4),
            "lr": round(float(optimizer.param_groups[-1]["lr"]), 8),
        }
        if ema_accuracy is not None:
            entry["ema_val_top1"] = round(ema_accuracy, 4)
        history.append(entry)
        log.info(
            "epoch %02d — loss %.4f — val top-1 %.4f%s",
            epoch,
            train_loss,
            val_accuracy,
            f" — ema {ema_accuracy:.4f}" if ema_accuracy is not None else "",
        )

        # The averaged weights compete with the live ones for the checkpoint on
        # equal terms: whichever reads higher on the holdout is what gets kept, so
        # EMA can never make the result worse than not having used it.
        candidates = [(val_accuracy, "live")]
        if ema_accuracy is not None:
            candidates.append((ema_accuracy, "ema"))
        epoch_best, source = max(candidates)

        if epoch_best > best_accuracy + 1e-4:
            best_accuracy = epoch_best
            best_source = source
            if source == "ema" and ema is not None:
                best_state = {k: v.clone() for k, v in ema.state_dict(model).items()}
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("Early stopping at epoch %d (no val gain for %d epochs)", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Kept the %s weights — val top-1 %.4f", best_source, best_accuracy)

    test_accuracy, per_class_f1 = _evaluate(model, test_rows or val_rows, class_names, batch_size, dev)

    tag = version or _next_version_tag()
    checkpoint_path = MODEL_DIR / f"efficientnet_{tag}.pt"
    torch.save(
        {
            # CPU tensors on purpose: the checkpoint has to load on a machine with
            # no GPU, which is the normal case for serving.
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "classes": list(class_names),
            "version": tag,
            "input_resolution": INPUT_RESOLUTION,
            "val_top1": round(best_accuracy, 4),
            "test_top1": round(test_accuracy, 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        checkpoint_path,
    )

    summary = {
        "version": tag,
        "checkpoint": str(checkpoint_path),
        "classes": len(class_names),
        "images": len(samples),
        "device": str(dev),
        "epochs_run": len(history),
        "val_top1": round(best_accuracy, 4),
        "test_top1": round(test_accuracy, 4),
        "per_class_f1": per_class_f1,
        "history": history,
        # The recipe travels with the number. v1's summary recorded the accuracy
        # but not the fact that 70% of the network was frozen, so the one setting
        # that explained the result was the one nobody could read back afterwards.
        "recipe": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "freeze_fraction": freeze_fraction,
            "layer_decay": layer_decay,
            "warmup_epochs": warmup,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
            "mix_prob": mix_prob,
            "ema_decay": ema_decay,
            "input_resolution": INPUT_RESOLUTION,
            "weights_kept": best_source,
        },
        "warm_start": warm_start,
        "class_names": list(class_names),
    }
    _append_training_log(summary)
    log.info("Saved %s — test top-1 %.4f", checkpoint_path, test_accuracy)
    return summary


def _next_version_tag() -> str:
    existing = sorted(MODEL_DIR.glob("efficientnet_v*.pt"))
    numbers = []
    for path in existing:
        digits = "".join(ch for ch in path.stem.split("_")[-1] if ch.isdigit())
        if digits:
            numbers.append(int(digits))
    return f"v{max(numbers) + 1 if numbers else 1}"


def _append_training_log(summary: dict[str, Any]) -> None:
    """design.md §9 / §14.4 — a plain markdown log instead of MLflow."""
    path = PROJECT_DIR / "TRAINING_LOG.md"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    worst = sorted(summary["per_class_f1"].items(), key=lambda row: row[1])[:5]
    entry = [
        f"\n## {summary['version']} — {stamp}\n",
        f"- Checkpoint: `{Path(summary['checkpoint']).name}`",
        f"- Dataset: {summary['images']} images across {summary['classes']} classes",
        f"- Epochs run: {summary['epochs_run']}",
        f"- **Val top-1: {summary['val_top1']:.2%}**",
        f"- **Test top-1: {summary['test_top1']:.2%}** (target ≥85%, design.md §17)",
        f"- Weakest per-class F1: {', '.join(f'{name} {score:.2f}' for name, score in worst) or 'n/a'}",
        "",
        "<details><summary>Epoch history</summary>\n",
        "```json",
        json.dumps(summary["history"], indent=2),
        "```",
        "</details>",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))


classifier = DishClassifier()


# One definition, two front doors. `classify.py --train` and
# `tools/train_kaggle.py` both build their parser from this table and both read
# their kwargs back out of it, so a flag cannot come to mean one thing in one and
# something else in the other. `test_the_two_entry_points_speak_the_same_language`
# checks the vocabularies match; sharing the table is what makes them match by
# construction instead of by remembering to edit two files.
RECIPE_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--init-from",
        {
            "default": None,
            "metavar": "CHECKPOINT",
            "help": "Start from an existing .pt instead of ImageNet weights. Head rows are "
                    "matched by class name, so a widened class list keeps what it can.",
        },
    ),
    (
        "--freeze-fraction",
        {
            "type": float,
            "default": 0.0,
            "help": "Fraction of parameter groups held at zero learning rate. v1 used 0.70 "
                    "and underfit; prefer --layer-decay to this. (default: 0.0)",
        },
    ),
    (
        "--layer-decay",
        {
            "type": float,
            "default": 0.75,
            "help": "Per-depth learning-rate multiplier; shallow layers train slower. "
                    "1.0 trains every layer at one rate. (default: 0.75)",
        },
    ),
    (
        "--warmup-epochs",
        {
            "type": int,
            "default": 2,
            "help": "Ramp the learning rate from 5%% over N epochs before the cosine decay. "
                    "(default: 2)",
        },
    ),
    (
        "--weight-decay",
        {"type": float, "default": 0.01, "help": "AdamW decay, skipped on norms and biases."},
    ),
    (
        "--label-smoothing",
        {"type": float, "default": 0.1, "help": "Cross-entropy label smoothing. (default: 0.1)"},
    ),
    (
        "--mixup-alpha",
        {"type": float, "default": 0.2, "help": "Beta parameter for mixup; 0 disables it."},
    ),
    (
        "--cutmix-alpha",
        {"type": float, "default": 1.0, "help": "Beta parameter for CutMix; 0 disables it."},
    ),
    (
        "--mix-prob",
        {
            "type": float,
            "default": 0.5,
            "help": "Probability a batch gets mixed at all. (default: 0.5)",
        },
    ),
    (
        "--ema-decay",
        {
            "type": float,
            "default": 0.999,
            "help": "Decay for the averaged weights, which compete with the live ones for "
                    "the checkpoint. 0 disables. (default: 0.999)",
        },
    ),
)


def _recipe_destination(flag: str) -> str:
    return flag.removeprefix("--").replace("-", "_")


def add_recipe_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the recipe flags to either entry point's parser."""
    for flag, options in RECIPE_ARGUMENTS:
        parser.add_argument(flag, **options)


def recipe_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the recipe flags back into `train()` keyword arguments."""
    return {
        _recipe_destination(flag): getattr(args, _recipe_destination(flag))
        for flag, _ in RECIPE_ARGUMENTS
    }


def _main(argv: list[str]) -> int:
    """Train from an already-assembled ImageFolder tree.

    `tools/train_kaggle.py` is the front door for a Kaggle run — it builds that
    tree out of raw dataset folders first. This entry point is for when the tree
    already exists and only the training knobs matter. Every flag `train()`
    accepts is exposed here, and the names match `train_kaggle.py`'s, so a
    command that worked there works here: silently dropping `--device` would
    have meant a GPU run quietly falling back to CPU.
    """
    parser = argparse.ArgumentParser(description="EfficientNet-B3 dish classifier")
    parser.add_argument("--train", action="store_true", help="run fine-tuning")
    parser.add_argument("--data", default=str(PROJECT_DIR / "data" / "processed"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Stop after N epochs without a better val top-1."
    )
    parser.add_argument("--workers", type=int, default=4, help="Image-decoding threads.")
    parser.add_argument("--device", default=None, help="Force a device, e.g. cuda or cpu.")
    parser.add_argument("--version", default=None, help="Checkpoint tag, e.g. v1.")
    add_recipe_arguments(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.train:
        parser.print_help()
        return 0

    summary = train(
        Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        version=args.version,
        device=args.device,
        workers=args.workers,
        **recipe_from_args(args),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
