"""Stage 3 — Depth estimation and volume/weight geometry (design.md §7.3).

Two independent concerns live here, deliberately separated so the second one is
testable without a neural network:

* **Depth source** — MiDaS v3 (pretrained, inference only) when torch can load
  it; otherwise a shape-from-mask elevation proxy (distance transform blended
  with shading). Either way the output is *relative* depth.
* **Geometry** — pure functions that turn a relative elevation field plus the
  plate-diameter assumption into millilitres and then grams.

### The scale assumption, stated openly

Monocular depth is scale-ambiguous, so this module never pretends to recover
absolute depth. It recovers each of the three scales it needs from a different,
explicitly-named source:

1. **Lateral scale (px → cm)** from the detected plate diameter (design.md §5.3).
2. **Elevation *shape*** from the depth map — where the food is piled high
   versus thin at the edges.
3. **Elevation *magnitude*** from a per-category serving-height prior, gently
   scaled by the item's footprint.

This is the same class of simplifying assumption used by ECUSTFD- and
Nutrition5k-style geometric pipelines, and it's why design.md §17 targets a
mean error band rather than exact grams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from config import settings
from detection import Detection, PlateEstimate
from imaging import PreparedImage

log = logging.getLogger("nutriai.depth")

# Typical *mean* served depth in cm over the item's visible footprint, i.e.
# (serving volume ÷ footprint area) for a nominal portion. Mean depth rather
# than peak depth is deliberate: it is the quantity portion tables actually
# pin down, and it makes the estimate independent of the elevation field's
# shape — see `integrate_volume`. Bowl-served items are deep relative to the
# area they show; flatbreads are a few millimetres.
SERVING_DEPTH_CM: dict[str, float] = {
    "dal": 3.3,
    "curry": 3.1,
    "rice": 3.1,
    "dairy": 2.5,
    "dessert": 2.6,
    "grain": 2.7,
    "salad": 2.6,
    "dry_sabzi": 2.4,
    "protein": 2.3,
    "fried": 2.2,
    "fruit": 2.6,
    "steamed": 1.7,
    "condiment": 1.5,
    "fast_food": 1.6,
    "bread": 0.6,
    "unknown": 2.4,
}

# Footprint at which `SERVING_DEPTH_CM` is exactly right.
NOMINAL_AREA_CM2 = 60.0

# Portion volume grows a little faster than footprint area (a bigger helping is
# both wider *and* slightly deeper), so mean depth rises weakly with area.
# Exponent ≈0.2 keeps that effect gentle; a square-root law would let a large
# footprint inflate the weight quadratically.
AREA_DEPTH_EXPONENT = 0.22

# Plausible served-weight envelope per category, in grams. Anything outside is
# clamped and reported with `weight_estimated=True` so the UI can say so.
WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "curry": (45.0, 340.0),
    "dal": (50.0, 360.0),
    "rice": (50.0, 420.0),
    "grain": (40.0, 380.0),
    "bread": (18.0, 220.0),
    "dry_sabzi": (35.0, 300.0),
    "salad": (20.0, 240.0),
    "dairy": (30.0, 300.0),
    "dessert": (25.0, 260.0),
    "fried": (20.0, 260.0),
    "protein": (30.0, 330.0),
    "fruit": (40.0, 400.0),
    "condiment": (8.0, 120.0),
    "fast_food": (40.0, 400.0),
    "steamed": (30.0, 300.0),
    "unknown": (25.0, 350.0),
}

MIN_DEPTH_CM = 0.2
MAX_DEPTH_CM = 6.0


@dataclass
class VolumeEstimate:
    volume_ml: float
    weight_g: float
    area_cm2: float
    mean_height_cm: float
    peak_height_cm: float
    density_g_per_ml: float
    method: str
    clamped: bool


# --------------------------------------------------------------------------
# Depth source
# --------------------------------------------------------------------------

class DepthEstimator:
    """Stage-3 singleton, loaded once at startup (design.md §12.1)."""

    def __init__(self) -> None:
        self.backend = "unloaded"
        self.version = "n/a"
        self._model = None
        self._transform = None

    def load(self) -> None:
        if settings.enable_torch_models:
            try:
                import torch

                model = torch.hub.load("intel-isl/MiDaS", settings.midas_model, trust_repo=True)
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                model.eval()
                self._model = model
                self._transform = (
                    transforms.small_transform
                    if "small" in settings.midas_model.lower()
                    else transforms.dpt_transform
                )
                self.backend = "midas"
                self.version = settings.midas_model
                log.info("MiDaS depth model loaded (%s)", settings.midas_model)
                return
            except Exception as exc:
                log.warning("MiDaS unavailable (%s) — using shape-from-mask elevation", exc)
        self.backend = "shape-prior"
        self.version = "distance-transform-v1"

    @property
    def ready(self) -> bool:
        return self.backend != "unloaded"

    def relative_depth(self, prepared: PreparedImage) -> np.ndarray | None:
        """Relative depth, higher = closer to the camera. None if unavailable."""
        if self._model is None or self._transform is None:
            return None
        try:
            import torch

            sample = self._transform(np.asarray(prepared.image)).to("cpu")
            with torch.no_grad():
                prediction = self._model(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(prepared.height, prepared.width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy().astype(np.float32)
            span = float(depth.max() - depth.min())
            if span <= 1e-6:
                return None
            return (depth - depth.min()) / span
        except Exception as exc:
            log.warning("MiDaS inference failed (%s) — degrading to shape prior", exc)
            return None


# --------------------------------------------------------------------------
# Geometry — pure functions, unit-tested
# --------------------------------------------------------------------------

def normalized_elevation(mask: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    """Elevation field in [0, 1] over `mask`: 0 at the rim, 1 at the peak.

    With a depth map, the plate-adjacent rim of the blob anchors "zero height"
    and relative depth supplies the shape. Without one, the Euclidean distance
    transform of the mask gives the classic food-mound dome.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)

    dome = ndimage.distance_transform_edt(mask).astype(np.float32)
    peak = float(dome.max())
    dome = dome / peak if peak > 0 else dome

    if depth is None:
        field = dome
    else:
        inside = depth[mask]
        rim_reference = float(np.percentile(inside, 12.0))
        span = float(np.percentile(inside, 98.0) - rim_reference)
        if span <= 1e-4:
            field = dome
        else:
            relative = np.zeros_like(dome)
            relative[mask] = np.clip((depth[mask] - rim_reference) / span, 0.0, 1.0)
            # Depth carries the shape; the dome prior keeps thin edges from
            # collapsing when MiDaS is noisy on textured food.
            field = 0.68 * relative + 0.32 * dome

    smoothed = ndimage.gaussian_filter(field * mask, sigma=1.6)
    smoothed = np.where(mask, np.clip(smoothed, 0.0, 1.0), 0.0)
    return smoothed.astype(np.float32)


def mean_depth_cm(category: str, area_cm2: float) -> float:
    """Mean served depth prior for a category, adjusted for footprint size."""
    base = SERVING_DEPTH_CM.get(category, SERVING_DEPTH_CM["unknown"])
    ratio = max(area_cm2, 1.0) / NOMINAL_AREA_CM2
    scale = float(np.clip(ratio**AREA_DEPTH_EXPONENT, 0.65, 1.6))
    return float(np.clip(base * scale, MIN_DEPTH_CM, MAX_DEPTH_CM))


def integrate_volume(elevation: np.ndarray, mean_cm: float, px_area_cm2: float) -> float:
    """Riemann sum of the elevation field: Σ h(px) · pixel_area → millilitres.

    `elevation` carries only the *shape* of the mound, so it is rescaled to have
    unit mean before summing. That decouples the volume from how peaked the
    shape happens to be: a dome and a flat slab of the same footprint and the
    same mean depth hold the same volume, which is the physically correct
    result. 1 cm³ == 1 ml, so no unit conversion follows the sum.
    """
    if mean_cm <= 0 or px_area_cm2 <= 0:
        return 0.0
    total = float(elevation.sum())
    if total <= 0:
        return 0.0
    area_px = float(np.count_nonzero(elevation))
    if area_px <= 0:
        return 0.0
    heights = elevation * (mean_cm * area_px / total)
    return float(heights.sum() * px_area_cm2)


def weight_from_volume(volume_ml: float, density_g_per_ml: float) -> float:
    """grams = millilitres × density (design.md §7.3)."""
    if volume_ml < 0:
        raise ValueError("volume_ml must be non-negative")
    if density_g_per_ml <= 0:
        raise ValueError("density must be positive")
    return volume_ml * density_g_per_ml


def clamp_weight(weight_g: float, category: str) -> tuple[float, bool]:
    low, high = WEIGHT_BOUNDS.get(category, WEIGHT_BOUNDS["unknown"])
    clamped = float(np.clip(weight_g, low, high))
    return clamped, not np.isclose(clamped, weight_g, rtol=1e-3, atol=0.5)


def mask_for(detection: Detection, food_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resolve a pixel mask for a detection.

    The segmenter supplies one directly. A YOLO box doesn't, so intersect the
    global food mask with the box, and fall back to an inscribed ellipse if that
    intersection is empty.
    """
    if detection.mask is not None and detection.mask.any():
        return detection.mask

    height, width = shape
    x, y, w, h = detection.bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    box = np.zeros((height, width), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return box
    box[y0:y1, x0:x1] = True

    intersection = box & food_mask
    if intersection.sum() > 0.12 * box.sum():
        return intersection

    yy, xx = np.ogrid[y0:y1, x0:x1]
    cy, cx = (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0
    ry, rx = max(1.0, (y1 - y0) / 2.0), max(1.0, (x1 - x0) / 2.0)
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    box[:] = False
    box[y0:y1, x0:x1] = ellipse
    return box


def estimate_volume(
    detection: Detection,
    *,
    category: str,
    density_g_per_ml: float,
    plate: PlateEstimate,
    depth: np.ndarray | None,
    food_mask: np.ndarray,
    shape: tuple[int, int],
) -> VolumeEstimate:
    """Full stage-3 estimate for one detected item."""
    mask = mask_for(detection, food_mask, shape)
    area_px = int(mask.sum())
    px_area = plate.px_area_cm2
    area_cm2 = area_px * px_area

    if area_px == 0 or area_cm2 <= 0.5:
        # Degenerate region: fall back to the category's lower bound so the item
        # still yields a number, flagged as estimated (design.md §12.2).
        low, _ = WEIGHT_BOUNDS.get(category, WEIGHT_BOUNDS["unknown"])
        return VolumeEstimate(
            volume_ml=round(low / max(density_g_per_ml, 0.1), 1),
            weight_g=round(low, 1),
            area_cm2=round(max(area_cm2, 0.0), 2),
            mean_height_cm=0.0,
            peak_height_cm=0.0,
            density_g_per_ml=density_g_per_ml,
            method="fallback-portion",
            clamped=True,
        )

    elevation = normalized_elevation(mask, depth)
    mean_cm = mean_depth_cm(category, area_cm2)
    volume_ml = integrate_volume(elevation, mean_cm, px_area)
    raw_weight = weight_from_volume(volume_ml, density_g_per_ml)
    weight_g, clamped = clamp_weight(raw_weight, category)

    if clamped and volume_ml > 0:
        volume_ml = weight_g / density_g_per_ml

    mean_height = volume_ml / area_cm2 if area_cm2 > 0 else 0.0
    shape_peak = float(elevation.max()) * mean_height / max(float(elevation.mean()), 1e-6)
    method = "midas+geometry" if depth is not None else "shape-prior+geometry"

    return VolumeEstimate(
        volume_ml=round(volume_ml, 1),
        weight_g=round(weight_g, 1),
        area_cm2=round(area_cm2, 2),
        mean_height_cm=round(mean_height, 2),
        peak_height_cm=round(min(shape_peak, MAX_DEPTH_CM * 2.5), 2),
        density_g_per_ml=density_g_per_ml,
        method=method,
        clamped=clamped,
    )


depth_estimator = DepthEstimator()
