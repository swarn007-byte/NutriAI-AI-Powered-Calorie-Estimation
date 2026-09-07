"""Stage 2 — Food detection (design.md §7.2).

Primary path: pretrained YOLOv8 via Ultralytics — no training, per design.md §6.2.

Secondary path: a plate-aware classical segmenter built on numpy/scipy. It
exists because YOLOv8 weights are a ~6 MB network download that isn't always
available (offline machine, locked-down CI, first-run demo), and design.md
§12.2 asks explicitly for graceful degradation rather than a dead pipeline.
The segmenter also produces per-item pixel masks, which plain YOLO boxes don't,
and stage 3 uses those masks directly for volume integration.

Both paths return the same `Detection` shape, so `pipeline.py` never branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from config import settings
from imaging import PreparedImage, chroma

log = logging.getLogger("nutriai.detection")

# COCO classes that can plausibly be food on a plate. Everything else YOLO
# reports (person, chair, tv...) is discarded.
_FOOD_COCO_CLASSES = {
    "banana", "apple", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "sandwich", "bowl", "cup", "wine glass",
}


@dataclass
class Detection:
    """One localized food region."""

    bbox: tuple[int, int, int, int]  # x, y, w, h in working-image pixels
    label: str  # coarse label
    confidence: float
    mask: np.ndarray | None = None  # bool HxW over the working image
    area_px: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def cy(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0


@dataclass
class PlateEstimate:
    """The reference frame that converts pixels into centimetres (§5.3)."""

    center: tuple[float, float]
    radius_px: float
    mask: np.ndarray
    detected: bool
    diameter_cm: float

    @property
    def cm_per_px(self) -> float:
        return self.diameter_cm / max(1.0, self.radius_px * 2.0)

    @property
    def px_area_cm2(self) -> float:
        return self.cm_per_px**2


# --------------------------------------------------------------------------
# Plate localisation
# --------------------------------------------------------------------------

def estimate_plate(prepared: PreparedImage, diameter_cm: float | None = None) -> PlateEstimate:
    """Find the plate disc: bright, low-chroma, near-centre, holes filled.

    Filling holes is the trick that matters — food sitting on the plate punches
    holes in the bright-region mask, so `binary_fill_holes` recovers the whole
    disc, which is what the pixel→cm scale must be measured from.
    """
    diameter = float(diameter_cm or settings.default_plate_diameter_cm)
    lab = prepared.lab
    height, width = prepared.height, prepared.width
    lightness = lab[..., 0]
    sat = chroma(lab)

    bright = lightness > np.percentile(lightness, 52.0)
    pale = sat < max(14.0, float(np.percentile(sat, 45.0)))
    candidate = bright & pale

    candidate = ndimage.binary_opening(candidate, np.ones((5, 5), bool))
    filled = ndimage.binary_fill_holes(candidate)
    labels, count = ndimage.label(filled)

    best_mask: np.ndarray | None = None
    if count:
        cy, cx = height / 2.0, width / 2.0
        best_score = -1.0
        for index in range(1, count + 1):
            region = labels == index
            area = int(region.sum())
            if area < 0.06 * height * width:
                continue
            ry, rx = ndimage.center_of_mass(region)
            centre_penalty = ((ry - cy) / height) ** 2 + ((rx - cx) / width) ** 2
            score = (area / (height * width)) - 1.4 * centre_penalty
            if score > best_score:
                best_score = score
                best_mask = region

    def fallback_plate() -> PlateEstimate:
        # No plate found (dark plate, banana leaf, close-up). Assume the plate
        # fills most of the frame — stated openly as an assumption (§5.3).
        radius = 0.46 * min(height, width)
        yy, xx = np.ogrid[:height, :width]
        mask = ((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2) <= radius**2
        return PlateEstimate((width / 2.0, height / 2.0), radius, mask, False, diameter)

    if best_mask is None:
        return fallback_plate()

    # Food and serving bowls can split a white plate into several bright
    # components. When the winning component spans almost the whole frame but
    # contains less than half the pixels, its centroid/radius describe the
    # surviving bright fragments rather than the physical plate. That bad
    # scale then makes every food box drift toward an empty plate region. Use
    # the documented centered-plate prior for this recognizable failure mode.
    ys, xs = np.nonzero(best_mask)
    span_x = (xs.max() - xs.min() + 1) / width if xs.size else 0.0
    span_y = (ys.max() - ys.min() + 1) / height if ys.size else 0.0
    if best_mask.mean() < 0.42 and span_x > 0.82 and span_y > 0.82:
        log.info("Plate mask fragmented across the frame — using centered plate prior")
        return fallback_plate()

    area = float(best_mask.sum())
    radius = float(np.sqrt(area / np.pi))
    ry, rx = ndimage.center_of_mass(best_mask)
    return PlateEstimate((float(rx), float(ry)), radius, best_mask, True, diameter)


# --------------------------------------------------------------------------
# Coarse colour-based labelling
# --------------------------------------------------------------------------

def coarse_label(mean_lab: np.ndarray, texture: float) -> str:
    lightness, a_star, b_star = (float(v) for v in mean_lab)
    sat = float(np.hypot(a_star, b_star))
    hue = float(np.degrees(np.arctan2(b_star, a_star)) % 360.0)

    if sat < 11.0:
        return "rice_or_bread" if lightness > 62.0 else "dark_side"
    if 12.0 <= hue < 55.0:
        return "gravy" if sat > 26.0 else "brown_dish"
    if 55.0 <= hue < 88.0:
        return "dal_or_yellow"
    if 88.0 <= hue < 165.0:
        return "vegetable_green"
    if hue >= 300.0 or hue < 12.0:
        return "red_dish"
    if lightness > 70.0 and texture < 2.0:
        return "rice_or_bread"
    return "mixed_dish"


# --------------------------------------------------------------------------
# Classical segmenter
# --------------------------------------------------------------------------

def local_texture(lightness: np.ndarray, radius: int = 4) -> np.ndarray:
    """Local standard deviation of lightness — a smoothness/roughness field.

    This is what separates pale food from the pale plate underneath it. Rice on
    a white plate is nearly invisible to a colour-difference test (both are
    bright and unsaturated) but it is visibly *lumpy*, while glazed ceramic is
    smooth. Computed as sqrt(E[L²] − E[L]²) over a uniform window.
    """
    size = radius * 2 + 1
    mean = ndimage.uniform_filter(lightness, size=size)
    mean_square = ndimage.uniform_filter(lightness * lightness, size=size)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def _flatten_illumination(field: np.ndarray, plate: PlateEstimate) -> np.ndarray:
    """Remove the slow lighting gradient across the plate, keeping local detail.

    Absolute lightness cannot separate pale food from pale ceramic: a camera's
    vignette and an off-centre light both swing plate lightness by more than a
    mound of rice does, so an absolute test flags the whole plate. A gradient is
    low-frequency and food is not, so subtracting a heavily-blurred copy leaves
    the food and cancels the lighting. Sigma is tied to the plate radius, which
    keeps the behaviour identical whether the photo is 640 px or 4000 px wide.
    """
    sigma = max(6.0, plate.radius_px * 0.32)
    smooth = ndimage.gaussian_filter(field, sigma=sigma, mode="nearest")
    return field - smooth


def _robust_scale(values: np.ndarray, floor: float) -> float:
    """Median absolute deviation, rescaled to a standard-deviation equivalent."""
    if values.size == 0:
        return floor
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, mad * 1.4826)


def plate_deviation(prepared: PreparedImage, plate: PlateEstimate) -> np.ndarray:
    """Score how unlike bare plate each pixel looks. Higher means more food-like.

    The bare plate is modelled from the rim ring — median and a MAD-based scale
    per channel — and every pixel is scored against that model. Scoring against
    a background model rather than splitting a histogram matters because a
    served plate holds at least three populations (ceramic, pale food,
    saturated food) and any two-class split puts its boundary between the two
    *food* classes, discarding rice and idli as background.

    Each channel is one-sided, following what food physically does to a plate:
    it adds colour, adds roughness, and casts shadow. Bare glazed ceramic is the
    brightest, smoothest, least saturated surface in the frame, so *brighter
    than the plate* is not evidence of food. The score is the largest of the
    three — one channel deviating strongly is enough.

    The MAD scales are floored at per-channel sensor-noise estimates, so the
    result is a normalised score rather than a true z-statistic; the decision
    boundary that goes with it is `FOOD_DEVIATION_Z`.
    """
    lab = prepared.lab
    lightness = lab[..., 0]
    sat = chroma(lab)
    texture = local_texture(lightness)
    relief = _flatten_illumination(lightness, plate)

    core = ndimage.binary_erosion(plate.mask, np.ones((9, 9), bool), iterations=3)
    ring = plate.mask & ~core
    if ring.sum() < 200:
        ring = plate.mask

    median_relief, scale_relief = _median_scale(relief[ring], 0.9)
    median_chroma, scale_chroma = _median_scale(sat[ring], 1.5)
    median_texture, scale_texture = _median_scale(texture[ring], 0.5)

    z_relief = np.abs(relief - median_relief) / scale_relief
    z_chroma = np.maximum(sat - median_chroma, 0.0) / scale_chroma
    z_texture = np.maximum(texture - median_texture, 0.0) / scale_texture
    return np.maximum(np.maximum(z_relief, z_chroma), z_texture)


def _median_scale(sample: np.ndarray, floor: float) -> tuple[float, float]:
    return float(np.median(sample)) if sample.size else 0.0, _robust_scale(sample, floor)


# Decision boundary on the `plate_deviation` score, calibrated against the
# synthetic samples in tools/check_segmentation.py (whose true food areas are
# known exactly). Adaptive rules — Otsu, a multiple of the median — were tried
# and all of them collapse on the hardest real case, pale food on a pale plate:
# they put the split between the two *food* populations and throw the rice away.
# A fixed boundary on a normalised score holds across all four samples.
FOOD_DEVIATION_Z = 6.5


def food_mask(
    prepared: PreparedImage,
    plate: PlateEstimate,
    *,
    z_threshold: float = FOOD_DEVIATION_Z,
) -> np.ndarray:
    """Pixels inside the plate that don't look like bare plate surface."""
    deviation = plate_deviation(prepared, plate)

    # A patterned rim or a hard shadow across the plate can push the whole
    # surface past the boundary; tighten until the food fraction is plausible
    # for a served plate rather than trusting a runaway mask.
    plate_area = max(1.0, float(plate.mask.sum()))
    for relaxed in (z_threshold, z_threshold * 1.6, z_threshold * 2.6, z_threshold * 4.0):
        mask = plate.mask & (deviation > relaxed)
        if mask.sum() / plate_area <= 0.88:
            break

    mask = ndimage.binary_opening(mask, np.ones((3, 3), bool))
    mask = ndimage.binary_closing(mask, np.ones((7, 7), bool))
    mask = ndimage.binary_fill_holes(mask)

    if mask.sum() < 0.02 * plate.mask.sum() and _holds_any_structure(deviation, plate):
        # Very low-contrast plate: fall back to the inner plate region.
        mask = ndimage.binary_erosion(plate.mask, np.ones((11, 11), bool), iterations=2)
    return mask


def _holds_any_structure(deviation: np.ndarray, plate: PlateEstimate) -> bool:
    """Whether the plate holds *anything* — the guard on the low-contrast fallback.

    Without it, a uniform frame (blank upload, lens cap, photo of a clean plate)
    takes the fallback and is served back as a full plate of food, inventing a
    ~700 kcal meal out of nothing. The margin is enormous: a truly uniform frame
    scores 0.00 here at every percentile, while the hardest genuine case — pale
    food on a pale plate, ΔL ≈ 1.5 — reaches 86. So this rejects blanks without
    costing the low-contrast case anything it previously had.
    """
    inside = deviation[plate.mask]
    return inside.size > 0 and float(np.percentile(inside, 99.9)) > 1.0


def _cluster_food(prepared: PreparedImage, mask: np.ndarray, max_clusters: int) -> np.ndarray:
    """Spatially-aware colour clustering; returns an int label map (0 = background)."""
    coords = np.argwhere(mask)
    if coords.shape[0] < 64:
        return mask.astype(np.int32)

    lab = prepared.lab[mask]
    height, width = prepared.height, prepared.width
    ys = coords[:, 0] / height
    xs = coords[:, 1] / width
    features = np.column_stack(
        [
            lab[:, 0] / 24.0,
            lab[:, 1] / 12.0,
            lab[:, 2] / 12.0,
            ys * 3.4,
            xs * 3.4,
        ]
    ).astype(np.float32)

    # Subsample for speed; assign every pixel afterwards.
    limit = 12000
    if features.shape[0] > limit:
        step = features.shape[0] // limit + 1
        sample = features[::step]
    else:
        sample = features

    k = int(np.clip(round(np.sqrt(coords.shape[0]) / 26.0), 2, max_clusters))
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, n_init=4, random_state=7).fit(sample)
        assignments = kmeans.predict(features)
    except Exception as exc:  # pragma: no cover - sklearn is present in the stack
        log.warning("KMeans unavailable (%s); using single-region segmentation", exc)
        return mask.astype(np.int32)

    label_map = np.zeros((height, width), dtype=np.int32)
    label_map[coords[:, 0], coords[:, 1]] = assignments + 1
    return label_map


def _fragmented_plate_regions(prepared: PreparedImage, plate: PlateEstimate) -> list[Detection]:
    """Recover compact food blobs when plate localisation used its prior.

    The generic deviation mask treats plate shadows as food on photos with a
    fragmented bright-plate mask. Colour-connected regions are safer there:
    saturated blobs capture dosa/curry, while a compact high-texture pale blob
    captures chutney without accepting the smooth plate around it.
    """
    lab = prepared.lab
    lightness = lab[..., 0]
    saturation = chroma(lab)
    texture = local_texture(lightness)
    plate_area = float(max(1.0, plate.mask.sum()))
    min_area = max(220.0, 0.012 * plate_area)
    candidates = [
        (plate.mask & (saturation > 25.0), 0.0),
        (plate.mask & (texture > 7.0) & (lightness > 75.0) & (saturation < 30.0), 0.14),
    ]

    regions: list[Detection] = []
    for candidate, max_fraction in candidates:
        candidate = ndimage.binary_opening(candidate, np.ones((3, 3), bool))
        if max_fraction:
            candidate = ndimage.binary_closing(candidate, np.ones((5, 5), bool))
            candidate = ndimage.binary_fill_holes(candidate)
        labels, count = ndimage.label(candidate)
        for index in range(1, count + 1):
            region = labels == index
            area = int(region.sum())
            if area < min_area or (max_fraction and area > max_fraction * plate_area):
                continue
            ys, xs = np.nonzero(region)
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
            fill_ratio = area / float(max(1, w * h))
            if max_fraction and fill_ratio < 0.20:
                continue
            mean_lab = lab[region].mean(axis=0)
            region_kind = "pale_textured" if max_fraction else "saturated"
            if not max_fraction:
                aspect = w / float(max(1, h))
                region_kind = "saturated_round" if 0.72 <= aspect <= 1.35 and area / plate_area < 0.20 else "saturated"
            confidence = float(
                np.clip(0.38 + 0.38 * fill_ratio + 0.8 * (area / plate_area), 0.3, 0.94)
            )
            regions.append(
                Detection(
                    bbox=(x, y, w, h),
                    label=coarse_label(mean_lab, float(np.std(lightness[region]))),
                    confidence=round(confidence, 4),
                    mask=region,
                    area_px=area,
                    meta={"source": "fragmented-plate-segmenter", "region_kind": region_kind},
                )
            )

    regions = _merge_overlaps(regions, threshold=0.35)
    regions.sort(key=lambda item: item.area_px, reverse=True)
    return regions


class FoodDetector:
    """Stage-2 singleton. Loaded once at startup (design.md §12.1)."""

    def __init__(self) -> None:
        self.backend = "unloaded"
        self.version = "n/a"
        self._model = None

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if settings.enable_torch_models:
            try:
                from ultralytics import YOLO

                self._model = YOLO(settings.yolo_weights)
                self.backend = "yolov8"
                self.version = str(settings.yolo_weights)
                log.info("YOLOv8 detector loaded from %s", settings.yolo_weights)
                return
            except Exception as exc:
                log.warning("YOLOv8 unavailable (%s) — using plate-aware segmenter", exc)
        self.backend = "segmenter"
        self.version = "plate-segmenter-v1"

    @property
    def ready(self) -> bool:
        return self.backend != "unloaded"

    # -- inference -------------------------------------------------------
    def detect(self, prepared: PreparedImage, plate: PlateEstimate) -> list[Detection]:
        if self._model is not None:
            detections = self._detect_yolo(prepared)
            if detections:
                return detections
            log.info("YOLOv8 found no food classes — falling back to segmenter for this image")
        return self._detect_segments(prepared, plate)

    def _detect_yolo(self, prepared: PreparedImage) -> list[Detection]:
        assert self._model is not None
        try:
            results = self._model.predict(
                np.asarray(prepared.image), verbose=False, conf=0.25, imgsz=640
            )
        except Exception as exc:
            log.warning("YOLOv8 inference failed (%s)", exc)
            return []

        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            for box in getattr(result, "boxes", []) or []:
                name = str(names.get(int(box.cls[0]), "")).lower()
                if name not in _FOOD_COCO_CLASSES:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                w, h = int(round(x2 - x1)), int(round(y2 - y1))
                if w < 12 or h < 12:
                    continue
                detections.append(
                    Detection(
                        bbox=(int(round(x1)), int(round(y1)), w, h),
                        label=name,
                        confidence=float(box.conf[0]),
                        area_px=w * h,
                        meta={"source": "yolov8"},
                    )
                )
        detections.sort(key=lambda d: d.area_px, reverse=True)
        return detections[: settings.max_items_per_plate]

    def _detect_segments(self, prepared: PreparedImage, plate: PlateEstimate) -> list[Detection]:
        if not plate.detected:
            recovered = _fragmented_plate_regions(prepared, plate)
            if len(recovered) >= 2:
                return recovered[: settings.max_items_per_plate]

        mask = food_mask(prepared, plate)
        label_map = _cluster_food(prepared, mask, max_clusters=settings.max_items_per_plate + 1)

        plate_area = float(max(1.0, plate.mask.sum()))
        min_area = max(220.0, 0.012 * plate_area)

        regions: list[Detection] = []
        for cluster_id in range(1, int(label_map.max()) + 1):
            cluster = label_map == cluster_id
            if not cluster.any():
                continue
            components, count = ndimage.label(cluster)
            for index in range(1, count + 1):
                region = components == index
                area = int(region.sum())
                if area < min_area:
                    continue
                region = ndimage.binary_closing(region, np.ones((5, 5), bool))
                ys, xs = np.nonzero(region)
                x, y = int(xs.min()), int(ys.min())
                w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
                if w < 10 or h < 10:
                    continue

                mean_lab = prepared.lab[region].mean(axis=0)
                texture = float(np.std(prepared.lab[..., 0][region]))
                fill_ratio = area / float(w * h)
                # Confidence blends how solid the blob is with how much of the
                # plate it occupies — compact, plate-filling blobs are real food.
                confidence = float(
                    np.clip(0.34 + 0.44 * fill_ratio + 0.9 * (area / plate_area), 0.3, 0.94)
                )
                regions.append(
                    Detection(
                        bbox=(x, y, w, h),
                        label=coarse_label(mean_lab, texture),
                        confidence=round(confidence, 4),
                        mask=region,
                        area_px=area,
                        meta={
                            "source": "segmenter",
                            "mean_lab": [round(float(v), 2) for v in mean_lab],
                            "texture": round(texture, 3),
                            "fill_ratio": round(fill_ratio, 3),
                        },
                    )
                )

        regions = _merge_adjacent_similar(regions)
        regions = _merge_small_with_nearby(regions)
        regions = _merge_overlaps(regions)
        regions.sort(key=lambda d: d.area_px, reverse=True)
        return regions[: settings.max_items_per_plate]


def _iou(a: Detection, b: Detection) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    return overlap / float(aw * ah + bw * bh - overlap)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    return x, y, int(xs.max() - x + 1), int(ys.max() - y + 1)


def _merge_adjacent_similar(
    regions: list[Detection],
    *,
    colour_tolerance: float = 18.0,
    dilation: int = 7,
) -> list[Detection]:
    """Fuse touching regions of the same colour — one dish, not several.

    KMeans happily cuts a single bowl of curry into four colour clusters, and
    because those clusters sit side by side rather than overlapping, an
    IoU test never rejoins them. Left alone this multiplies the reported
    calories by the number of clusters, which is the single largest error the
    segmenter can make. So regions that *touch* and whose mean Lab colours are
    within `colour_tolerance` (roughly a just-noticeable ΔE) become one item.

    Also merges small regions (< 15% of largest) that are within 2x bounding
    box distance, even if colours differ slightly — a chunk of chicken in gravy
    is one dish, not two.
    """
    if len(regions) < 2:
        return regions

    footprint = np.ones((dilation, dilation), bool)
    working = list(regions)
    merged_any = True
    while merged_any and len(working) > 1:
        merged_any = False
        halos = [
            ndimage.binary_dilation(r.mask, footprint) if r.mask is not None else None
            for r in working
        ]
        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                first, second = working[i], working[j]
                if first.mask is None or second.mask is None:
                    continue
                lab_a = np.asarray(first.meta.get("mean_lab", [0.0, 0.0, 0.0]), dtype=np.float64)
                lab_b = np.asarray(second.meta.get("mean_lab", [0.0, 0.0, 0.0]), dtype=np.float64)
                if float(np.linalg.norm(lab_a - lab_b)) > colour_tolerance:
                    continue
                halo_a, halo_b = halos[i], halos[j]
                if halo_a is None or halo_b is None:
                    continue
                if not (halo_a & second.mask).any() and not (halo_b & first.mask).any():
                    continue

                union = first.mask | second.mask
                total = float(first.area_px + second.area_px) or 1.0
                blended = (lab_a * first.area_px + lab_b * second.area_px) / total
                first.mask = union
                first.area_px = int(union.sum())
                first.bbox = _bbox_of(union)
                first.meta["mean_lab"] = [round(float(v), 2) for v in blended]
                first.meta["merged"] = int(first.meta.get("merged", 1)) + 1
                first.label = coarse_label(blended, float(first.meta.get("texture", 0.0)))
                working.pop(j)
                merged_any = True
                break
            if merged_any:
                break
    return working


def _merge_small_with_nearby(
    regions: list[Detection],
    *,
    small_fraction: float = 0.15,
    max_distance_factor: float = 2.0,
) -> list[Detection]:
    """Merge small regions into nearby larger ones.

    A chunk of chicken in a bowl of curry is one dish, not two. Small regions
    (< 15% of the largest region's area) get absorbed by the nearest large
    region if they're within 2x their combined bounding box size.
    """
    if len(regions) < 2:
        return regions

    regions = sorted(regions, key=lambda d: d.area_px, reverse=True)
    largest_area = regions[0].area_px

    merged = []
    absorbed = set()

    for i, small in enumerate(regions):
        if i in absorbed:
            continue
        if small.area_px > small_fraction * largest_area:
            merged.append(small)
            continue

        # Find nearest large region
        best_target = None
        best_dist = float("inf")
        sx, sy, sw, sh = small.bbox
        scx, scy = sx + sw / 2, sy + sh / 2

        for j, large in enumerate(regions):
            if j <= i or j in absorbed:
                continue
            if large.area_px <= small.area_px:
                continue
            lx, ly, lw, lh = large.bbox
            lcx, lcy = lx + lw / 2, ly + lh / 2
            dist = ((scx - lcx) ** 2 + (scy - lcy) ** 2) ** 0.5
            max_dist = max_distance_factor * (max(sw, sh) + max(lw, lh))
            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best_target = j

        if best_target is not None:
            target = regions[best_target]
            # Merge into target
            if small.mask is not None and target.mask is not None:
                union = target.mask | small.mask
                target.mask = union
                target.area_px = int(union.sum())
                target.bbox = _bbox_of(union)
                lab_a = np.asarray(target.meta.get("mean_lab", [0, 0, 0]), dtype=np.float64)
                lab_b = np.asarray(small.meta.get("mean_lab", [0, 0, 0]), dtype=np.float64)
                total = float(target.area_px + small.area_px) or 1.0
                blended = (lab_a * target.area_px + lab_b * small.area_px) / total
                target.meta["mean_lab"] = [round(float(v), 2) for v in blended]
                target.meta["merged"] = int(target.meta.get("merged", 1)) + 1
            else:
                target.area_px += small.area_px
                target.bbox = (
                    min(target.bbox[0], small.bbox[0]),
                    min(target.bbox[1], small.bbox[1]),
                    max(target.bbox[0] + target.bbox[2], small.bbox[0] + small.bbox[2]) - min(target.bbox[0], small.bbox[0]),
                    max(target.bbox[1] + target.bbox[3], small.bbox[1] + small.bbox[3]) - min(target.bbox[1], small.bbox[1]),
                )
            absorbed.add(i)
        else:
            merged.append(small)

    return merged


def _merge_overlaps(regions: list[Detection], threshold: float = 0.55) -> list[Detection]:
    """Fuse regions whose boxes overlap heavily, after colour merging."""
    regions = sorted(regions, key=lambda d: d.area_px, reverse=True)
    merged: list[Detection] = []
    for region in regions:
        target = next((m for m in merged if _iou(m, region) > threshold), None)
        if target is None:
            merged.append(region)
            continue
        tx, ty, tw, th = target.bbox
        rx, ry, rw, rh = region.bbox
        x, y = min(tx, rx), min(ty, ry)
        target.bbox = (x, y, max(tx + tw, rx + rw) - x, max(ty + th, ry + rh) - y)
        if target.mask is not None and region.mask is not None:
            target.mask = target.mask | region.mask
            target.area_px = int(target.mask.sum())
        else:
            target.area_px += region.area_px
    return merged


detector = FoodDetector()
