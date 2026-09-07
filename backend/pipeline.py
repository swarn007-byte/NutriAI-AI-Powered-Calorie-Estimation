"""Pipeline orchestration — the five stages of design.md §6.1, in order.

    Stage 1  input handling / preprocessing        imaging.py
    Stage 2  food detection                        detection.py
    Stage 3  depth → volume → weight               depth.py
    Stage 4  fine-grained dish classification      classify.py   ← the trained model
    Stage 5  nutrition lookup                      nutrition.py

`warm_models()` is called once from the FastAPI lifespan hook. Nothing in this
module ever constructs a model — design.md §12.1 flags per-request model loading
as *the* mistake that makes a working pipeline feel broken in a demo.

The five stages run in two phases, split at the point where the plate scale
first matters:

    scan_image()      stages 1, 2 and 4 — "what is on this plate"
    analyze_scanned() stages 3 and 5    — "how much of it, and what it costs"

The seam is real, not cosmetic. Inside these stages `plate_diameter_cm` reaches
nothing but `PlateEstimate.diameter_cm`, and from there only `cm_per_px` /
`px_area_cm2`; the plate mask and radius come off the image alone. So detection
and classification can answer before the user has told us how big the plate is,
and the user can strike a wrong item off the list before we spend a depth pass
and a nutrition lookup on it. `analyze_image()` still runs both halves in one
call for the single-shot endpoint.

`remeasure_for_plate()` is the one place the plate scale is applied outside a
`PlateEstimate` — for a user who corrects the width *after* the analysis, on a
photo no longer in memory. It reproduces stage 3's arithmetic rather than scaling
its output; its docstring says why the difference is 20%.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

import detection as detection_module
import nutrition
from calorieclip import calorieclip
from classify import classifier
from config import settings
from depth import (
    MAX_DEPTH_CM,
    clamp_weight,
    depth_estimator,
    estimate_volume,
    mask_for,
    mean_depth_cm,
    weight_from_volume,
)
from detection import Detection, PlateEstimate, detector
from imaging import InvalidImageError, PreparedImage, crop_box, prepare, read_image, resize_max

log = logging.getLogger("nutriai.pipeline")

WORK_DIM = 640

# A region has to cover at least this much of the plate, in cm², to be treated
# as food. Below it there is nothing to measure: `estimate_volume` would take
# its `fallback-portion` branch and hand back the category's *minimum* served
# weight, which is how a lemon wedge and a basket weave each became 45 g of
# curry on a plate of samosas. Dropping the region is the honest answer —
# a detection has to earn its area. Items the *user* adds are exempt; they have
# no region by construction and get a nominal portion instead.
MIN_ITEM_AREA_CM2 = 0.5

# Widest plausible piece count for a single region, used when guessing a count
# from area. Beyond a dozen the guess is worthless and the user is faster
# typing the number than correcting ours.
MAX_PIECE_GUESS = 12

# ---- Regional food bias (based on GPS location) ----------------------------
# Maps Indian regions to their signature food items. When CLIP is uncertain,
# it biases towards foods common in the user's region.
REGION_FOODS: dict[str, list[str]] = {
    "south": [
        "dosa", "idli", "vada_pav", "masala_dosa", "medu_vada",
        "sambhar", "coconut_chutney", "green_chutney",
        "rasam", "curd_yogurt", "pongal",
    ],
    "north": [
        "roti_chapati", "naan", "paratha", "chole_bhature", "rajma_masala",
        "dal_makhani", "paneer_butter_masala", "butter_chicken",
        "chicken_biryani", "dal_tadka",
    ],
    "east": [
        "fish_curry", "plain_rice", "dal", "begun_bhaja",
        "luchi", "rasgulla", "mishti_doii",
    ],
    "west": [
        "vada_pav", "pav_bhaji", "dhokla", "thepla",
        "dal_bati", "gatte_ki_sabzi",
    ],
    "northeast": [
        "fish_curry", "plain_rice", "bamboo_shoot", "thukpa",
        "momos", "chicken_curry",
    ],
    "central": [
        "poha", "dal_tadka", "roti_chapati", "bhindi_masala",
        "aloo_gobi", "jeera_rice",
    ],
}

# Approximate lat/lng bounds for Indian regions
REGION_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "south": {"lat": (8.0, 15.5), "lng": (72.0, 81.0)},
    "north": {"lat": (28.0, 37.0), "lng": (68.0, 98.0)},
    "east": {"lat": (21.0, 28.0), "lng": (85.0, 98.0)},
    "west": {"lat": (20.0, 28.0), "lng": (68.0, 78.0)},
    "northeast": {"lat": (21.0, 30.0), "lng": (89.0, 98.0)},
    "central": {"lat": (21.0, 27.0), "lng": (74.0, 85.0)},
}


def _get_region_from_coords(latitude: float, longitude: float) -> str | None:
    """Map GPS coordinates to an Indian region name."""
    for region, bounds in REGION_BOUNDS.items():
        lat_min, lat_max = bounds["lat"]
        lng_min, lng_max = bounds["lng"]
        if lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max:
            return region
    return None


def _get_region_bias(location: dict | None) -> list[str] | None:
    """Get regional food list from location, or None if outside India."""
    if not location:
        return None
    lat = location.get("latitude")
    lng = location.get("longitude")
    if lat is None or lng is None:
        return None
    region = _get_region_from_coords(lat, lng)
    if region and region in REGION_FOODS:
        log.info("Location %.2f,%.2f → region=%s, biasing foods: %s", lat, lng, region, REGION_FOODS[region][:5])
        return REGION_FOODS[region]
    log.info("Location %.2f,%.2f → no regional bias", lat, lng)
    return None


class NoFoodDetectedError(RuntimeError):
    """Nothing food-like in the frame → HTTP 422 (design.md §10)."""


class PipelineUnavailableError(RuntimeError):
    """A stage never initialised → HTTP 503 (design.md §10)."""


@dataclass
class AnalyzedItem:
    detected_label: str
    classified_label: str
    display_name: str
    confidence: float
    low_confidence: bool
    unrecognized: bool
    estimated_weight_g: float
    estimated_volume_ml: float
    weight_estimated: bool
    nutrients: dict[str, float]
    nutrition_source: str
    bbox: dict[str, float]
    alternatives: list[dict[str, Any]]
    geometry: dict[str, Any]

    @property
    def calories(self) -> float:
        return float(self.nutrients.get("calories", 0.0))


@dataclass
class AnalysisResult:
    items: list[AnalyzedItem]
    totals: dict[str, float]
    plate: dict[str, Any]
    engine: str
    model_versions: dict[str, str]
    timings_ms: dict[str, float]
    image: Image.Image
    warnings: list[str] = field(default_factory=list)
    meal_type: str = "unknown"  # breakfast, lunch, dinner, snack


@dataclass
class ScannedItem:
    """One provisional item: named and measured in area, but not yet costed.

    `mask` is the region the weight will later be integrated over. It is a numpy
    array and so cannot cross a JSON boundary — the API layer persists these as
    a PNG label map and hands them back on the second call.
    """

    index: int
    label: str
    display_name: str
    detected_label: str
    category: str
    confidence: float
    low_confidence: bool
    unrecognized: bool
    bbox: dict[str, float]
    alternatives: list[dict[str, Any]]
    area_cm2: float
    mask: np.ndarray | None = None
    detection: Detection | None = None
    piece_weight_g: float | None = None
    piece_count: int | None = None
    piece_count_estimated: bool = False
    # True for a row the user typed in rather than the detector found. It has no
    # region, so its weight comes from `nutrition.nominal_portion_g`.
    user_added: bool = False

    @property
    def countable(self) -> bool:
        return self.piece_weight_g is not None


@dataclass
class ScanResult:
    """Everything the review step needs, and everything the deep pass resumes from."""

    items: list[ScannedItem]
    plate: dict[str, Any]
    plate_estimate: PlateEstimate
    prepared: PreparedImage
    image: Image.Image
    engine: str
    model_versions: dict[str, str]
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    dropped: int = 0
    meal_type: str = "unknown"  # breakfast, lunch, dinner, snack


def _detect_meal_type(items: list[ScannedItem]) -> str:
    """Detect meal type based on time of day and food items present.

    Uses heuristics:
    - 6-10 AM: likely breakfast (idli, dosa, poha, upma, paratha)
    - 12-3 PM: likely lunch (rice, dal, sabzi, roti)
    - 7-10 PM: likely dinner (similar to lunch)
    - Other times: based on food items
    """
    from datetime import datetime, timezone

    hour = datetime.now(timezone.utc).hour

    # Time-based heuristics (IST offset = +5:30)
    ist_hour = (hour + 5) % 24

    # Get all labels
    labels = {item.label for item in items}

    # Breakfast foods
    breakfast_foods = {"idli", "dosa", "masala_dosa", "poha", "upma", "paratha", "medu_vada", "poori"}
    # Lunch/dinner foods
    main_meal_foods = {"rice", "plain_rice", "jeera_rice", "roti_chapati", "naan", "dal_tadka", "dal_makhani"}

    if 6 <= ist_hour < 10:
        if labels & breakfast_foods:
            return "breakfast"
        return "breakfast"  # Time suggests breakfast
    elif 12 <= ist_hour < 15:
        return "lunch"
    elif 19 <= ist_hour < 22:
        return "dinner"
    elif labels & breakfast_foods:
        return "breakfast"
    elif labels & main_meal_foods:
        return "lunch"  # or dinner, but lunch is more common
    else:
        return "snack"


def normalized_outline(
    mask: np.ndarray | None,
    *,
    max_points: int = 180,
) -> list[list[float]]:
    """Return an ordered, normalized polygon around a pixel region.

    The detector's masks are the actual regions used for area and volume. This
    converts their pixel boundary to a compact polygon so clients can draw the
    same shape instead of expanding every region to its bounding rectangle.
    """
    if mask is None or not np.asarray(mask).any():
        return []

    region = np.asarray(mask, dtype=bool)
    height, width = region.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_edge(start: tuple[int, int], end: tuple[int, int]) -> None:
        edges.setdefault(start, []).append(end)

    ys, xs = np.nonzero(region)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if y == 0 or not region[y - 1, x]:
            add_edge((x, y), (x + 1, y))
        if x == width - 1 or not region[y, x + 1]:
            add_edge((x + 1, y), (x + 1, y + 1))
        if y == height - 1 or not region[y + 1, x]:
            add_edge((x + 1, y + 1), (x, y + 1))
        if x == 0 or not region[y, x - 1]:
            add_edge((x, y + 1), (x, y))

    if not edges:
        return []

    # Mask regions are normally single connected components. At a diagonal
    # contact there can be more than one outgoing edge; choose the continuation
    # with the smallest clockwise turn so the visible outer boundary remains
    # ordered rather than jumping across the region.
    start = min(edges, key=lambda point: (point[1], point[0]))
    points: list[tuple[int, int]] = []
    current = start
    previous_direction: tuple[int, int] | None = None
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    limit = len(edges) * 4 + 8

    for _ in range(limit):
        points.append(current)
        candidates = [
            end for end in edges.get(current, []) if (current, end) not in used
        ]
        if not candidates:
            break
        if previous_direction is None:
            next_point = candidates[0]
        else:
            def turn_rank(end: tuple[int, int]) -> tuple[int, int, int]:
                direction = (end[0] - current[0], end[1] - current[1])
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                old_index = directions.index(previous_direction)
                new_index = directions.index(direction)
                return ((new_index - old_index) % 4, end[1], end[0])

            next_point = min(candidates, key=turn_rank)
        used.add((current, next_point))
        previous_direction = (next_point[0] - current[0], next_point[1] - current[1])
        current = next_point
        if current == start:
            break

    if len(points) < 3:
        return []

    # Remove collinear points first; this keeps the payload small for long
    # straight edges before the point-count simplification below.
    reduced: list[tuple[int, int]] = []
    for point in points:
        if len(reduced) >= 2:
            ax, ay = reduced[-2]
            bx, by = reduced[-1]
            cx, cy = point
            if (bx - ax) * (cy - by) == (by - ay) * (cx - bx):
                reduced[-1] = point
                continue
        reduced.append(point)
    points = reduced

    if len(points) > max_points:
        step = max(1, (len(points) + max_points - 1) // max_points)
        points = points[::step]

    return [
        [round(max(0.0, min(1.0, x / width)), 5), round(max(0.0, min(1.0, y / height)), 5)]
        for x, y in points
    ]


def warm_models() -> None:
    """Load all networks exactly once, at startup (design.md §12.1)."""
    started = time.perf_counter()
    detector.load()
    depth_estimator.load()
    classifier.load()
    if settings.enable_calorieclip:
        calorieclip.load()
    if settings.enable_segmentation:
        from food_segmentation import segmentation
        segmentation.load()
    if settings.enable_indian_classification:
        from indian_food_classifier import classifier as indian_classifier
        indian_classifier.load()
    _warm_numeric_stack()
    log.info(
        "Models ready in %.2fs — detector=%s depth=%s classifier=%s calorieclip=%s",
        time.perf_counter() - started,
        detector.backend,
        depth_estimator.backend,
        classifier.backend,
        calorieclip.backend,
    )


def _warm_numeric_stack() -> None:
    """Pay sklearn/BLAS thread-pool setup now instead of on the first upload.

    Untouched, the first KMeans call costs 2–10 s while joblib spins up its
    worker pool — which lands squarely on the first user's analysis and makes a
    0.5 s pipeline feel broken. Same reasoning as §12.1's model preloading.
    """
    try:
        blank = Image.new("RGB", (96, 96), (140, 120, 96))
        prepared = prepare(blank, work_dim=96)
        plate = detection_module.estimate_plate(prepared, None)
        detection_module._cluster_food(prepared, plate.mask, max_clusters=3)
    except Exception as exc:
        log.debug("Numeric warm-up skipped (%s)", exc)


def model_status() -> dict[str, Any]:
    from food_segmentation import segmentation
    from indian_food_classifier import classifier as indian_classifier
    
    return {
        "detection": {"backend": detector.backend, "version": detector.version, "ready": detector.ready},
        "depth": {
            "backend": depth_estimator.backend,
            "version": depth_estimator.version,
            "ready": depth_estimator.ready,
        },
        "classification": {
            "backend": classifier.backend,
            "version": classifier.version,
            "ready": classifier.ready,
            "trained_model": classifier.is_trained_model,
            "classes": len(classifier.classes),
            "tta_passes": settings.classifier_tta_passes,
            # What answers if the primary engine cannot. Surfaced because
            # "trained_model: true" with a remote primary is only true while the
            # remote is reachable, and this is what says what happens if it
            # isn't.
            "fallback": classifier.fallbacks,
        },
        "calorieclip": {
            "backend": calorieclip.backend,
            "version": calorieclip.version,
            "ready": calorieclip.backend == "calorieclip",
            "enabled": settings.enable_calorieclip,
        },
        "segmentation": {
            "backend": segmentation.backend,
            "version": segmentation.version,
            "ready": segmentation.backend != "unloaded",
            "enabled": settings.enable_segmentation,
        },
        "indian_classifier": {
            "backend": indian_classifier.backend,
            "version": indian_classifier.version,
            "ready": indian_classifier.backend != "unloaded",
            "enabled": settings.enable_indian_classification,
            "classes": 24,
        },
        "nutrition": {
            "backend": "usda+ifct" if settings.usda_api_key else "ifct",
            "version": "ifct-2017/usda-fdc",
            "ready": True,
        },
    }


def engine_name() -> str:
    """Whether this result came from the trained stack or the fallback engine.

    Public because `/api/health` reports the same label and must not drift from
    what an analysis reports. "Trained" counts a remote classifier: it is the
    fine-tuned weights, just not in this process.
    """
    trained = classifier.is_trained_model
    pretrained = detector.backend == "yolov8" and depth_estimator.backend == "midas"
    calorieclip_active = calorieclip.backend == "calorieclip"

    if calorieclip_active and trained:
        return "calorieclip+classifier"
    if calorieclip_active:
        return "calorieclip"
    if trained and pretrained:
        return "full"
    if trained or pretrained:
        return "partial"
    return "heuristic"


def _resolve_label(prediction_label: str, confidence: float, coarse: str) -> tuple[str, bool, bool]:
    """Apply the confidence policy from design.md §12.2.

    Returns `(label, low_confidence, unrecognized)`. Below the unrecognized
    threshold the pipeline refuses to name a dish rather than assert a wrong one,
    falling back to the coarse group ("Lentil or Yellow Dish") which is a
    description it can actually stand behind.
    """
    if confidence < settings.unrecognized_threshold:
        if coarse in nutrition.COARSE_FALLBACK:
            return coarse, True, True
        aliased = nutrition.DETECTOR_ALIASES.get(coarse)
        return (aliased or prediction_label), True, True
    return prediction_label, confidence < settings.low_confidence_threshold, False


def _plate_summary(plate: PlateEstimate, prepared: PreparedImage) -> dict[str, Any]:
    return {
        "detected": plate.detected,
        "diameter_cm": round(plate.diameter_cm, 1),
        "radius_px": round(plate.radius_px, 1),
        "cm_per_px": round(plate.cm_per_px, 5),
        "center": [round(plate.center[0] / prepared.width, 4), round(plate.center[1] / prepared.height, 4)],
        "radius_frac": round(plate.radius_px / max(prepared.width, prepared.height), 4),
    }


def _model_versions() -> dict[str, str]:
    return {
        "detection": f"{detector.backend}:{detector.version}",
        "depth": f"{depth_estimator.backend}:{depth_estimator.version}",
        "classification": f"{classifier.backend}:{classifier.version}",
        "calorieclip": f"{calorieclip.backend}:{calorieclip.version}",
        "nutrition": "usda+ifct" if settings.usda_api_key else "ifct",
    }


def guess_piece_count(label: str, area_cm2: float) -> int | None:
    """How many pieces an `area_cm2` region probably holds, or `None`.

    Four samosas in a basket come back from the detector as one blob, so area ÷
    per-piece footprint is the only handle we have on "how many". It is a guess
    and the UI is required to say so; the point is that it is a *correctable*
    guess, which a back-solved total weight never was.
    """
    footprint = nutrition.piece_footprint_cm2(label)
    if footprint is None or footprint <= 0:
        return None
    return int(min(max(round(area_cm2 / footprint), 1), MAX_PIECE_GUESS))


# Labels that a tiny region should never have — they are full-dish names that
# the classifier hallucinates on small sauce/chutney bowls.
_SAUCE_IMPOSTERS: frozenset[str] = frozenset({
    "chicken_curry", "butter_chicken", "fish_curry", "egg_curry",
    "paneer_butter_masala", "rajma_masala", "chole_masala",
    "mixed_veg_curry", "dal_tadka", "sambhar", "manchurian",
})


def _classify_sauce(prediction: "classify.Prediction", fallback: str) -> str:
    """Reclassify a tiny region that was mislabelled as a full dish.

    Uses the dominant hue from the prediction's coarse label (if available)
    and the pixel colour to pick the most likely condiment.
    """
    from classify import SIGNATURES

    # The prediction's top label is wrong at this size; look at colour.
    # Extract a rough hue from the signature RGB to decide chutney type.
    rgb = SIGNATURES.get(fallback, {}).get("rgb", (128, 128, 128))
    r, g, b = rgb

    # Green-dominant → mint/coriander chutney
    if g > r and g > b and g > 90:
        return "green_chutney"
    # Pale/white → coconut chutney
    if r > 200 and g > 200 and b > 180:
        return "coconut_chutney"
    # Default for small brown/red bowls → tamarind chutney
    return "tamarind_chutney"


def scan_image(
    payload: bytes,
    *,
    plate_diameter_cm: float | None = None,
    location: dict | None = None,
) -> ScanResult:
    """Phase one: name and outline what is on the plate. No nutrition at all.

    Runs stages 1, 2 and 4. Stage 3's depth pass is skipped entirely — it is
    only useful for volume, and volume needs a plate scale the user has not
    given us yet. That makes the scan the cheap half as well as the honest one.

    `plate_diameter_cm` is still accepted, and still only sets the pixel→cm
    scale. The scan uses it to report each region's area and to guess piece
    counts; the deep pass recomputes both from whatever the user finally enters.

    `location` is an optional dict with latitude/longitude for regional food
    detection bias — helps CLIP pick the right regional dish.
    """
    if not (detector.ready and classifier.ready):
        raise PipelineUnavailableError("The analysis pipeline is still initialising.")

    timings: dict[str, float] = {}
    warnings: list[str] = []
    clock = time.perf_counter()

    # ---- Stage 1: input -------------------------------------------------
    image = resize_max(read_image(payload))
    prepared = prepare(image, work_dim=WORK_DIM)
    timings["input"] = round((time.perf_counter() - clock) * 1000, 1)

    # ---- plate reference frame -----------------------------------------
    clock = time.perf_counter()
    plate = detection_module.estimate_plate(prepared, plate_diameter_cm)
    if not plate.detected:
        warnings.append(
            "No plate rim found — assumed the plate fills most of the frame. "
            "Adjust the plate size if the weights look off."
        )

    # ---- Stage 2: detection --------------------------------------------
    detections = detector.detect(prepared, plate)
    timings["detection"] = round((time.perf_counter() - clock) * 1000, 1)
    if not detections:
        raise NoFoodDetectedError("No food items found in that photo.")

    shape = (prepared.height, prepared.width)
    plate_area_px = float(max(1.0, plate.mask.sum()))
    global_food_mask = detection_module.food_mask(prepared, plate)

    # ---- Stage 4, pass 1: measure every region, then classify once ------
    # Two passes rather than one loop because stage 4 may live on another host.
    # Per item it would be up to `max_items_per_plate` HTTPS round trips, and
    # against a Space that cold-starts the round trip costs far more than the
    # forward pass. Gathering the crops first makes one analysis one request —
    # and the local torch path gets the same win from a single stacked batch.
    measured: list[dict[str, Any]] = []
    dropped = 0
    for found in detections:
        # Resolve the region now, not in the deep pass, because whether it has
        # any measurable area decides whether this item exists at all.
        resolved = mask_for(found, global_food_mask, shape)
        area_px = int(resolved.sum())
        area_cm2 = area_px * plate.px_area_cm2
        if area_px == 0 or area_cm2 <= MIN_ITEM_AREA_CM2:
            dropped += 1
            log.info(
                "Dropped a detection with no measurable area (label=%s conf=%.2f area_px=%d): "
                "nothing to weigh",
                found.label,
                found.confidence,
                area_px,
            )
            continue

        region = found.mask if found.mask is not None and found.mask.any() else resolved
        mean_lab = prepared.lab[region].mean(axis=0)
        texture = float(np.std(prepared.lab[..., 0][region]))

        measured.append(
            {
                "found": found,
                "mask": resolved,
                "area_cm2": area_cm2,
                "coarse": (
                    found.label
                    if found.meta.get("source") == "segmenter"
                    else detection_module.coarse_label(mean_lab, texture)
                ),
                "area_frac": int(region.sum()) / plate_area_px,
                "crop": crop_box(prepared.image, found.bbox),
            }
        )

    if not measured:
        # Every region failed the area test. Saying "no food" is more accurate
        # than serving up a plate of minimum portions.
        raise NoFoodDetectedError("Nothing on that plate was big enough to measure.")
    if dropped:
        warnings.append(
            f"Ignored {dropped} region{'s' if dropped > 1 else ''} too small to measure — "
            "add anything that's missing below."
        )

    clock = time.perf_counter()

    # CLIP: first classify the FULL plate to get context
    # If location is provided, get regional food bias
    region_bias = _get_region_bias(location) if location else None
    plate_labels = classifier.classify_plate(prepared.image, top_k=6, region_bias=region_bias)
    if plate_labels:
        log.info("CLIP plate-level: %s", [(p["label"], p["confidence"]) for p in plate_labels])

    # Then classify each crop, biased by plate context
    predictions = classifier.predict_crops(
        [row["crop"] for row in measured],
        coarse_labels=[row["coarse"] for row in measured],
        area_fracs=[row["area_frac"] for row in measured],
        plate_labels=plate_labels,
    )
    timings["classification"] = round((time.perf_counter() - clock) * 1000, 1)

    # ---- Stage 4, pass 2: interpret each prediction ---------------------
    items: list[ScannedItem] = []
    for index, (row, prediction) in enumerate(zip(measured, predictions)):
        found = row["found"]
        coarse = row["coarse"]

        label, low_confidence, unrecognized = _resolve_label(
            prediction.label, prediction.confidence, coarse
        )

        # The v1 checkpoint was trained on 24 classes and therefore cannot
        # emit two catalog dishes that this app still supports. For fragmented
        # plate photos, the detector gives us a reliable visual region type:
        # a compact pale textured bowl is chutney, and a small round saturated
        # bowl is sambhar. Keep these explicit fallbacks local to that recovery
        # path rather than rewriting every low-confidence classifier result.
        region_kind = str(found.meta.get("region_kind") or "")
        if found.meta.get("source") == "fragmented-plate-segmenter":
            if region_kind == "pale_textured":
                label, low_confidence, unrecognized = "coconut_chutney", True, False
            elif region_kind == "saturated_round" and prediction.label in {"dal_tadka", "fish_curry"}:
                label, low_confidence, unrecognized = "sambhar", True, False

        # A tiny region (< 12 cm²) that the classifier calls a curry is almost
        # certainly a dipping sauce or chutney, not a full dish.  Override the
        # label based on colour: green → green_chutney, brown/red →
        # tamarind_chutney, pale → coconut_chutney.  This catches the common
        # mistake of momo/samosa chutneys being labelled "chicken_curry".
        area_cm2 = float(row["area_cm2"])
        if area_cm2 < 12.0 and label in _SAUCE_IMPOSTERS:
            label = _classify_sauce(prediction, label)
            low_confidence, unrecognized = True, False
        guess = guess_piece_count(label, area_cm2)
        items.append(
            ScannedItem(
                index=index,
                label=label,
                display_name=nutrition.display_name(label),
                detected_label=coarse,
                category=nutrition.category_of(label),
                confidence=round(float(prediction.confidence), 4),
                low_confidence=low_confidence,
                unrecognized=unrecognized,
                bbox=_normalized_bbox(found.bbox, shape),
                alternatives=[
                    {
                        "label": alternative["label"],
                        "display_name": nutrition.display_name(alternative["label"]),
                        "confidence": alternative["confidence"],
                    }
                    for alternative in prediction.alternatives
                ],
                area_cm2=round(area_cm2, 2),
                mask=row["mask"],
                detection=found,
                piece_weight_g=nutrition.piece_weight_g(label),
                piece_count=guess,
                piece_count_estimated=guess is not None,
            )
        )

    timings["total"] = round(sum(timings.values()), 1)

    return ScanResult(
        items=items,
        plate=_plate_summary(plate, prepared),
        plate_estimate=plate,
        prepared=prepared,
        image=image,
        engine=engine_name(),
        model_versions=_model_versions(),
        timings_ms=timings,
        warnings=warnings,
        dropped=dropped,
        meal_type=_detect_meal_type(items),
    )


def analyze_scanned(
    scan: ScanResult,
    *,
    session: Session | None = None,
    plate_diameter_cm: float | None = None,
) -> AnalysisResult:
    """Phase two: cost the reviewed list.

    Runs stages 3 and 5 over `scan.items` as they stand — the caller has already
    applied the user's edits (renames, counts, removals, additions). Weight comes
    from the first of these that applies:

      1. a piece count on a countable food → `count × grams per piece`. A number
         the user can see and correct beats a number derived from a blob.
      2. a region → `estimate_volume`, exactly as the one-shot pipeline always did
      3. neither, i.e. a hand-added item → `nutrition.nominal_portion_g`
    """
    if not depth_estimator.ready:
        raise PipelineUnavailableError("The analysis pipeline is still initialising.")
    if not scan.items:
        raise NoFoodDetectedError("There's nothing left on this plate to analyse.")

    plate = scan.plate_estimate
    prepared = scan.prepared
    if plate_diameter_cm is not None and abs(plate_diameter_cm - plate.diameter_cm) > 1e-6:
        # `PlateEstimate` is frozen in spirit if not in code; rebuild it so the
        # new scale flows through `cm_per_px` rather than mutating shared state.
        plate = PlateEstimate(
            center=plate.center,
            radius_px=plate.radius_px,
            mask=plate.mask,
            detected=plate.detected,
            diameter_cm=float(plate_diameter_cm),
        )

    timings = dict(scan.timings_ms)
    timings.pop("total", None)
    warnings = list(scan.warnings)
    shape = (prepared.height, prepared.width)

    # ---- Stage 3a: depth map -------------------------------------------
    # Deferred to here: a depth map is only ever consumed by `estimate_volume`,
    # so running it during the scan would have been work spent on items the user
    # was about to delete.
    clock = time.perf_counter()
    depth_map = depth_estimator.relative_depth(prepared)
    if depth_map is None and depth_estimator.backend == "midas":
        warnings.append("Depth model failed on this image — portion sizes use the shape prior.")
    global_food_mask = detection_module.food_mask(prepared, plate)
    timings["depth"] = round((time.perf_counter() - clock) * 1000, 1)

    volume_ms = 0.0
    nutrition_ms = 0.0
    items: list[AnalyzedItem] = []

    for position, scanned in enumerate(scan.items):
        label = scanned.label

        # ---- Stage 5a: composition + density ---------------------------
        clock = time.perf_counter()
        per_100g, source = nutrition.resolve_per_100g(session, label)
        nutrition_ms += (time.perf_counter() - clock) * 1000
        category = str(per_100g.get("category") or "unknown")
        density = float(per_100g.get("density_g_per_ml") or nutrition.density_for(label))

        # ---- Stage 3b: volume → weight ---------------------------------
        clock = time.perf_counter()
        piece_weight = nutrition.piece_weight_g(label)
        count = scanned.piece_count if piece_weight is not None else None

        if count is not None:
            weight_g = round(count * piece_weight, 1)
            # Re-measured rather than carried over from the scan. The scan sized
            # this region against the *provisional* plate width, and the results
            # page prints footprints for counted and geometry-measured items in
            # one list — at two different scales the smaller number can describe
            # the larger region. The weight is `count × grams`, which needs no
            # scale at all, and that is precisely why this was easy to leave
            # stale. A hand-added item has no region, so it keeps its zero.
            area_cm2 = (
                round(int(scanned.mask.sum()) * plate.px_area_cm2, 2)
                if scanned.mask is not None
                else scanned.area_cm2
            )
            geometry = {
                "area_cm2": area_cm2,
                "mean_height_cm": 0.0,
                "peak_height_cm": 0.0,
                "density_g_per_ml": round(density, 3),
                "method": "piece-count",
                "piece_count": int(count),
                "piece_weight_g": round(piece_weight, 1),
                "piece_count_estimated": bool(scanned.piece_count_estimated),
            }
            volume_ml = round(weight_g / max(density, 0.05), 1)
            # Not "estimated": a count is a discrete quantity the user can see
            # and has been given the chance to fix, unlike a clamped volume.
            weight_estimated = bool(scanned.piece_count_estimated)
        elif scanned.detection is not None:
            volume = estimate_volume(
                scanned.detection,
                category=category,
                density_g_per_ml=density,
                plate=plate,
                depth=depth_map,
                food_mask=global_food_mask,
                shape=shape,
            )
            weight_g = volume.weight_g
            volume_ml = volume.volume_ml
            weight_estimated = volume.clamped or volume.method == "fallback-portion"
            geometry = {
                "area_cm2": volume.area_cm2,
                "mean_height_cm": volume.mean_height_cm,
                "peak_height_cm": volume.peak_height_cm,
                "density_g_per_ml": round(volume.density_g_per_ml, 3),
                "method": volume.method,
            }
        else:
            # Hand-added: no region, so no area and no depth. A nominal portion
            # for the category is the only defensible number, and it is flagged
            # as estimated so the UI offers the weight slider.
            weight_g = nutrition.nominal_portion_g(label)
            volume_ml = round(weight_g / max(density, 0.05), 1)
            weight_estimated = True
            geometry = {
                "area_cm2": 0.0,
                "mean_height_cm": 0.0,
                "peak_height_cm": 0.0,
                "density_g_per_ml": round(density, 3),
                "method": "nominal-portion",
            }
        volume_ms += (time.perf_counter() - clock) * 1000

        geometry["position"] = position
        geometry["coarse_confidence"] = round(
            float(scanned.detection.confidence) if scanned.detection is not None else 1.0, 4
        )
        geometry["outline"] = normalized_outline(scanned.mask)

        # ---- Stage 5b: scale nutrients ---------------------------------
        clock = time.perf_counter()
        nutrients = nutrition.scale_nutrients(per_100g, weight_g)

        # ---- CalorieCLIP override: blend or replace calories with direct prediction ----
        calorieclip_calories = 0.0
        if calorieclip.backend == "calorieclip" and scanned.detection is not None:
            try:
                crop = crop_box(scan.image, scanned.detection.bbox)
                cc_result = calorieclip.predict(crop)
                calorieclip_calories = cc_result["calories"]
                table_calories = nutrients.get("calories", 0.0)
                if calorieclip_calories > 0 and table_calories > 0:
                    ratio = calorieclip_calories / table_calories
                    # 1) Known piece-weighted foods: trust the table (samosa, idli, etc.)
                    if piece_weight is not None:
                        calorieclip_calories = 0.0
                    # 2) CalorieCLIP is way off (>2x or <0.5x): reject
                    elif ratio > 2.0 or ratio < 0.5:
                        log.debug(
                            "CalorieCLIP rejected for %s: predicted %.0f vs table %.0f (ratio %.2f)",
                            label, calorieclip_calories, table_calories, ratio,
                        )
                        calorieclip_calories = 0.0
                    # 3) Within reasonable range: blend 70% table + 30% CalorieCLIP
                    else:
                        blended = table_calories * 0.7 + calorieclip_calories * 0.3
                        nutrients["calories"] = round(blended, 1)
                        nutrients["protein_g"] = round(nutrients.get("protein_g", 0) * (blended / table_calories), 1)
                        nutrients["carbs_g"] = round(nutrients.get("carbs_g", 0) * (blended / table_calories), 1)
                        nutrients["fat_g"] = round(nutrients.get("fat_g", 0) * (blended / table_calories), 1)
                        source = "calorieclip+table"
            except Exception as exc:
                log.debug("CalorieCLIP prediction failed for item %d (%s)", position, exc)

        nutrition_ms += (time.perf_counter() - clock) * 1000

        items.append(
            AnalyzedItem(
                detected_label=scanned.detected_label,
                classified_label=label,
                display_name=nutrition.display_name(label),
                confidence=round(float(scanned.confidence), 4),
                low_confidence=scanned.low_confidence,
                unrecognized=scanned.unrecognized,
                estimated_weight_g=weight_g,
                estimated_volume_ml=volume_ml,
                weight_estimated=weight_estimated,
                nutrients=nutrients,
                nutrition_source=source if calorieclip_calories == 0 else "calorieclip",
                bbox=scanned.bbox,
                alternatives=scanned.alternatives,
                geometry=geometry,
            )
        )

    timings["volume"] = round(volume_ms, 1)
    timings["nutrition"] = round(nutrition_ms, 1)
    timings["total"] = round(sum(timings.values()), 1)

    totals = nutrition.sum_nutrients([item.nutrients for item in items])
    # No low-confidence warning here on purpose: every item already carries a
    # `low_confidence` flag, so a warning would restate a field the caller has,
    # and the client can attach the correction affordance to the specific items
    # rather than to a sentence about them.

    return AnalysisResult(
        items=items,
        totals=totals,
        plate=_plate_summary(plate, prepared),
        engine=scan.engine,
        model_versions=scan.model_versions,
        timings_ms=timings,
        image=scan.image,
        warnings=warnings,
        meal_type=scan.meal_type,
    )


def analyze_image(
    payload: bytes,
    *,
    session: Session | None = None,
    plate_diameter_cm: float | None = None,
    location: dict | None = None,
) -> AnalysisResult:
    """Run both phases back to back, with nobody reviewing in between.

    This is the single-shot endpoint: same five stages, same order, no user in
    the loop. Piece counts are the area guess, since there is no one to correct
    them. Location is used for regional food detection bias.
    """
    scan = scan_image(payload, plate_diameter_cm=plate_diameter_cm, location=location)
    return analyze_scanned(scan, session=session, plate_diameter_cm=plate_diameter_cm)


def _normalized_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> dict[str, float]:
    """Store boxes as fractions so the UI can overlay them at any render size."""
    height, width = shape
    x, y, w, h = bbox
    return {
        "x": round(max(0.0, x / width), 5),
        "y": round(max(0.0, y / height), 5),
        "w": round(min(1.0, w / width), 5),
        "h": round(min(1.0, h / height), 5),
    }


def recompute_item(
    label: str,
    weight_g: float,
    session: Session | None = None,
) -> tuple[dict[str, float], str]:
    """Recompute nutrients after a user correction (design.md §10 PATCH)."""
    per_100g, source = nutrition.resolve_per_100g(session, label)
    return nutrition.scale_nutrients(per_100g, max(0.0, float(weight_g))), source


@dataclass
class Remeasurement:
    """One item's stage-3 numbers, recomputed against a different plate width."""

    weight_g: float
    volume_ml: float
    weight_estimated: bool
    geometry: dict[str, Any]
    nutrients: dict[str, float]
    nutrition_source: str


def remeasure_for_plate(
    label: str,
    geometry: dict[str, Any],
    *,
    area_ratio: float,
    session: Session | None = None,
) -> Remeasurement | None:
    """Re-run stage 3 for a corrected plate width. `None` = leave this item alone.

    A plate correction is not a scaling of the finished weight, it is the same
    measurement taken with a different ruler — and the two are not the same
    number. `depth.mean_depth_cm` grows *sub*-linearly with footprint
    (`AREA_DEPTH_EXPONENT`, deliberately ≈0.2 so a wide helping cannot inflate
    its own depth), and both that prior and the served-weight envelope clip. So
    weight follows no fixed power of the diameter: measured end to end on a real
    photo it came out near d^2.3, while the d³ this handler used to assume
    over-reported a 26 → 34 cm correction by 20% against a fresh analysis of the
    same photo at 34 cm. Two routes, one stated plate size, two answers.

    Redoing the arithmetic costs nothing, because none of it needs the photo:

    - footprint is an area, so it scales with the square of the diameter;
    - `depth.integrate_volume` normalises the elevation field to unit mean, which
      makes volume *exactly* footprint × mean depth — the shape of the mound
      cancels out, so MiDaS never has to see the image again;
    - the stored geometry already carries the footprint and the density that the
      forward pass used.

    `None` comes back for an item whose weight never went through pixel area at
    all: a counted one (four samosas are four samosas on any plate) or a
    hand-added one with no region. Its footprint still wants rescaling, so the
    caller gets `None` rather than an unchanged copy — the distinction is
    "do not re-measure", not "measured the same".
    """
    method = str(geometry.get("method") or "")
    area_cm2 = float(geometry.get("area_cm2") or 0.0) * float(area_ratio)
    if method in {"piece-count", "nominal-portion"} or area_cm2 <= 0:
        return None

    per_100g, source = nutrition.resolve_per_100g(session, label)
    category = str(per_100g.get("category") or "unknown")
    density = float(
        geometry.get("density_g_per_ml")
        or per_100g.get("density_g_per_ml")
        or nutrition.density_for(label)
    )

    volume_ml = area_cm2 * mean_depth_cm(category, area_cm2)
    weight_g, clamped = clamp_weight(weight_from_volume(volume_ml, density), category)
    if clamped and volume_ml > 0:
        volume_ml = weight_g / density

    mean_height = volume_ml / area_cm2
    # `peak / mean` describes the mound's shape, not its scale, so it survives a
    # change of ruler untouched — which is the whole reason the peak can be
    # carried forward without the elevation field it was measured from.
    previous_mean = float(geometry.get("mean_height_cm") or 0.0)
    peakiness = float(geometry.get("peak_height_cm") or 0.0) / previous_mean if previous_mean > 0 else 0.0

    updated = {
        **geometry,
        "area_cm2": round(area_cm2, 2),
        "mean_height_cm": round(mean_height, 2),
        "peak_height_cm": round(min(mean_height * peakiness, MAX_DEPTH_CM * 2.5), 2),
        "density_g_per_ml": round(density, 3),
        "method": "plate-recalibrated",
    }
    return Remeasurement(
        weight_g=round(weight_g, 1),
        volume_ml=round(volume_ml, 1),
        weight_estimated=clamped,
        geometry=updated,
        nutrients=nutrition.scale_nutrients(per_100g, weight_g),
        nutrition_source=source,
    )


__all__ = [
    "MAX_PIECE_GUESS",
    "MIN_ITEM_AREA_CM2",
    "AnalysisResult",
    "AnalyzedItem",
    "InvalidImageError",
    "NoFoodDetectedError",
    "PipelineUnavailableError",
    "Remeasurement",
    "ScanResult",
    "ScannedItem",
    "normalized_outline",
    "analyze_image",
    "analyze_scanned",
    "guess_piece_count",
    "model_status",
    "recompute_item",
    "remeasure_for_plate",
    "scan_image",
    "warm_models",
]
