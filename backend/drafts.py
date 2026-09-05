"""Persisting a scan between the two phases (design.md §6.1, two-phase flow).

A `ScanResult` is mostly JSON-shaped, with two exceptions that decide the whole
design of this module:

  * **region masks** are `HxW` boolean numpy arrays. A list of 640×640 booleans
    per item is not something to put in a JSON column, so all of them go into
    one uint8 PNG label map — pixel value `i + 1` marks item `i`. Flat regions
    compress to a few kB and the round trip is exact.
  * **the prepared image and the plate estimate** are both pure functions of the
    stored photo, so they are re-derived on load rather than stored. `prepare`
    and `estimate_plate` are deterministic; re-running them is cheaper than
    persisting two more arrays, and it keeps the mask and the image guaranteed
    consistent with each other.

What is *not* re-derived is detection or classification: the user's edits are
indexed against the scan's item list, so re-running the detector could silently
renumber the rows underneath them.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

import detection as detection_module
import nutrition
from detection import Detection, PlateEstimate
from imaging import prepare, read_image
from pipeline import WORK_DIM, ScannedItem, ScanResult

log = logging.getLogger("nutriai.drafts")

# uint8 label map: 255 distinct regions, against `max_items_per_plate` of a
# handful. The guard exists so a future limit change fails loudly here.
MAX_REGIONS = 255


def encode_regions(items: list[ScannedItem], shape: tuple[int, int]) -> bytes:
    """Pack every item's mask into one indexed PNG.

    Overlaps resolve to the later item, matching the way the analysis loop reads
    them back one at a time — an ambiguous pixel is counted once, not twice.
    """
    if len(items) > MAX_REGIONS:
        raise ValueError(f"Too many regions for a uint8 label map: {len(items)}")
    height, width = shape
    label_map = np.zeros((height, width), dtype=np.uint8)
    for position, item in enumerate(items):
        if item.mask is None:
            continue
        label_map[item.mask] = position + 1
    buffer = io.BytesIO()
    # No `mode=` — a 2-D uint8 array already infers "L", and the argument is
    # deprecated in Pillow 12.
    Image.fromarray(label_map).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def decode_regions(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Read a label map back, resampled to `shape` if the working size moved.

    Re-encoding the upload as JPEG can shift the working image by a pixel, and a
    mask that is one row taller than the image it indexes is an IndexError at the
    worst possible moment. Nearest-neighbour keeps the values integral.
    """
    label_map = Image.open(io.BytesIO(payload))
    label_map.load()
    if label_map.mode != "L":
        label_map = label_map.convert("L")
    height, width = shape
    if label_map.size != (width, height):
        label_map = label_map.resize((width, height), Image.NEAREST)
    return np.asarray(label_map, dtype=np.uint8)


def item_payload(item: ScannedItem) -> dict[str, Any]:
    """The JSON half of one scanned item."""
    return {
        "index": item.index,
        "label": item.label,
        "detected_label": item.detected_label,
        "confidence": item.confidence,
        "low_confidence": item.low_confidence,
        "unrecognized": item.unrecognized,
        "bbox": item.bbox,
        "alternatives": item.alternatives,
        "area_cm2": item.area_cm2,
        "piece_count": item.piece_count,
        "piece_count_estimated": item.piece_count_estimated,
        # Kept so the mask can be paired back to the right label-map value, and
        # so a region-less row (there are none at scan time, but a future source
        # may produce them) stays region-less.
        "has_region": item.mask is not None,
        "coarse_confidence": (
            round(float(item.detection.confidence), 4) if item.detection is not None else 1.0
        ),
    }


def scan_payload(scan: ScanResult) -> dict[str, Any]:
    """Everything about a scan that belongs in the `meal_drafts.payload` column."""
    return {
        "plate": scan.plate,
        "engine": scan.engine,
        "model_versions": scan.model_versions,
        "timings_ms": scan.timings_ms,
        "warnings": scan.warnings,
        "dropped": scan.dropped,
        "work_size": [scan.prepared.width, scan.prepared.height],
        "items": [item_payload(item) for item in scan.items],
    }


def _denormalize_bbox(bbox: dict[str, Any], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x = int(round(float(bbox.get("x", 0.0)) * width))
    y = int(round(float(bbox.get("y", 0.0)) * height))
    w = max(1, int(round(float(bbox.get("w", 0.0)) * width)))
    h = max(1, int(round(float(bbox.get("h", 0.0)) * height)))
    return x, y, w, h


def restore_scan(
    image_bytes: bytes,
    regions_bytes: bytes | None,
    payload: dict[str, Any],
    *,
    plate_diameter_cm: float | None = None,
) -> ScanResult:
    """Rebuild a `ScanResult` from the stored image, label map and payload.

    The result is ready for `pipeline.analyze_scanned` once the caller has
    applied the user's edits to `.items`.
    """
    image = read_image(image_bytes)
    prepared = prepare(image, work_dim=WORK_DIM)
    shape = (prepared.height, prepared.width)

    stored_plate = dict(payload.get("plate") or {})
    diameter = plate_diameter_cm if plate_diameter_cm is not None else stored_plate.get("diameter_cm")
    plate = detection_module.estimate_plate(prepared, float(diameter) if diameter else None)

    # Take the *stored* radius and centre in preference to the re-derived ones.
    # The photo on disk is a re-encode of the upload, so `estimate_plate` can land
    # a fraction of a percent off where it did during the scan — and weight goes
    # as `cm_per_px²`, so drift there quietly moves every number. The scan's
    # figures are also the ones the user saw an area computed from. The freshly
    # derived *mask* is still used, since only the image can supply that.
    stored_radius_frac = stored_plate.get("radius_frac")
    stored_center = stored_plate.get("center")
    if stored_radius_frac and isinstance(stored_center, (list, tuple)) and len(stored_center) == 2:
        longest = float(max(prepared.width, prepared.height))
        plate = PlateEstimate(
            center=(float(stored_center[0]) * prepared.width, float(stored_center[1]) * prepared.height),
            radius_px=max(1.0, float(stored_radius_frac) * longest),
            mask=plate.mask,
            detected=bool(stored_plate.get("detected", plate.detected)),
            diameter_cm=plate.diameter_cm,
        )

    label_map = decode_regions(regions_bytes, shape) if regions_bytes else None

    items: list[ScannedItem] = []
    for position, row in enumerate(payload.get("items") or []):
        label = str(row.get("label") or "unknown")
        mask: np.ndarray | None = None
        if label_map is not None and row.get("has_region", True):
            candidate = label_map == position + 1
            mask = candidate if candidate.any() else None

        detection: Detection | None = None
        if mask is not None:
            bbox = _denormalize_bbox(dict(row.get("bbox") or {}), shape)
            detection = Detection(
                bbox=bbox,
                label=str(row.get("detected_label") or "unknown"),
                confidence=float(row.get("coarse_confidence") or 1.0),
                mask=mask,
                area_px=int(mask.sum()),
                meta={"source": "draft"},
            )

        items.append(
            ScannedItem(
                index=int(row.get("index", position)),
                label=label,
                display_name=nutrition.display_name(label),
                detected_label=str(row.get("detected_label") or "unknown"),
                category=nutrition.category_of(label),
                confidence=float(row.get("confidence") or 0.0),
                low_confidence=bool(row.get("low_confidence")),
                unrecognized=bool(row.get("unrecognized")),
                bbox=dict(row.get("bbox") or {}),
                alternatives=list(row.get("alternatives") or []),
                area_cm2=float(row.get("area_cm2") or 0.0),
                mask=mask,
                detection=detection,
                piece_weight_g=nutrition.piece_weight_g(label),
                piece_count=row.get("piece_count"),
                piece_count_estimated=bool(row.get("piece_count_estimated")),
            )
        )

    return ScanResult(
        items=items,
        plate={
            **stored_plate,
            "detected": plate.detected,
            "diameter_cm": round(plate.diameter_cm, 1),
            "radius_px": round(plate.radius_px, 1),
            "cm_per_px": round(plate.cm_per_px, 5),
        },
        plate_estimate=plate,
        prepared=prepared,
        image=image,
        engine=str(payload.get("engine") or "heuristic"),
        model_versions=dict(payload.get("model_versions") or {}),
        timings_ms=dict(payload.get("timings_ms") or {}),
        warnings=list(payload.get("warnings") or []),
        dropped=int(payload.get("dropped") or 0),
    )


__all__ = [
    "MAX_REGIONS",
    "decode_regions",
    "encode_regions",
    "item_payload",
    "restore_scan",
    "scan_payload",
]
