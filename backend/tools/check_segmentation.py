"""Measure the segmenter against the synthetic samples, whose true areas we know.

`make_samples.py` draws each mound at a known fraction of the plate radius, so
the true food-to-plate area ratio is exact. That makes it a real (if synthetic)
regression check on stage 2 rather than an eyeball test. Run:

    python tools/check_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detection as D  # noqa: E402
from imaging import prepare, read_image, resize_max  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent.parent / "frontend" / "samples"

# name -> (expected food fraction of plate area, expected item count)
# Radii come straight from make_samples.py; fraction = Σ(r/R_plate)².
EXPECTED = {
    "thali": (0.291, 5),
    "dosa": (0.328, 3),
    "breakfast": (0.238, 5),
    "curry-bowl": (0.55, 2),
}


def main() -> int:
    print(f"{'sample':<12} {'mask%':>7} {'true%':>7} {'ratio':>6} {'items':>6} {'want':>5}")
    print("-" * 50)
    failures = 0
    for name, (expected_fraction, expected_items) in EXPECTED.items():
        image = resize_max(read_image((SAMPLES / f"{name}.jpg").read_bytes()))
        prepared = prepare(image, work_dim=640)
        plate = D.estimate_plate(prepared, None)
        mask = D.food_mask(prepared, plate)
        detections = D.detector._detect_segments(prepared, plate)

        fraction = mask.sum() / max(1.0, float(plate.mask.sum()))
        ratio = fraction / expected_fraction
        flag = "" if 0.6 <= ratio <= 1.6 else "  <-- area off"
        if flag or not (expected_items - 2 <= len(detections) <= expected_items + 1):
            failures += 1
        print(
            f"{name:<12} {fraction * 100:6.1f}% {expected_fraction * 100:6.1f}% "
            f"{ratio:6.2f} {len(detections):6d} {expected_items:5d}{flag}"
        )
    print("-" * 50)
    print(f"{len(EXPECTED) - failures}/{len(EXPECTED)} samples within tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
