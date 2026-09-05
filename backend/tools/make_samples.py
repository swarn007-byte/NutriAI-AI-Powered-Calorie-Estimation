"""Generate synthetic thali photographs for local testing and the UI's sample tray.

These are not a substitute for real food photos — they exist so the pipeline,
the API and the frontend can be exercised end-to-end without shipping
copyrighted imagery in the repo. Run:

    python tools/make_samples.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "samples"

SIZE = 900


def _noise(size: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, size // int(scale)), max(2, size // int(scale))))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _table(seed: int, tone: tuple[int, int, int]) -> Image.Image:
    grain = _noise(SIZE, 6, seed) * 0.18 + _noise(SIZE, 40, seed + 1) * 0.3
    base = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for channel in range(3):
        base[..., channel] = tone[channel] * (0.82 + 0.32 * grain)
    # soft vignette so the plate reads as the brightest thing in frame
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    radial = np.sqrt(((yy - SIZE / 2) / SIZE) ** 2 + ((xx - SIZE / 2) / SIZE) ** 2)
    base *= (1.0 - 0.55 * np.clip(radial - 0.18, 0, 1))[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _mound(
    draw: ImageDraw.ImageDraw,
    centre: tuple[float, float],
    radius: float,
    colour: tuple[int, int, int],
    *,
    seed: int,
    lumps: int = 13,
    jitter: float = 0.3,
    shadow: ImageDraw.ImageDraw | None = None,
) -> None:
    """An irregular blob — food is never a clean circle."""
    rng = random.Random(seed)
    cx, cy = centre
    # Sum a few low-order harmonics with fixed integer frequencies: high
    # frequencies turn the outline into a star instead of a serving of food.
    harmonics = [(k, jitter / k, rng.random() * math.tau) for k in (2, 3, 5)]
    points: list[tuple[float, float]] = []
    for step in range(96):
        angle = step / 96.0 * math.tau
        wobble = 1.0 + sum(amp * math.sin(k * angle + phase) for k, amp, phase in harmonics)
        r = radius * float(np.clip(wobble, 0.7, 1.25))
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    if shadow is not None:
        # Every real photo has ambient occlusion where food meets the plate.
        # Without it, pale food on a pale plate has literally no edge to find,
        # which makes synthetic tests harder than reality rather than easier.
        offset = radius * 0.10
        shadow.polygon([(x + offset, y + offset * 1.4) for x, y in points], fill=118)
    draw.polygon(points, fill=colour)
    for _ in range(lumps):
        angle = rng.random() * math.tau
        dist = rng.random() * radius * 0.62
        lump_r = radius * rng.uniform(0.16, 0.34)
        lx, ly = cx + dist * math.cos(angle), cy + dist * math.sin(angle)
        # One multiplier for the whole lump — sampling per channel would tint it
        # a random hue instead of just lightening or darkening it.
        level = rng.uniform(0.9, 1.1)
        shade = tuple(int(np.clip(channel * level, 0, 255)) for channel in colour)
        draw.ellipse((lx - lump_r, ly - lump_r, lx + lump_r, ly + lump_r), fill=shade)


class Scene:
    """Two-pass renderer: contact shadows are blurred under the food, not over it."""

    def __init__(self, table_seed: int, tone: tuple[int, int, int]) -> None:
        self.image = _table(table_seed, tone)
        self.shadow = Image.new("L", (SIZE, SIZE), 0)
        self._pending: list[tuple] = []

    def plate(self, centre: tuple[float, float], radius: float, rim: tuple[int, int, int]) -> None:
        draw = ImageDraw.Draw(self.image)
        cx, cy = centre
        draw.ellipse(
            (cx - radius * 1.05, cy - radius * 1.05, cx + radius * 1.05, cy + radius * 1.05),
            fill=(196, 194, 190),
        )
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=rim)
        draw.ellipse(
            (cx - radius * 0.9, cy - radius * 0.9, cx + radius * 0.9, cy + radius * 0.9),
            fill=tuple(min(255, channel + 7) for channel in rim),
        )

    def food(self, centre, radius, colour, **kwargs) -> None:
        self._pending.append((centre, radius, colour, kwargs))

    def render(self, blur: float = 1.1) -> Image.Image:
        shadow_draw = ImageDraw.Draw(self.shadow)
        for centre, radius, _colour, kwargs in self._pending:
            _mound(_NULL_DRAW, centre, radius, (0, 0, 0), shadow=shadow_draw, **kwargs)
        softened = self.shadow.filter(ImageFilter.GaussianBlur(SIZE * 0.012))
        darkened = Image.eval(self.image, lambda value: int(value * 0.72))
        self.image = Image.composite(darkened, self.image, softened)
        draw = ImageDraw.Draw(self.image)
        for centre, radius, colour, kwargs in self._pending:
            _mound(draw, centre, radius, colour, **kwargs)
        return self.image.filter(ImageFilter.GaussianBlur(blur))


class _NullDraw:
    """Swallows the food pass while only shadows are being accumulated."""

    def polygon(self, *args, **kwargs) -> None:
        return None

    def ellipse(self, *args, **kwargs) -> None:
        return None


_NULL_DRAW = _NullDraw()


def thali() -> Image.Image:
    """Rice + dal + curry + roti + salad — the canonical Indian plate."""
    scene = Scene(3, (96, 74, 58))
    cx, cy = SIZE / 2, SIZE / 2 + 10
    radius = SIZE * 0.40
    scene.plate((cx, cy), radius, (243, 241, 236))
    scene.food((cx - radius * 0.34, cy + radius * 0.16), radius * 0.31, (247, 243, 232), seed=11)
    scene.food((cx + radius * 0.36, cy - radius * 0.28), radius * 0.23, (214, 155, 46), seed=22)
    scene.food((cx + radius * 0.30, cy + radius * 0.34), radius * 0.22, (176, 68, 38), seed=33)
    scene.food((cx - radius * 0.30, cy - radius * 0.44), radius * 0.26, (222, 191, 138), seed=44, jitter=0.16)
    scene.food((cx + radius * 0.02, cy + radius * 0.60), radius * 0.16, (86, 142, 62), seed=55)
    return scene.render()


def dosa_plate() -> Image.Image:
    scene = Scene(9, (72, 68, 78))
    cx, cy = SIZE / 2, SIZE / 2
    radius = SIZE * 0.42
    scene.plate((cx, cy), radius, (238, 236, 230))
    scene.food((cx - radius * 0.14, cy - radius * 0.06), radius * 0.52, (226, 183, 106), seed=71, jitter=0.12, lumps=6)
    scene.food((cx + radius * 0.50, cy - radius * 0.34), radius * 0.18, (222, 148, 52), seed=72)
    scene.food((cx + radius * 0.52, cy + radius * 0.26), radius * 0.16, (238, 240, 228), seed=73)
    return scene.render(1.0)


def breakfast() -> Image.Image:
    scene = Scene(17, (118, 108, 96))
    cx, cy = SIZE / 2, SIZE / 2 - 6
    radius = SIZE * 0.38
    scene.plate((cx, cy), radius, (247, 246, 243))
    for index, (dx, dy) in enumerate([(-0.38, -0.22), (0.02, -0.34), (-0.16, 0.18)]):
        scene.food(
            (cx + radius * dx, cy + radius * dy),
            radius * 0.24,
            (250, 249, 243),
            seed=90 + index,
            jitter=0.1,
            lumps=5,
        )
    scene.food((cx + radius * 0.44, cy + radius * 0.30), radius * 0.22, (216, 142, 44), seed=95)
    scene.food((cx - radius * 0.02, cy + radius * 0.62), radius * 0.13, (236, 238, 226), seed=96)
    return scene.render(1.0)


def bowl_curry() -> Image.Image:
    scene = Scene(23, (54, 58, 64))
    cx, cy = SIZE / 2, SIZE / 2
    radius = SIZE * 0.34
    scene.plate((cx, cy), radius, (232, 233, 236))
    scene.food((cx, cy), radius * 0.74, (188, 84, 44), seed=131, jitter=0.09, lumps=18)
    scene.food((cx - radius * 0.18, cy - radius * 0.12), radius * 0.2, (245, 240, 226), seed=132, lumps=4)
    return scene.render(1.2)


SAMPLES = {
    "thali.jpg": thali,
    "dosa.jpg": dosa_plate,
    "breakfast.jpg": breakfast,
    "curry-bowl.jpg": bowl_curry,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in SAMPLES.items():
        path = OUT_DIR / name
        factory().save(path, "JPEG", quality=88)
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
