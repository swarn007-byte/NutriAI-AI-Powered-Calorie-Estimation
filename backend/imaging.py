"""Shared image I/O and colour-space helpers used by stages 1–4.

Kept in one small module so `detection.py`, `depth.py` and `classify.py` don't
have to import each other just to convert a colour space.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from config import settings


class InvalidImageError(ValueError):
    """Raised for corrupt files or non-image uploads (design.md §12.2)."""


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    rgb: np.ndarray  # float32 HxWx3 in [0, 1]
    lab: np.ndarray  # float32 HxWx3, L in [0, 100]
    original_size: tuple[int, int]

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def read_image(payload: bytes) -> Image.Image:
    """Decode an upload, honouring EXIF rotation. Validates real image content."""
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("That file isn't a readable image.") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if min(image.size) < 48:
        raise InvalidImageError("That image is too small to analyse.")
    return image


def resize_max(image: Image.Image, max_dim: int | None = None) -> Image.Image:
    """Cap the longest edge so a huge photo can't stall the pipeline (§12.1)."""
    limit = max_dim or settings.max_image_dimension
    longest = max(image.size)
    if longest <= limit:
        return image
    scale = limit / float(longest)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def autocorrect(image: Image.Image) -> Image.Image:
    """Light brightness/contrast normalisation (design.md §7.1)."""
    corrected = ImageOps.autocontrast(image, cutoff=1)
    return ImageEnhance.Color(corrected).enhance(1.04)


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0–1) to CIE L*a*b*. Vectorised, no external CV dependency."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T
    scaled = xyz / _D65_WHITE
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(scaled > epsilon, np.cbrt(scaled), (kappa * scaled + 16.0) / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def chroma(lab: np.ndarray) -> np.ndarray:
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).astype(np.float32)


def hue_degrees(lab: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0).astype(np.float32)


def prepare(image: Image.Image, *, work_dim: int = 640) -> PreparedImage:
    original_size = image.size
    working = autocorrect(resize_max(image, work_dim))
    rgb = to_array(working)
    return PreparedImage(image=working, rgb=rgb, lab=rgb_to_lab(rgb), original_size=original_size)


def crop_box(image: Image.Image, box: tuple[int, int, int, int], pad: float = 0.06) -> Image.Image:
    """Crop `(x, y, w, h)` with a little context padding, clamped to bounds."""
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    left = max(0, x - px)
    top = max(0, y - py)
    right = min(image.width, x + w + px)
    bottom = min(image.height, y + h + py)
    if right <= left or bottom <= top:
        return image.copy()
    return image.crop((left, top, right, bottom))


def texture_energy(lab_l: np.ndarray) -> float:
    """Mean absolute Laplacian response — a cheap "how busy is this patch" score."""
    if lab_l.size < 9:
        return 0.0
    kernel_v = np.abs(np.diff(lab_l, axis=0)).mean() if lab_l.shape[0] > 1 else 0.0
    kernel_h = np.abs(np.diff(lab_l, axis=1)).mean() if lab_l.shape[1] > 1 else 0.0
    return float((kernel_v + kernel_h) / 2.0)


def encode_jpeg(image: Image.Image, quality: int = 86, max_dim: int | None = None) -> bytes:
    buffer = io.BytesIO()
    out = resize_max(image, max_dim) if max_dim else image
    out.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
