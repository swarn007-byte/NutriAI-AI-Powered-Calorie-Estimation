"""Runtime configuration for the Nutri-AI backend.

Everything is read from the environment with sane local-development defaults so
`uvicorn main:app --reload` works on a clean machine with no setup.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DATA_DIR = PROJECT_DIR / "data"
# Overridable because training does not always run where the service does: on
# Kaggle the repo is mounted read-only under /kaggle/input and the checkpoint has
# to land in /kaggle/working, and a container deploy usually mounts weights.
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (BACKEND_DIR / "models"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or (BACKEND_DIR / "uploads"))


def _load_dotenv() -> None:
    """Minimal .env reader (python-dotenv is not a hard dependency)."""
    for candidate in (BACKEND_DIR / ".env", PROJECT_DIR / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Single flat settings object — easy for one person to hold in their head."""

    app_name = "Nutri-AI"
    version = "0.9.0-mvp"

    # --- Database -------------------------------------------------------
    # Postgres is the documented target (design.md §8.2). SQLite is the
    # zero-setup default so the project runs before Postgres exists.
    database_url = os.getenv("DATABASE_URL") or f"sqlite:///{BACKEND_DIR / 'nutriai.db'}"

    # --- Auth -----------------------------------------------------------
    jwt_secret = os.getenv("JWT_SECRET") or "dev-only-insecure-secret-change-me"
    jwt_algorithm = "HS256"
    jwt_ttl_seconds = _env_int("JWT_TTL_SECONDS", 60 * 60 * 24 * 30)

    # --- Nutrition source ----------------------------------------------
    usda_api_key = os.getenv("USDA_API_KEY") or ""
    usda_base_url = os.getenv("USDA_BASE_URL") or "https://api.nal.usda.gov/fdc/v1"
    usda_timeout_s = _env_float("USDA_TIMEOUT_S", 6.0)

    # --- Upload limits (design.md §12.1) --------------------------------
    max_upload_bytes = _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    max_image_dimension = _env_int("MAX_IMAGE_DIMENSION", 1600)
    allowed_mime = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif")

    # --- Pipeline tuning ------------------------------------------------
    # Default plate diameter assumption (design.md §5.3). User-correctable.
    default_plate_diameter_cm = _env_float("DEFAULT_PLATE_DIAMETER_CM", 26.0)
    # Below this confidence an item is flagged low_confidence (design.md §12.2).
    low_confidence_threshold = _env_float("LOW_CONFIDENCE_THRESHOLD", 0.55)
    # Below this the dish is reported as "unrecognized" rather than a wrong guess.
    unrecognized_threshold = _env_float("UNRECOGNIZED_THRESHOLD", 0.28)
    max_items_per_plate = _env_int("MAX_ITEMS_PER_PLATE", 6)

    # --- Model weights (auto-detected; heuristic engine used if absent) --
    yolo_weights = os.getenv("YOLO_WEIGHTS") or "yolov8n.pt"
    midas_model = os.getenv("MIDAS_MODEL") or "MiDaS_small"
    classifier_checkpoint = os.getenv("CLASSIFIER_CHECKPOINT") or str(
        MODEL_DIR / "efficientnet_v1.pt"
    )
    enable_torch_models = _env_bool("ENABLE_TORCH_MODELS", True)

    # --- PortionNet (opt-in) ------------------------------------------------
    # Set CLASSIFIER_ENGINE=portionnet to use PortionNet instead of EfficientNet.
    # Requires torch + the PortionNet repo cloned into backend/tools/portionnet.
    classifier_engine = os.getenv("CLASSIFIER_ENGINE") or "efficientnet"
    portionnet_checkpoint = os.getenv("PORTIONNET_CHECKPOINT") or str(
        MODEL_DIR / "portionnet_seed7.pt"
    )

    # --- Stage 4 over HTTP ----------------------------------------------
    # The trained classifier is the one stage worth hosting separately: it is the
    # only component that needs torch, and keeping it out of this process means
    # the API can deploy on a 512 MB box. Set to the /classify URL of a
    # model_api/ deployment. Unset means "use a local checkpoint if there is
    # one, otherwise the signature prior" — the URL is never required.
    classifier_url = (os.getenv("CLASSIFIER_URL") or "").strip().rstrip("/")
    classifier_token = os.getenv("CLASSIFIER_TOKEN") or ""
    # Generous relative to usda_timeout_s: a free Hugging Face Space cold-starts,
    # and the first request after an idle period pays for the container waking up.
    classifier_timeout_s = _env_float("CLASSIFIER_TIMEOUT_S", 25.0)
    # Horizontal-flip test-time augmentation is cheap and valid for plated food.
    # Keep it configurable because hosted CPU inference may prefer one pass.
    classifier_tta_passes = max(1, _env_int("CLASSIFIER_TTA_PASSES", 2))

    # --- Misc ------------------------------------------------------------
    cors_origins = [
        o.strip()
        for o in (os.getenv("CORS_ORIGINS") or "http://localhost:5173,http://localhost:8000").split(",")
        if o.strip()
    ]
    serve_frontend = _env_bool("SERVE_FRONTEND", True)
    # Give a brand-new account one pre-analysed sample meal so the history, day
    # strip and statistics screens have something to render on first launch.
    seed_welcome_meal = _env_bool("SEED_WELCOME_MEAL", True)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
