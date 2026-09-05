"""Stage-4 classifier as a standalone HTTP service.

Why this is a separate app
--------------------------
The classifier is the only part of Nutri-AI that needs torch. Bundling it into
the API means every deploy of the API ships ~2.5 GB of CUDA-capable wheels and
needs a box that can hold EfficientNet-B3 in memory, for one stage out of five.
Splitting it out means the backend runs on a 512 MB instance and this service
scales — or sleeps — on its own.

Deliberately self-contained: it shares no imports with `backend/`, because it is
deployed from its own directory (a Hugging Face Space, a Fly machine, anything
that runs a container). The two pieces of `backend/classify.py` it has to agree
with are the preprocessing arithmetic and the checkpoint payload shape, and
`backend/tests/test_model_api.py` asserts that agreement numerically rather than
trusting this comment.

Run locally:
    CHECKPOINT_PATH=../backend/models/efficientnet_v1.pt uvicorn app:app --port 7860
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("nutriai.model_api")

# Must match backend/classify.py INPUT_RESOLUTION and the ImageNet constants.
INPUT_RESOLUTION = 300
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "8"))
MAX_BYTES_PER_IMAGE = int(os.getenv("MAX_BYTES_PER_IMAGE", str(4 * 1024 * 1024)))
TTA_PASSES = max(1, int(os.getenv("CLASSIFIER_TTA_PASSES", "2")))

API_TOKEN = os.getenv("API_TOKEN", "").strip()


# ------------------------------------------------------------------ Model


def preprocess(image: Image.Image, size: int = INPUT_RESOLUTION):
    """Identical arithmetic to `backend/classify._preprocess`.

    Any divergence here — a different resample filter, a different channel order —
    shifts the input distribution away from what the model was trained on and
    shows up as quietly worse accuracy rather than as an error.
    """
    import torch

    resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
    array = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def build_backbone(num_classes: int):
    """EfficientNet-B3 with a fresh head, weights supplied by the checkpoint."""
    try:
        import timm

        return timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    except ImportError:
        pass
    from torchvision.models import efficientnet_b3
    import torch.nn as nn

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def resolve_checkpoint() -> Path | None:
    """Local file if given, otherwise pull the weights from the Hugging Face Hub.

    Two sources because the deployment shapes differ: a container mount or a
    baked-in file for self-hosting, and the Hub for a Space that should not carry
    a 50 MB binary in git.
    """
    local = os.getenv("CHECKPOINT_PATH", "").strip()
    if local:
        path = Path(local)
        return path if path.is_file() else None

    repo = os.getenv("HF_REPO_ID", "").strip()
    if not repo:
        return None
    filename = os.getenv("HF_FILENAME", "efficientnet_v1.pt").strip()
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo, filename=filename, token=os.getenv("HF_TOKEN") or None
        )
        log.info("Fetched %s/%s from the Hub", repo, filename)
        return Path(downloaded)
    except Exception as exc:  # network, auth, missing file
        log.error("Could not fetch %s/%s: %s", repo, filename, exc)
        return None


class Classifier:
    """Holds the model for the process lifetime.

    Loaded once at startup, never per request (readme: 'the single most common
    mistake that makes a working pipeline feel broken in a demo').
    """

    def __init__(self) -> None:
        self.model = None
        self.classes: list[str] = []
        self.version = "unloaded"
        self.input_resolution = INPUT_RESOLUTION
        self.error: str | None = None
        self.loaded_at: float | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch

        path = resolve_checkpoint()
        if path is None:
            self.error = (
                "No checkpoint. Set CHECKPOINT_PATH to a local file, or HF_REPO_ID "
                "(plus HF_FILENAME) to pull one from the Hugging Face Hub."
            )
            log.error(self.error)
            return

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            classes = list(payload.get("classes") or [])
            if not classes:
                raise ValueError("checkpoint has no 'classes' list")
            model = build_backbone(len(classes))
            model.load_state_dict(payload["state_dict"])
            model.eval()
        except Exception as exc:
            self.error = f"Checkpoint at {path} failed to load: {exc}"
            log.exception(self.error)
            return

        torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "2"))))
        self.model = model
        self.classes = classes
        self.version = str(payload.get("version") or path.stem)
        self.input_resolution = int(payload.get("input_resolution") or INPUT_RESOLUTION)
        self.error = None
        self.loaded_at = time.time()
        log.info("Loaded %s — %d classes, %dpx", self.version, len(classes), self.input_resolution)

    def predict(self, images: list[Image.Image], top_k: int) -> list[dict[str, Any]]:
        """One forward pass for the whole batch.

        Batched on purpose: a plate carries up to six items, and six separate
        forward passes over the network would cost six round trips from the
        backend for no reason.
        """
        import torch

        assert self.model is not None
        views = [images]
        if TTA_PASSES >= 2:
            views.append([image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images])
        batches = [
            torch.stack([preprocess(image, self.input_resolution) for image in view])
            for view in views
        ]
        with torch.no_grad():
            probabilities = np.mean(
                [torch.softmax(self.model(batch), dim=1).cpu().numpy() for batch in batches],
                axis=0,
            )

        results = []
        width = max(1, min(top_k, len(self.classes)))
        for row in probabilities:
            order = np.argsort(row)[::-1][:width]
            ranked = [
                {"label": self.classes[int(index)], "confidence": round(float(row[index]), 4)}
                for index in order
            ]
            results.append(
                {
                    "label": ranked[0]["label"],
                    "confidence": ranked[0]["confidence"],
                    "alternatives": ranked[1:],
                }
            )
        return results


classifier = Classifier()


# -------------------------------------------------------------------- App

app = FastAPI(
    title="Nutri-AI dish classifier",
    version="1.0.0",
    description="EfficientNet-B3 fine-tuned on Indian and Western plated dishes.",
    docs_url="/docs",
)

# The backend calls this server to server, so CORS is not what protects it —
# API_TOKEN is. Browsers are allowed in only for the interactive docs page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in (os.getenv("CORS_ORIGINS") or "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    """Optional shared-secret auth.

    Unset means open, which is fine for a local run and wrong for a public Space:
    inference costs CPU, and an open endpoint is someone else's free GPU. The
    startup log says which mode it is in so the choice is never accidental.
    """
    if not API_TOKEN:
        return
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provide the shared secret as `Authorization: Bearer <token>`.",
        )


@app.on_event("startup")
def _startup() -> None:
    classifier.load()
    log.info("Auth: %s", "bearer token required" if API_TOKEN else "OPEN (set API_TOKEN to restrict)")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "nutri-ai-classifier",
        "ready": classifier.ready,
        "docs": "/docs",
        "classify": "POST /classify (multipart field `images`)",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Shape mirrors the backend's own /api/health model block, so the two can be
    read side by side when something looks wrong."""
    return {
        "status": "ok" if classifier.ready else "degraded",
        "ready": classifier.ready,
        "engine": "efficientnet_b3" if classifier.ready else "none",
        "version": classifier.version,
        "classes": classifier.classes,
        "input_resolution": classifier.input_resolution,
        "tta_passes": TTA_PASSES,
        "uptime_s": round(time.time() - classifier.loaded_at, 1) if classifier.loaded_at else None,
        "error": classifier.error,
    }


@app.post("/classify", dependencies=[Depends(require_token)])
async def classify_endpoint(
    images: list[UploadFile] = File(..., description="One crop per detected food item."),
    top_k: int = Form(4),
) -> dict[str, Any]:
    if not classifier.ready:
        # 503, not 500: the caller should fall back to its local engine and may
        # succeed on a later request once the checkpoint is in place.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=classifier.error or "Model is not loaded.",
        )
    if not images:
        raise HTTPException(422, "Send at least one image.")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            # 413 and 422 are written as literals rather than status.HTTP_*:
            # Starlette renamed both constants (REQUEST_ENTITY_TOO_LARGE ->
            # CONTENT_TOO_LARGE, UNPROCESSABLE_ENTITY -> UNPROCESSABLE_CONTENT)
            # and this file has to import cleanly across every version a
            # `fastapi>=0.110` pin allows.
            413,
            f"At most {MAX_IMAGES_PER_REQUEST} images per request.",
        )

    decoded: list[Image.Image] = []
    for upload in images:
        payload = await upload.read()
        if len(payload) > MAX_BYTES_PER_IMAGE:
            raise HTTPException(
                413,
                f"{upload.filename or 'image'} exceeds {MAX_BYTES_PER_IMAGE} bytes.",
            )
        try:
            decoded.append(Image.open(io.BytesIO(payload)).convert("RGB"))
        except Exception:
            raise HTTPException(
                422,
                f"{upload.filename or 'image'} is not a readable image.",
            )

    started = time.perf_counter()
    results = classifier.predict(decoded, top_k=top_k)
    return {
        "engine": "efficientnet_b3",
        "version": classifier.version,
        "input_resolution": classifier.input_resolution,
        "tta_passes": TTA_PASSES,
        "took_ms": round((time.perf_counter() - started) * 1000, 1),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
