"""Nutri-AI backend — single FastAPI service (design.md §6, §10).

One process serves the JSON API, orchestrates the five ML stages, and (in
development) serves the built frontend. There is no second service and no
message queue: design.md §2 makes that a deliberate choice for this phase.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import drafts
import nutrition
import pipeline
import schemas
from auth import (
    create_guest_user,
    create_token,
    current_user,
    current_user_or_guest,
    hash_password,
    verify_password,
)
from config import FRONTEND_DIR, UPLOAD_DIR, settings
from db import (
    Meal,
    MealDraft,
    MealItem,
    User,
    count_meals,
    get_draft,
    get_meal,
    get_session,
    get_user_by_email,
    init_db,
    list_meals,
    new_id,
    session_scope,
    stale_draft_ids,
    utcnow,
)
from imaging import InvalidImageError, encode_jpeg
from pipeline import NoFoodDetectedError, PipelineUnavailableError, ScanResult, ScannedItem
from seed import seed_welcome_meal

# How long an abandoned draft keeps its image on disk. Long enough that a user
# who walks away mid-review can come back to the same tab; short enough that
# unreviewed photos aren't an accumulating liability.
DRAFT_TTL = timedelta(hours=6)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("nutriai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _sweep_stale_drafts()
    # design.md §12.1 — the single most important performance decision here.
    pipeline.warm_models()
    yield


app = FastAPI(
    title="Nutri-AI",
    version=settings.version,
    description="AI-based calorie prediction and nutrient estimation from multi-item food images.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Media capability URLs (design.md §12.3)
# --------------------------------------------------------------------------

def _media_token(meal_id: str) -> str:
    return hmac.new(settings.jwt_secret.encode(), meal_id.encode(), hashlib.sha256).hexdigest()[:32]


def _verify_media_token(meal_id: str, token: str) -> bool:
    return hmac.compare_digest(_media_token(meal_id), token or "")


def _image_paths(meal_id: str) -> tuple[Path, Path]:
    return UPLOAD_DIR / f"{meal_id}.jpg", UPLOAD_DIR / f"{meal_id}_thumb.jpg"


def _regions_path(draft_id: str) -> Path:
    return UPLOAD_DIR / f"{draft_id}_regions.png"


def _media_urls_for(row_id: str) -> tuple[str | None, str | None]:
    """Capability URLs for a stored image, keyed on an opaque id.

    Works for meals and drafts alike: `_media_token` signs the id, and a draft
    id is a uuid4 like any other, so nothing here needs to know which it has.
    """
    full, thumb = _image_paths(row_id)
    if not full.exists():
        return None, None
    token = _media_token(row_id)
    return (
        f"/media/{row_id}?t={token}",
        f"/media/{row_id}?t={token}&size=thumb" if thumb.exists() else f"/media/{row_id}?t={token}",
    )


def _media_urls(meal: Meal) -> tuple[str | None, str | None]:
    return _media_urls_for(meal.id)


def _sweep_stale_drafts() -> None:
    """Drop drafts nobody came back to, and their three files.

    Runs at startup rather than on a timer: there is one process and no
    scheduler (design.md §2), and a restart is frequent enough for a 6-hour TTL.
    Best-effort by design — a failure here must not stop the app from booting.
    """
    try:
        with session_scope() as session:
            ids = stale_draft_ids(session, utcnow() - DRAFT_TTL)
            for draft_id in ids:
                row = get_draft(session, draft_id)
                if row is not None:
                    session.delete(row)
        for draft_id in ids:
            for path in (*_image_paths(draft_id), _regions_path(draft_id)):
                path.unlink(missing_ok=True)
        if ids:
            log.info("Swept %d abandoned draft(s) older than %s", len(ids), DRAFT_TTL)
    except Exception as exc:  # pragma: no cover - startup must not depend on this
        log.warning("Draft sweep skipped (%s)", exc)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _num(value: Any) -> float:
    return round(float(value or 0.0), 2)


def _item_out(item: MealItem) -> schemas.ItemOut:
    geometry = dict((item.nutrients or {}).get("_geometry") or {})
    return schemas.ItemOut(
        id=item.id,
        position=int(item.position or 0),
        detected_label=item.detected_label,
        classified_label=item.classified_label,
        display_name=nutrition.display_name(item.classified_label or "unknown"),
        confidence=_num(item.confidence),
        low_confidence=bool(item.low_confidence),
        user_corrected=bool(item.user_corrected),
        weight_estimated=bool(item.weight_estimated),
        estimated_weight_g=_num(item.estimated_weight_g),
        estimated_volume_ml=_num(item.estimated_volume_ml),
        calories=_num(item.calories),
        protein_g=_num(item.protein_g),
        carbs_g=_num(item.carbs_g),
        fat_g=_num(item.fat_g),
        nutrients=dict(item.nutrients or {}),
        bbox=schemas.BoundingBox(**item.bbox) if item.bbox else None,
        outline=list(geometry.get("outline") or []),
        alternatives=list(item.alternatives or []),
        nutrition_source=item.nutrition_source,
        geometry=geometry,
        # Lifted out of the geometry blob rather than stored separately: an
        # existing SQLite file gets no ALTER TABLE from `create_all`, and this is
        # exactly the kind of per-item detail `_geometry` already carries.
        piece_count=geometry.get("piece_count"),
        piece_weight_g=geometry.get("piece_weight_g"),
    )


def _meal_out(meal: Meal, *, warnings: list[str] | None = None, meal_type: str = "unknown") -> schemas.MealOut:
    image_url, thumb_url = _media_urls(meal)
    micros = dict(meal.micros or {})
    return schemas.MealOut(
        meal_id=meal.id,
        captured_at=meal.captured_at,
        image_url=image_url,
        thumb_url=thumb_url,
        image_width=meal.image_width,
        image_height=meal.image_height,
        engine=meal.engine,
        model_versions=dict(meal.model_versions or {}),
        plate_diameter_cm=_num(meal.plate_diameter_cm) or None,
        items=[_item_out(item) for item in meal.items],
        totals=schemas.Totals(
            calories=_num(meal.total_calories),
            protein_g=_num(meal.total_protein_g),
            carbs_g=_num(meal.total_carbs_g),
            fat_g=_num(meal.total_fat_g),
        ),
        micronutrients=micros,
        daily_values=nutrition.daily_value_percent(micros),
        low_confidence=any(item.low_confidence for item in meal.items),
        warnings=warnings if warnings is not None else list((meal.model_versions or {}).get("_warnings") or []),
        timings_ms=dict((meal.model_versions or {}).get("_timings") or {}),
        notes=meal.notes,
        meal_type=meal_type,
    )


def _user_out(user: User) -> schemas.UserOut:
    return schemas.UserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        is_guest=bool(user.is_guest),
        created_at=user.created_at,
        preferences=dict(user.preferences or {}),
    )


def _persist_items(session: Session, meal_id: str, items: list[pipeline.AnalyzedItem]) -> None:
    """Write the analysed items for a meal. Shared by both analysis endpoints."""
    for position, item in enumerate(items):
        session.add(
            MealItem(
                id=new_id(),
                meal_id=meal_id,
                position=position,
                detected_label=item.detected_label,
                classified_label=item.classified_label,
                confidence=item.confidence,
                estimated_weight_g=item.estimated_weight_g,
                estimated_volume_ml=item.estimated_volume_ml,
                calories=item.nutrients["calories"],
                protein_g=item.nutrients["protein_g"],
                carbs_g=item.nutrients["carbs_g"],
                fat_g=item.nutrients["fat_g"],
                low_confidence=item.low_confidence,
                weight_estimated=item.weight_estimated,
                bbox=item.bbox,
                nutrients={**item.nutrients, "_geometry": item.geometry},
                alternatives=item.alternatives,
                nutrition_source=item.nutrition_source,
            )
        )


def _recalculate_meal(meal: Meal) -> None:
    """Re-derive meal totals + micronutrient roll-up from its items (§7.6)."""
    rows = [dict(item.nutrients or {}) for item in meal.items]
    totals = nutrition.sum_nutrients(rows)
    meal.total_calories = totals["calories"]
    meal.total_protein_g = totals["protein_g"]
    meal.total_carbs_g = totals["carbs_g"]
    meal.total_fat_g = totals["fat_g"]
    meal.micros = {key: totals[key] for key in nutrition.MICRO_KEYS if key in totals}


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------

@app.get("/api/health", response_model=schemas.Health, tags=["meta"])
def health() -> schemas.Health:
    # Delegated rather than recomputed here: this used to match backend names
    # against a hardcoded set, which silently reported "heuristic" the moment a
    # backend gained a suffix (`efficientnet_b3@remote`). The pipeline owns the
    # question of which engines are live, so it answers it.
    status = pipeline.model_status()
    return schemas.Health(
        status="ok",
        version=settings.version,
        engine=pipeline.engine_name(),
        database="postgresql" if not settings.is_sqlite else "sqlite",
        models=status,
        limits={
            "max_upload_bytes": settings.max_upload_bytes,
            "max_image_dimension": settings.max_image_dimension,
            "max_items_per_plate": settings.max_items_per_plate,
            "low_confidence_threshold": settings.low_confidence_threshold,
            "unrecognized_threshold": settings.unrecognized_threshold,
            "default_plate_diameter_cm": settings.default_plate_diameter_cm,
            "allowed_mime": list(settings.allowed_mime),
        },
    )


@app.get("/api/nutrition/catalog", response_model=list[schemas.CatalogEntry], tags=["nutrition"])
def nutrition_catalog() -> list[schemas.CatalogEntry]:
    return [schemas.CatalogEntry(**row) for row in nutrition.catalog()]


@app.get("/api/nutrition/lookup", response_model=schemas.NutritionLookupOut, tags=["nutrition"])
def nutrition_lookup(
    food: str = Query(min_length=1, max_length=120),
    session: Session = Depends(get_session),
) -> schemas.NutritionLookupOut:
    """Direct nutrition lookup — internal use + debugging (design.md §10)."""
    label = food.strip().lower().replace(" ", "_")
    per_100g, source = nutrition.resolve_per_100g(session, label)
    return schemas.NutritionLookupOut(
        food=label,
        display_name=str(per_100g.get("display_name") or nutrition.display_name(label)),
        source=source,
        per_100g={k: v for k, v in per_100g.items() if k not in {"display_name", "density_g_per_ml"}},
        density_g_per_ml=float(per_100g.get("density_g_per_ml") or 1.0),
    )


# --------------------------------------------------------------------------
# Auth (design.md §10 — "JWT-based, minimal")
# --------------------------------------------------------------------------

@app.post("/api/auth/guest", response_model=schemas.AuthPayload, tags=["auth"])
def auth_guest(session: Session = Depends(get_session)) -> schemas.AuthPayload:
    user = create_guest_user(session)
    session.flush()
    return schemas.AuthPayload(token=create_token(user.id, is_guest=True), user=_user_out(user))


@app.post("/api/auth/register", response_model=schemas.AuthPayload, tags=["auth"])
def auth_register(
    body: schemas.RegisterRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> schemas.AuthPayload:
    if get_user_by_email(session, body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "That email already has an account.")

    # If the caller is holding a guest token, upgrade that same row so the meals
    # they already analysed carry over instead of being orphaned.
    existing = _optional_user(request, session)
    upgraded = existing is not None and existing.is_guest
    user = existing if upgraded else User(id=new_id())
    user.email = body.email
    user.name = (body.name or "").strip() or body.email.split("@")[0].title()
    user.password_hash = hash_password(body.password)
    user.is_guest = False
    preferences = dict(user.preferences or {})
    preferences.setdefault("calorie_goal", 2000)
    preferences.setdefault("plate_diameter_cm", settings.default_plate_diameter_cm)
    user.preferences = preferences
    session.add(user)
    session.flush()
    # An upgraded guest keeps the welcome meal it was already given, plus whatever
    # it analysed; only an account that starts from scratch needs one seeded.
    if not upgraded:
        seed_welcome_meal(session, user)
    return schemas.AuthPayload(token=create_token(user.id), user=_user_out(user))


@app.post("/api/auth/login", response_model=schemas.AuthPayload, tags=["auth"])
def auth_login(
    body: schemas.LoginRequest, session: Session = Depends(get_session)
) -> schemas.AuthPayload:
    user = get_user_by_email(session, body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect.")
    return schemas.AuthPayload(token=create_token(user.id), user=_user_out(user))


@app.get("/api/auth/me", response_model=schemas.UserOut, tags=["auth"])
def auth_me(user: User = Depends(current_user)) -> schemas.UserOut:
    return _user_out(user)


@app.patch("/api/users/me/preferences", response_model=schemas.UserOut, tags=["auth"])
def update_preferences(
    body: schemas.Preferences,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.UserOut:
    preferences = dict(user.preferences or {})
    preferences.update({k: v for k, v in body.model_dump().items() if v is not None})
    user.preferences = preferences
    session.add(user)
    session.flush()
    return _user_out(user)


def _optional_user(request: Request, session: Session) -> User | None:
    from auth import decode_token
    from db import get_user

    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        payload = decode_token(token.strip())
    except HTTPException:
        return None
    return get_user(session, str(payload.get("sub", "")))


# --------------------------------------------------------------------------
# Meals
# --------------------------------------------------------------------------

@app.post("/api/meals/analyze", response_model=schemas.MealOut, tags=["meals"])
async def analyze_meal(
    image: UploadFile = File(...),
    plate_diameter_cm: float | None = Form(default=None),
    notes: str | None = Form(default=None),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    user: User = Depends(current_user_or_guest),
    session: Session = Depends(get_session),
) -> Any:
    """Multipart upload → full pipeline → structured nutrition (design.md §10).

    The one-shot path: no review step, so piece counts stay at the area guess.
    Kept for API clients and as the fallback the frontend can always reach for.
    """
    payload = await image.read()
    _read_upload_bytes(image, payload)

    preferences = dict(user.preferences or {})
    plate_cm = plate_diameter_cm or preferences.get("plate_diameter_cm") or settings.default_plate_diameter_cm

    # Build location context for regional food detection
    location = None
    if latitude is not None and longitude is not None:
        location = {"latitude": latitude, "longitude": longitude}

    try:
        result = pipeline.analyze_image(
            payload, session=session, plate_diameter_cm=float(plate_cm), location=location,
        )
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except NoFoodDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    meal_id = new_id()
    full_path, thumb_path = _image_paths(meal_id)
    full_path.write_bytes(encode_jpeg(result.image, quality=88, max_dim=1400))
    thumb_path.write_bytes(encode_jpeg(result.image, quality=78, max_dim=420))

    meal = Meal(
        id=meal_id,
        user_id=user.id,
        image_url=f"/media/{meal_id}",
        image_width=result.image.width,
        image_height=result.image.height,
        captured_at=utcnow(),
        engine=result.engine,
        model_versions={
            **result.model_versions,
            "_timings": result.timings_ms,
            "_warnings": result.warnings,
        },
        plate_diameter_cm=float(plate_cm),
        notes=(notes or "").strip() or None,
    )
    session.add(meal)
    _persist_items(session, meal_id, result.items)

    session.flush()
    session.refresh(meal)
    _recalculate_meal(meal)
    session.flush()

    body = _meal_out(meal, warnings=result.warnings, meal_type=result.meal_type).model_dump(mode="json")
    body["low_confidence"] = any(item.low_confidence for item in meal.items)
    body["token"] = create_token(user.id, is_guest=bool(user.is_guest))
    body["user"] = _user_out(user).model_dump(mode="json")
    return JSONResponse(body)


def _read_upload_bytes(image: UploadFile, payload: bytes) -> None:
    """Shared upload validation for the scan and single-shot endpoints."""
    if image.content_type and image.content_type.lower() not in settings.allowed_mime:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported image type {image.content_type!r}. Use JPEG, PNG or WebP.",
        )
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No image data received.")
    if len(payload) > settings.max_upload_bytes:
        megabytes = settings.max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"That image is {len(payload) / 1024 / 1024:.1f} MB — the limit is {megabytes:.0f} MB.",
        )


def _scanned_item_out(item: ScannedItem) -> schemas.ScannedItemOut:
    return schemas.ScannedItemOut(
        index=item.index,
        label=item.label,
        display_name=item.display_name,
        detected_label=item.detected_label,
        category=item.category,
        confidence=_num(item.confidence),
        low_confidence=bool(item.low_confidence),
        unrecognized=bool(item.unrecognized),
        bbox=schemas.BoundingBox(**item.bbox) if item.bbox else None,
        outline=pipeline.normalized_outline(item.mask),
        alternatives=list(item.alternatives or []),
        area_cm2=_num(item.area_cm2),
        piece_weight_g=item.piece_weight_g,
        piece_count=item.piece_count,
        piece_count_estimated=bool(item.piece_count_estimated),
    )


@app.post("/api/meals/scan", response_model=schemas.DraftOut, tags=["meals"])
async def scan_meal(
    image: UploadFile = File(...),
    notes: str | None = Form(default=None),
    user: User = Depends(current_user_or_guest),
    session: Session = Depends(get_session),
) -> Any:
    """Phase one: what's on the plate, for the user to check before we cost it.

    Deliberately carries no nutrition. A calorie figure the user hasn't confirmed
    the *items* for is a number they will remember whether or not it survives
    review, so this endpoint doesn't produce one. Plate size isn't asked for here
    either — the scan only needs it to report areas, and the review step is where
    the user supplies the real one.
    """
    payload = await image.read()
    _read_upload_bytes(image, payload)

    preferences = dict(user.preferences or {})
    provisional_cm = preferences.get("plate_diameter_cm") or settings.default_plate_diameter_cm

    try:
        scan = pipeline.scan_image(payload, plate_diameter_cm=float(provisional_cm))
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except NoFoodDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    draft_id = new_id()
    full_path, thumb_path = _image_paths(draft_id)
    full_path.write_bytes(encode_jpeg(scan.image, quality=88, max_dim=1400))
    thumb_path.write_bytes(encode_jpeg(scan.image, quality=78, max_dim=420))
    _regions_path(draft_id).write_bytes(
        drafts.encode_regions(scan.items, (scan.prepared.height, scan.prepared.width))
    )

    draft = MealDraft(
        id=draft_id,
        user_id=user.id,
        created_at=utcnow(),
        image_width=scan.image.width,
        image_height=scan.image.height,
        plate_diameter_cm=float(provisional_cm),
        engine=scan.engine,
        notes=(notes or "").strip() or None,
        payload=drafts.scan_payload(scan),
    )
    session.add(draft)
    session.flush()

    image_url, thumb_url = _media_urls_for(draft_id)
    body = schemas.DraftOut(
        draft_id=draft_id,
        image_url=image_url,
        thumb_url=thumb_url,
        image_width=scan.image.width,
        image_height=scan.image.height,
        engine=scan.engine,
        plate=scan.plate,
        plate_diameter_cm=float(provisional_cm),
        items=[_scanned_item_out(item) for item in scan.items],
        model_versions=scan.model_versions,
        warnings=scan.warnings,
        timings_ms=scan.timings_ms,
        notes=draft.notes,
    ).model_dump(mode="json")
    # Same as `analyze_meal`: a guest is created on first upload, so the client
    # needs the token that upload minted before it can call anything else.
    body["token"] = create_token(user.id, is_guest=bool(user.is_guest))
    body["user"] = _user_out(user).model_dump(mode="json")
    return JSONResponse(body)


def _apply_edits(scan: ScanResult, edits: list[schemas.ItemEdit]) -> list[ScannedItem]:
    """Fold the user's review into the scanned list.

    An edit addresses a row by its scan `index`, not by list position, so a
    removal earlier in the list can't shift what a later edit refers to. `-1` is
    an item the user typed in: it has no region, and `analyze_scanned` gives it a
    nominal portion.
    """
    by_index = {item.index: item for item in scan.items}
    removed: set[int] = set()
    added: list[ScannedItem] = []

    for edit in edits:
        if edit.index == -1:
            if not edit.label or edit.removed:
                continue
            label = edit.label
            piece_weight = nutrition.piece_weight_g(label)
            added.append(
                ScannedItem(
                    index=-1,
                    label=label,
                    display_name=nutrition.display_name(label),
                    detected_label="user_added",
                    category=nutrition.category_of(label),
                    confidence=1.0,
                    low_confidence=False,
                    unrecognized=False,
                    bbox={},
                    alternatives=[],
                    area_cm2=0.0,
                    mask=None,
                    detection=None,
                    piece_weight_g=piece_weight,
                    piece_count=edit.piece_count if piece_weight is not None else None,
                    piece_count_estimated=False,
                    user_added=True,
                )
            )
            continue

        item = by_index.get(edit.index)
        if item is None:
            continue
        if edit.removed:
            removed.add(edit.index)
            continue
        if edit.label and edit.label != item.label:
            item.label = edit.label
            item.display_name = nutrition.display_name(edit.label)
            item.category = nutrition.category_of(edit.label)
            item.piece_weight_g = nutrition.piece_weight_g(edit.label)
            # A rename can change whether the food is countable at all. Re-guess
            # from the region's own area rather than carrying over a count that
            # belonged to a different dish.
            item.piece_count = pipeline.guess_piece_count(edit.label, item.area_cm2)
            item.piece_count_estimated = item.piece_count is not None
            # The user chose this name, so nothing about it is low-confidence.
            item.low_confidence = False
            item.unrecognized = False
            item.confidence = 1.0
        if edit.piece_count is not None and item.piece_weight_g is not None:
            item.piece_count = int(edit.piece_count)
            item.piece_count_estimated = False

    kept = [item for item in scan.items if item.index not in removed]
    return kept + added


@app.post("/api/meals/{draft_id}/analyze", response_model=schemas.MealOut, tags=["meals"])
def analyze_draft(
    draft_id: str,
    body: schemas.AnalyzeDraftRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Any:
    """Phase two: cost the list the user just reviewed, at the plate size they gave.

    The draft is consumed — one review, one meal. Re-analysing with a different
    plate size is what `PATCH /api/meals/{id}/plate` is for.
    """
    draft = get_draft(session, draft_id)
    if draft is None or draft.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That scan doesn't exist.")

    full_path, _ = _image_paths(draft_id)
    if not full_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That scan's photo is no longer available.")
    regions_path = _regions_path(draft_id)

    scan = drafts.restore_scan(
        full_path.read_bytes(),
        regions_path.read_bytes() if regions_path.exists() else None,
        dict(draft.payload or {}),
        plate_diameter_cm=body.plate_diameter_cm,
    )
    scan.items = _apply_edits(scan, list(body.items))
    if not scan.items:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Every item was removed — add at least one before analysing.",
        )

    try:
        result = pipeline.analyze_scanned(
            scan, session=session, plate_diameter_cm=body.plate_diameter_cm
        )
    except NoFoodDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    # The draft's photo becomes the meal's photo: rename rather than re-encode, so
    # the bytes the user reviewed are the bytes that get stored.
    meal_id = new_id()
    meal_full, meal_thumb = _image_paths(meal_id)
    draft_full, draft_thumb = _image_paths(draft_id)
    draft_full.replace(meal_full)
    if draft_thumb.exists():
        draft_thumb.replace(meal_thumb)
    else:
        meal_thumb.write_bytes(encode_jpeg(result.image, quality=78, max_dim=420))
    regions_path.unlink(missing_ok=True)

    notes = body.notes if body.notes is not None else draft.notes
    meal = Meal(
        id=meal_id,
        user_id=user.id,
        image_url=f"/media/{meal_id}",
        image_width=draft.image_width or result.image.width,
        image_height=draft.image_height or result.image.height,
        captured_at=utcnow(),
        engine=result.engine,
        model_versions={
            **result.model_versions,
            "_timings": result.timings_ms,
            "_warnings": result.warnings,
        },
        plate_diameter_cm=float(body.plate_diameter_cm),
        notes=(notes or "").strip() or None,
    )
    session.add(meal)
    _persist_items(session, meal_id, result.items)
    session.delete(draft)

    session.flush()
    session.refresh(meal)
    _recalculate_meal(meal)
    session.flush()

    body_out = _meal_out(meal, warnings=result.warnings).model_dump(mode="json")
    body_out["token"] = create_token(user.id, is_guest=bool(user.is_guest))
    body_out["user"] = _user_out(user).model_dump(mode="json")
    return JSONResponse(body_out)


@app.get("/api/meals/{meal_id}", response_model=schemas.MealOut, tags=["meals"])
def read_meal(
    meal_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.MealOut:
    meal = get_meal(session, meal_id)
    if meal is None or meal.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That meal doesn't exist.")
    return _meal_out(meal)


@app.delete("/api/meals/{meal_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["meals"])
def delete_meal(
    meal_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Response:
    """A real delete: the row *and* the stored image go (design.md §12.3)."""
    meal = get_meal(session, meal_id)
    if meal is None or meal.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That meal doesn't exist.")
    for path in _image_paths(meal_id):
        path.unlink(missing_ok=True)
    session.delete(meal)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.patch("/api/meals/{meal_id}/items/{item_id}", response_model=schemas.MealOut, tags=["meals"])
def correct_item(
    meal_id: str,
    item_id: str,
    body: schemas.CorrectionRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.MealOut:
    """User correction — the seed of the retraining feedback loop (design.md §10)."""
    meal = get_meal(session, meal_id)
    if meal is None or meal.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That meal doesn't exist.")
    item = next((row for row in meal.items if row.id == item_id), None)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That item isn't part of this meal.")
    if body.classified_label is None and body.estimated_weight_g is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Send a new label, a new weight, or both.")

    label = body.classified_label or item.classified_label or "unknown"
    weight = float(body.estimated_weight_g if body.estimated_weight_g is not None else item.estimated_weight_g)

    nutrients, source = pipeline.recompute_item(label, weight, session)
    geometry = dict((item.nutrients or {}).get("_geometry") or {})
    geometry["method"] = "user-corrected"
    if body.estimated_weight_g is not None:
        # A hand-set weight no longer equals count × grams per piece, so keeping
        # the count would let the UI display "4 pieces · 180 g" and be lying
        # about one of the two numbers.
        geometry.pop("piece_count", None)
        geometry.pop("piece_weight_g", None)
        geometry.pop("piece_count_estimated", None)

    item.classified_label = label
    item.estimated_weight_g = weight
    item.estimated_volume_ml = round(weight / max(nutrition.density_for(label), 0.05), 1)
    item.calories = nutrients["calories"]
    item.protein_g = nutrients["protein_g"]
    item.carbs_g = nutrients["carbs_g"]
    item.fat_g = nutrients["fat_g"]
    item.nutrients = {**nutrients, "_geometry": geometry}
    item.nutrition_source = source
    item.user_corrected = True
    item.low_confidence = False
    item.confidence = 1.0
    session.add(item)
    session.flush()
    session.refresh(meal)
    _recalculate_meal(meal)
    session.flush()
    return _meal_out(meal)


@app.delete("/api/meals/{meal_id}/items/{item_id}", response_model=schemas.MealOut, tags=["meals"])
def delete_item(
    meal_id: str,
    item_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.MealOut:
    """Remove an item from a meal and recalculate totals."""
    meal = get_meal(session, meal_id)
    if meal is None or meal.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That meal doesn't exist.")
    item = next((row for row in meal.items if row.id == item_id), None)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That item isn't part of this meal.")
    session.delete(item)
    session.flush()
    session.refresh(meal)
    _recalculate_meal(meal)
    session.flush()
    return _meal_out(meal)


@app.patch("/api/meals/{meal_id}/plate", response_model=schemas.MealOut, tags=["meals"])
def adjust_plate(
    meal_id: str,
    body: schemas.PlateAdjustRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.MealOut:
    """Re-measure portions against a corrected plate diameter (design.md §20, risk 1).

    Plate width enters the pipeline in exactly one place — the pixel→cm factor —
    so a correction re-runs stage 3's arithmetic with a different factor rather
    than applying a scaling rule to the weights that came out of it. That is not
    a pedantic distinction: `pipeline.remeasure_for_plate` explains why no fixed
    power of the diameter is available to scale by, and what the old d³ cost.

    Nothing here re-runs a model. Footprint scales with the square of the
    diameter, and everything downstream of it is arithmetic over numbers the
    item already carries.
    """
    meal = get_meal(session, meal_id)
    if meal is None or meal.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That meal doesn't exist.")

    previous = float(meal.plate_diameter_cm or settings.default_plate_diameter_cm)
    if previous <= 0:
        previous = settings.default_plate_diameter_cm
    area_ratio = (body.plate_diameter_cm / previous) ** 2

    for item in meal.items:
        if item.user_corrected:
            continue
        geometry = dict((item.nutrients or {}).get("_geometry") or {})
        label = item.classified_label or "unknown"
        measured = pipeline.remeasure_for_plate(
            label, geometry, area_ratio=area_ratio, session=session
        )
        if measured is None:
            # A counted or hand-added item: its weight never came from pixel
            # area, so the new ruler must not move it. The footprint printed
            # beside it did come from pixel area, so that alone is rescaled —
            # the results page lists it next to measured items' footprints, and
            # at two different scales the smaller number can describe the larger
            # region.
            area_cm2 = float(geometry.get("area_cm2") or 0.0)
            if area_cm2 > 0:
                geometry["area_cm2"] = round(area_cm2 * area_ratio, 2)
                item.nutrients = {**(item.nutrients or {}), "_geometry": geometry}
                session.add(item)
            continue

        item.estimated_weight_g = measured.weight_g
        item.estimated_volume_ml = measured.volume_ml
        item.weight_estimated = measured.weight_estimated
        item.calories = measured.nutrients["calories"]
        item.protein_g = measured.nutrients["protein_g"]
        item.carbs_g = measured.nutrients["carbs_g"]
        item.fat_g = measured.nutrients["fat_g"]
        item.nutrients = {**measured.nutrients, "_geometry": measured.geometry}
        item.nutrition_source = measured.nutrition_source
        session.add(item)

    meal.plate_diameter_cm = body.plate_diameter_cm
    session.flush()
    session.refresh(meal)
    _recalculate_meal(meal)
    session.flush()
    return _meal_out(meal)


# --------------------------------------------------------------------------
# History & daily summary (design.md §13.4, §13.5)
# --------------------------------------------------------------------------

def _require_self(user: User, user_id: str) -> None:
    if user_id not in {user.id, "me"}:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only read your own history.")


@app.get("/api/users/{user_id}/history", response_model=schemas.HistoryOut, tags=["history"])
def user_history(
    user_id: str,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.HistoryOut:
    _require_self(user, user_id)
    meals = list_meals(session, user.id, limit=limit, offset=offset)
    entries = []
    for meal in meals:
        _, thumb_url = _media_urls(meal)
        entries.append(
            schemas.HistoryEntry(
                meal_id=meal.id,
                captured_at=meal.captured_at,
                thumb_url=thumb_url,
                total_calories=_num(meal.total_calories),
                total_protein_g=_num(meal.total_protein_g),
                total_carbs_g=_num(meal.total_carbs_g),
                total_fat_g=_num(meal.total_fat_g),
                item_count=len(meal.items),
                top_items=[
                    nutrition.display_name(item.classified_label or "unknown")
                    for item in sorted(meal.items, key=lambda row: float(row.calories or 0), reverse=True)[:3]
                ],
                engine=meal.engine,
                has_low_confidence=any(item.low_confidence for item in meal.items),
            )
        )
    return schemas.HistoryOut(
        total=count_meals(session, user.id), limit=limit, offset=offset, meals=entries
    )


@app.get("/api/users/{user_id}/summary", response_model=schemas.SummaryOut, tags=["history"])
def user_summary(
    user_id: str,
    days: int = Query(default=14, ge=1, le=90),
    tz_offset: int = Query(default=0, ge=-840, le=840, description="Minutes East of UTC"),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> schemas.SummaryOut:
    """Smallest useful dashboard: today plus a short trend (design.md §13.5)."""
    _require_self(user, user_id)
    shift = timedelta(minutes=tz_offset)
    meals = list_meals(session, user.id, limit=1000, offset=0)

    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "meals": 0.0}
    )
    micro_today: dict[str, float] = {key: 0.0 for key in nutrition.MICRO_KEYS}

    today = (datetime.now(timezone.utc) + shift).date()
    for meal in meals:
        captured = meal.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        local_day = (captured + shift).date()
        key = local_day.isoformat()
        bucket = buckets[key]
        bucket["calories"] += float(meal.total_calories or 0)
        bucket["protein_g"] += float(meal.total_protein_g or 0)
        bucket["carbs_g"] += float(meal.total_carbs_g or 0)
        bucket["fat_g"] += float(meal.total_fat_g or 0)
        bucket["meals"] += 1
        if local_day == today:
            for micro_key, value in (meal.micros or {}).items():
                if micro_key in micro_today:
                    micro_today[micro_key] += float(value or 0)

    def bucket_for(day: date) -> schemas.DayBucket:
        row = buckets.get(day.isoformat())
        return schemas.DayBucket(
            date=day.isoformat(),
            calories=round(row["calories"], 1) if row else 0.0,
            protein_g=round(row["protein_g"], 1) if row else 0.0,
            carbs_g=round(row["carbs_g"], 1) if row else 0.0,
            fat_g=round(row["fat_g"], 1) if row else 0.0,
            meals=int(row["meals"]) if row else 0,
        )

    trend = [bucket_for(today - timedelta(days=offset)) for offset in range(days - 1, -1, -1)]

    streak = 0
    cursor = today
    while buckets.get(cursor.isoformat(), {}).get("meals", 0):
        streak += 1
        cursor -= timedelta(days=1)

    logged = [row for row in trend if row.meals]
    preferences = dict(user.preferences or {})
    calorie_goal = float(preferences.get("calorie_goal") or 2000)
    micro_today = {key: round(value, 1) for key, value in micro_today.items()}

    return schemas.SummaryOut(
        date=today.isoformat(),
        goal={
            "calories": calorie_goal,
            "protein_g": float(preferences.get("protein_goal_g") or round(calorie_goal * 0.25 / 4)),
            "carbs_g": float(preferences.get("carbs_goal_g") or round(calorie_goal * 0.50 / 4)),
            "fat_g": float(preferences.get("fat_goal_g") or round(calorie_goal * 0.25 / 9)),
        },
        today=bucket_for(today),
        micronutrients=micro_today,
        daily_values=nutrition.daily_value_percent(micro_today),
        trend=trend,
        streak_days=streak,
        meals_logged=count_meals(session, user.id),
        best_day=max(logged, key=lambda row: row.calories) if logged else None,
    )


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------

@app.get("/media/{meal_id}", tags=["media"])
def meal_media(meal_id: str, t: str = Query(default=""), size: str = Query(default="full")) -> FileResponse:
    """Capability-URL image access — the token only ever reaches the owner."""
    if not _verify_media_token(meal_id, t):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid image link.")
    full, thumb = _image_paths(meal_id)
    path = thumb if size == "thumb" and thumb.exists() else full
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


# --------------------------------------------------------------------------
# Frontend (single deployed instance, design.md §12.4)
# --------------------------------------------------------------------------
if settings.serve_frontend and FRONTEND_DIR.is_dir():

    class RevalidatedStatics(StaticFiles):
        """Serve /src with `no-cache`, so every module is revalidated.

        There is no bundler here, so module filenames carry no content hash. With
        only the ETag that StaticFiles sends, Chrome applies heuristic freshness
        and will reuse a module for hours without asking — which means a client
        can pair a fresh index.html with a stale store.js and run a module graph
        that never existed. `no-cache` still yields a 304 on the common path; it
        just forbids guessing.
        """

        def file_response(self, *args: Any, **kwargs: Any) -> Response:
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/src", RevalidatedStatics(directory=FRONTEND_DIR / "src"), name="src")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        candidate = (FRONTEND_DIR / full_path).resolve()
        if (
            full_path
            and FRONTEND_DIR.resolve() in candidate.parents
            and candidate.is_file()
        ):
            return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
        return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})
