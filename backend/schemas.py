"""Request/response contracts for the API in design.md §10."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Local pattern instead of pydantic's EmailStr so the backend has no
# email-validator dependency for what is a single field.
_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[A-Za-z]{2,}$")


def _clean_email(value: str) -> str:
    email = value.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address")
    return email


class Health(BaseModel):
    status: str
    version: str
    engine: str
    database: str
    models: dict[str, Any]
    limits: dict[str, Any]


class AuthPayload(BaseModel):
    token: str
    user: "UserOut"


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=120)

    _normalize_email = field_validator("email")(classmethod(lambda cls, v: _clean_email(v)))


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)

    _normalize_email = field_validator("email")(classmethod(lambda cls, v: _clean_email(v)))


class Preferences(BaseModel):
    calorie_goal: int | None = Field(default=None, ge=800, le=6000)
    protein_goal_g: int | None = Field(default=None, ge=20, le=400)
    carbs_goal_g: int | None = Field(default=None, ge=20, le=800)
    fat_goal_g: int | None = Field(default=None, ge=10, le=300)
    plate_diameter_cm: float | None = Field(default=None, ge=12.0, le=45.0)
    theme: str | None = None

    @field_validator("theme")
    @classmethod
    def _theme(cls, value: str | None) -> str | None:
        if value in (None, "dark", "light", "system"):
            return value
        raise ValueError("theme must be dark, light or system")


class UserOut(BaseModel):
    id: str
    email: str | None
    name: str | None
    is_guest: bool
    created_at: datetime
    preferences: dict[str, Any]


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ItemOut(BaseModel):
    id: str
    position: int
    detected_label: str | None
    classified_label: str | None
    display_name: str
    confidence: float
    low_confidence: bool
    user_corrected: bool
    weight_estimated: bool
    estimated_weight_g: float
    estimated_volume_ml: float | None
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    nutrients: dict[str, Any]
    bbox: BoundingBox | None
    outline: list[list[float]] = Field(default_factory=list)
    alternatives: list[dict[str, Any]]
    nutrition_source: str | None
    geometry: dict[str, Any] = Field(default_factory=dict)
    # Set when the weight came from a piece count rather than from geometry, so
    # the results page can say "4 pieces · 260 g" instead of a bare weight.
    # Read out of `geometry` rather than stored in its own column: an existing
    # SQLite file gets no ALTER TABLE from `create_all`, and this is exactly the
    # kind of per-item detail `nutrients["_geometry"]` already carries.
    piece_count: int | None = None
    piece_weight_g: float | None = None


class Totals(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class MealOut(BaseModel):
    meal_id: str
    captured_at: datetime
    image_url: str | None
    thumb_url: str | None
    image_width: int | None
    image_height: int | None
    engine: str | None
    model_versions: dict[str, Any]
    plate_diameter_cm: float | None
    items: list[ItemOut]
    totals: Totals
    micronutrients: dict[str, Any]
    daily_values: dict[str, float]
    low_confidence: bool
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class HistoryEntry(BaseModel):
    meal_id: str
    captured_at: datetime
    thumb_url: str | None
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    item_count: int
    top_items: list[str]
    engine: str | None
    has_low_confidence: bool


class HistoryOut(BaseModel):
    total: int
    limit: int
    offset: int
    meals: list[HistoryEntry]


class DayBucket(BaseModel):
    date: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    meals: int


class SummaryOut(BaseModel):
    date: str
    goal: dict[str, float]
    today: DayBucket
    micronutrients: dict[str, Any]
    daily_values: dict[str, float]
    trend: list[DayBucket]
    streak_days: int
    meals_logged: int
    best_day: DayBucket | None


class CorrectionRequest(BaseModel):
    classified_label: str | None = Field(default=None, max_length=120)
    estimated_weight_g: float | None = Field(default=None, ge=0, le=3000)

    @field_validator("classified_label")
    @classmethod
    def _label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower().replace(" ", "_")
        if not cleaned:
            raise ValueError("classified_label cannot be blank")
        return cleaned


class PlateAdjustRequest(BaseModel):
    plate_diameter_cm: float = Field(ge=12.0, le=45.0)


class NutritionLookupOut(BaseModel):
    food: str
    display_name: str
    source: str
    per_100g: dict[str, Any]
    density_g_per_ml: float


class CatalogEntry(BaseModel):
    label: str
    display_name: str
    category: str
    kcal_per_100g: float
    protein_g: float
    carbs_g: float
    fat_g: float
    density_g_per_ml: float
    # Non-null for foods a person counts. The review page reads this to decide
    # whether an added item gets a piece-count field or a nominal portion, and
    # the footprint to re-guess a count when a region is renamed.
    piece_weight_g: float | None = None
    piece_footprint_cm2: float | None = None


# --------------------------------------------------------------------------
# Two-phase analysis: scan → review → deep pass
# --------------------------------------------------------------------------
#
# The scan carries no nutrition at all, on purpose. Its job is to answer "what
# is on this plate", and a number the user hasn't confirmed yet is a number they
# will remember whether or not it survives review.


class ScannedItemOut(BaseModel):
    """One provisional item, before anything has been costed."""

    index: int
    label: str
    display_name: str
    detected_label: str | None
    category: str
    confidence: float
    low_confidence: bool
    unrecognized: bool
    bbox: BoundingBox | None
    outline: list[list[float]] = Field(default_factory=list)
    alternatives: list[dict[str, Any]]
    area_cm2: float
    # Non-null only for countable foods. `piece_count` is a guess from area and
    # `piece_count_estimated` says so, so the UI can mark it rather than present
    # it as measured.
    piece_weight_g: float | None = None
    piece_count: int | None = None
    piece_count_estimated: bool = False


class DraftOut(BaseModel):
    draft_id: str
    image_url: str | None
    thumb_url: str | None
    image_width: int | None
    image_height: int | None
    engine: str | None
    plate: dict[str, Any] = Field(default_factory=dict)
    plate_diameter_cm: float | None = None
    items: list[ScannedItemOut] = Field(default_factory=list)
    model_versions: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ItemEdit(BaseModel):
    """One row of the reviewed list.

    `index` refers back to `ScannedItemOut.index`, so the server can find the
    region this row was measured from. `-1` means the user added the item by
    hand and there is no region — it gets a nominal portion instead.
    """

    index: int = Field(ge=-1)
    label: str | None = Field(default=None, max_length=120)
    piece_count: int | None = Field(default=None, ge=1, le=60)
    removed: bool = False


class AnalyzeDraftRequest(BaseModel):
    plate_diameter_cm: float = Field(ge=12.0, le=45.0)
    notes: str | None = Field(default=None, max_length=280)
    items: list[ItemEdit] = Field(default_factory=list)


AuthPayload.model_rebuild()
