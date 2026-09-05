"""PostgreSQL persistence layer (design.md §8.2).

One database, four tables. SQLAlchemy's generic types are used so the exact
same models run on Postgres (JSONB, native UUID) and on SQLite, which keeps
local development possible before a Postgres instance exists.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON

from config import settings

# JSONB on Postgres (design.md §8.2 — replaces the need for MongoDB),
# plain JSON everywhere else.
FlexibleJSON = JSONB().with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_guest: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Per-user preferences: calorie/macro goals, plate diameter, theme.
    preferences: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    meals: Mapped[list["Meal"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Meal(Base):
    __tablename__ = "meals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    total_calories: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    total_protein_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    total_carbs_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    total_fat_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)

    # Which checkpoints produced this row (design.md §9 + §12.1) — so a
    # historical result can always be traced to the model that made it.
    engine: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_versions: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)
    plate_diameter_cm: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    micros: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)

    user: Mapped[User] = relationship(back_populates="meals")
    items: Mapped[list["MealItem"]] = relationship(
        back_populates="meal",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="MealItem.position",
    )


class MealItem(Base):
    __tablename__ = "meal_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    meal_id: Mapped[str] = mapped_column(String(36), ForeignKey("meals.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)

    detected_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    classified_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), default=0)
    estimated_weight_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    estimated_volume_ml: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)

    calories: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    protein_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    carbs_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    fat_g: Mapped[float] = mapped_column(Numeric(10, 2), default=0)

    user_corrected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    weight_estimated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    bbox: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)
    nutrients: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)
    alternatives: Mapped[list[Any]] = mapped_column(FlexibleJSON, default=list)
    nutrition_source: Mapped[str | None] = mapped_column(String(60), nullable=True)

    meal: Mapped[Meal] = relationship(back_populates="items")


class MealDraft(Base):
    """A scanned-but-not-yet-costed plate, waiting on the user's review.

    Short-lived by design: a draft exists only between the scan and the deep
    pass, and a startup sweep clears anything the user abandoned. Nothing here
    is nutrition — that is the whole point of the two-phase split.

    `payload` holds the plate summary and the per-item scan metadata. The region
    masks cannot go in it (numpy arrays don't serialise to JSON, and a list of
    640×640 booleans would be absurd if they did), so they live beside the image
    as a PNG label map.
    """

    __tablename__ = "meal_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    plate_diameter_cm: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    engine: Mapped[str | None] = mapped_column(String(40), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)


class NutritionCache(Base):
    __tablename__ = "nutrition_cache"

    food_label: Mapped[str] = mapped_column(String(160), primary_key=True)
    source: Mapped[str] = mapped_column(String(60))
    per_100g_json: Mapped[dict[str, Any]] = mapped_column(FlexibleJSON, default=dict)
    cached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=not settings.is_sqlite,
    connect_args=_connect_args,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(func.lower(User.email) == email.strip().lower()))


def get_user(session: Session, user_id: str) -> User | None:
    return session.get(User, user_id)


def get_meal(session: Session, meal_id: str) -> Meal | None:
    return session.get(Meal, meal_id)


def get_draft(session: Session, draft_id: str) -> MealDraft | None:
    return session.get(MealDraft, draft_id)


def stale_draft_ids(session: Session, older_than: datetime) -> list[str]:
    """Drafts abandoned before `older_than` — reviewed plates that never got analysed.

    Returns ids rather than rows so the caller can delete the images too; a
    draft's real footprint is the three files beside it, not the row.
    """
    return list(session.scalars(select(MealDraft.id).where(MealDraft.created_at < older_than)))


def list_meals(session: Session, user_id: str, limit: int = 50, offset: int = 0) -> list[Meal]:
    stmt = (
        select(Meal)
        .where(Meal.user_id == user_id)
        .order_by(Meal.captured_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.scalars(stmt))


def count_meals(session: Session, user_id: str) -> int:
    return int(session.scalar(select(func.count(Meal.id)).where(Meal.user_id == user_id)) or 0)


def cached_nutrition(session: Session, label: str) -> NutritionCache | None:
    return session.get(NutritionCache, label)


def put_cached_nutrition(session: Session, label: str, source: str, per_100g: dict[str, Any]) -> None:
    row = session.get(NutritionCache, label)
    if row is None:
        session.add(
            NutritionCache(food_label=label, source=source, per_100g_json=per_100g, cached_at=utcnow())
        )
    else:
        row.source = source
        row.per_100g_json = per_100g
        row.cached_at = utcnow()
