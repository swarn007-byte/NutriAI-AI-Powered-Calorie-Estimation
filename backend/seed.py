"""The welcome meal every new account starts with.

A brand-new user has nothing to look at: the history list, the day strip and the
statistics screen are all empty, so none of them can show what the app actually
produces. Seeding one real-looking meal makes the first launch legible.

The numbers are not invented — they come from the same composition table and the
same scaling function the pipeline uses, so the seeded totals add up exactly the
way an analysed plate's would. The image is the `thali` sample already shipped
for the demo pages.
"""

from __future__ import annotations

import logging
import shutil
from datetime import timedelta

from sqlalchemy.orm import Session

import nutrition
from config import FRONTEND_DIR, UPLOAD_DIR, settings
from db import Meal, MealItem, User, new_id, utcnow
from imaging import encode_jpeg, read_image

log = logging.getLogger("nutriai.seed")

SAMPLE_IMAGE = FRONTEND_DIR / "samples" / "thali.jpg"

# (label, grams, confidence, normalised bbox) — laid out to match where each
# item actually sits in the thali photo, so the results overlay lines up.
WELCOME_ITEMS = (
    ("plain_rice", 150.0, 0.91, (0.30, 0.09, 0.40, 0.36)),
    ("dal_tadka", 120.0, 0.88, (0.09, 0.44, 0.33, 0.34)),
    ("mixed_veg_curry", 110.0, 0.83, (0.56, 0.45, 0.34, 0.33)),
    ("roti_chapati", 90.0, 0.86, (0.30, 0.66, 0.36, 0.28)),
)

# Yesterday, not today: a new user's daily ring should read a truthful 0, and the
# meal is still the newest thing in history either way.
WELCOME_AGE = timedelta(days=1)


def _write_images(meal_id: str) -> bool:
    """Copy the sample in as this meal's image and derive its thumbnail."""
    if not SAMPLE_IMAGE.is_file():
        log.warning("Welcome meal skipped — sample image missing at %s", SAMPLE_IMAGE)
        return False
    full = UPLOAD_DIR / f"{meal_id}.jpg"
    shutil.copyfile(SAMPLE_IMAGE, full)
    thumb = UPLOAD_DIR / f"{meal_id}_thumb.jpg"
    thumb.write_bytes(encode_jpeg(read_image(full.read_bytes()), quality=78, max_dim=420))
    return True


def _build_items(meal_id: str) -> list[MealItem]:
    items = []
    for position, (label, grams, confidence, bbox) in enumerate(WELCOME_ITEMS):
        nutrients = nutrition.scale_nutrients(nutrition.per_100g(label), grams)
        density = nutrition.density_for(label)
        x, y, w, h = bbox
        items.append(
            MealItem(
                id=new_id(),
                meal_id=meal_id,
                position=position,
                detected_label="food",
                classified_label=label,
                confidence=confidence,
                estimated_weight_g=grams,
                estimated_volume_ml=round(grams / max(density, 0.05), 1),
                calories=nutrients["calories"],
                protein_g=nutrients["protein_g"],
                carbs_g=nutrients["carbs_g"],
                fat_g=nutrients["fat_g"],
                low_confidence=False,
                weight_estimated=False,
                bbox={"x": x, "y": y, "w": w, "h": h},
                nutrients={
                    **nutrients,
                    "_geometry": {
                        "area_cm2": round(grams / max(density, 0.05) / 2.4, 2),
                        "mean_height_cm": 2.4,
                        "peak_height_cm": 3.6,
                        "density_g_per_ml": round(density, 3),
                        "method": "welcome-sample",
                        "position": position,
                        "coarse_confidence": confidence,
                    },
                },
                alternatives=[],
                nutrition_source="Indian Food Composition Table",
            )
        )
    return items


def seed_welcome_meal(session: Session, user: User) -> Meal | None:
    """Give `user` one pre-analysed meal. Never raises — a failure just skips it.

    Signup and guest creation must not depend on this working: a missing sample
    file or an unwritable upload directory is a reason to have no demo meal, not
    a reason to refuse the account.
    """
    if not settings.seed_welcome_meal:
        return None
    meal_id = new_id()
    try:
        if not _write_images(meal_id):
            return None

        # Savepoint, not the outer transaction: a failure here has to discard the
        # half-written meal *without* taking the caller's new user row with it.
        with session.begin_nested():
            items = _build_items(meal_id)
            totals = nutrition.sum_nutrients([dict(item.nutrients) for item in items])
            meal = Meal(
                id=meal_id,
                user_id=user.id,
                image_url=f"/media/{meal_id}",
                image_width=900,
                image_height=900,
                captured_at=utcnow() - WELCOME_AGE,
                total_calories=totals["calories"],
                total_protein_g=totals["protein_g"],
                total_carbs_g=totals["carbs_g"],
                total_fat_g=totals["fat_g"],
                micros={key: totals[key] for key in nutrition.MICRO_KEYS if key in totals},
                engine="sample",
                model_versions={"_seeded": True},
                plate_diameter_cm=settings.default_plate_diameter_cm,
                notes="A sample meal so you can see how Nutri-AI reads a plate.",
            )
            session.add(meal)
            for item in items:
                session.add(item)
        session.flush()
        return meal
    except Exception as exc:  # pragma: no cover - must never block account creation
        log.warning("Welcome meal skipped (%s)", exc)
        for path in (UPLOAD_DIR / f"{meal_id}.jpg", UPLOAD_DIR / f"{meal_id}_thumb.jpg"):
            path.unlink(missing_ok=True)
        return None
