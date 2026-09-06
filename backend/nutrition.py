"""Stage 5 — Nutrition lookup (design.md §7.5).

Resolution order for a dish label:

1. `nutrition_cache` table (design.md §8.2)
2. Bundled Indian Food Composition Table (offline, always available)
3. USDA FoodData Central API, when `USDA_API_KEY` is configured
4. A conservative category average, flagged as `estimated`

Every value in `COMPOSITION` is per 100 g of the prepared dish. The arithmetic
that turns those into a per-item result is a pure function (`scale_nutrients`)
precisely because design.md §14.3 calls out nutrition-scaling math as the kind
of thing a solo builder gets subtly wrong with no reviewer to catch it.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from sqlalchemy.orm import Session

from config import settings

log = logging.getLogger("nutriai.nutrition")

# Micronutrients tracked end-to-end, with the reference intake used for %DV.
# Values follow ICMR-NIN 2020 RDA for an adult, rounded.
MICRO_KEYS: tuple[str, ...] = (
    "fiber_g",
    "sugar_g",
    "sodium_mg",
    "calcium_mg",
    "iron_mg",
    "potassium_mg",
    "magnesium_mg",
    "zinc_mg",
    "vitamin_a_mcg",
    "vitamin_c_mg",
    "vitamin_d_mcg",
    "vitamin_b12_mcg",
    "folate_mcg",
)

DAILY_VALUES: dict[str, float] = {
    "fiber_g": 30.0,
    "sugar_g": 50.0,
    "sodium_mg": 2000.0,
    "calcium_mg": 1000.0,
    "iron_mg": 18.0,
    "potassium_mg": 3500.0,
    "magnesium_mg": 400.0,
    "zinc_mg": 12.0,
    "vitamin_a_mcg": 900.0,
    "vitamin_c_mg": 80.0,
    "vitamin_d_mcg": 15.0,
    "vitamin_b12_mcg": 2.4,
    "folate_mcg": 300.0,
}

MICRO_LABELS: dict[str, str] = {
    "fiber_g": "Fibre",
    "sugar_g": "Sugars",
    "sodium_mg": "Sodium",
    "calcium_mg": "Calcium",
    "iron_mg": "Iron",
    "potassium_mg": "Potassium",
    "magnesium_mg": "Magnesium",
    "zinc_mg": "Zinc",
    "vitamin_a_mcg": "Vitamin A",
    "vitamin_c_mg": "Vitamin C",
    "vitamin_d_mcg": "Vitamin D",
    "vitamin_b12_mcg": "Vitamin B12",
    "folate_mcg": "Folate",
}

MICRO_UNITS: dict[str, str] = {
    "fiber_g": "g",
    "sugar_g": "g",
    "sodium_mg": "mg",
    "calcium_mg": "mg",
    "iron_mg": "mg",
    "potassium_mg": "mg",
    "magnesium_mg": "mg",
    "zinc_mg": "mg",
    "vitamin_a_mcg": "µg",
    "vitamin_c_mg": "mg",
    "vitamin_d_mcg": "µg",
    "vitamin_b12_mcg": "µg",
    "folate_mcg": "µg",
}


def _entry(
    name: str,
    category: str,
    kcal: float,
    protein: float,
    carbs: float,
    fat: float,
    density: float,
    **micros: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "display_name": name,
        "category": category,
        "kcal": kcal,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "density_g_per_ml": density,
    }
    for key in MICRO_KEYS:
        row[key] = float(micros.get(key, 0.0))
    return row


# --------------------------------------------------------------------------
# Indian Food Composition Table + common items (per 100 g, prepared)
# --------------------------------------------------------------------------
COMPOSITION: dict[str, dict[str, Any]] = {
    "paneer_butter_masala": _entry(
        "Paneer Butter Masala", "curry", 280, 11.0, 9.0, 23.0, 1.02,
        fiber_g=1.8, sugar_g=4.2, sodium_mg=430, calcium_mg=290, iron_mg=1.1,
        potassium_mg=210, magnesium_mg=26, zinc_mg=1.2, vitamin_a_mcg=180,
        vitamin_c_mg=6, vitamin_d_mcg=0.2, vitamin_b12_mcg=0.6, folate_mcg=18,
    ),
    "palak_paneer": _entry(
        "Palak Paneer", "curry", 190, 10.5, 7.0, 13.5, 1.02,
        fiber_g=2.6, sugar_g=2.4, sodium_mg=400, calcium_mg=320, iron_mg=2.4,
        potassium_mg=430, magnesium_mg=54, zinc_mg=1.1, vitamin_a_mcg=470,
        vitamin_c_mg=18, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.5, folate_mcg=82,
    ),
    "dal_tadka": _entry(
        "Dal Tadka", "dal", 128, 6.4, 15.5, 4.4, 1.03,
        fiber_g=3.4, sugar_g=1.6, sodium_mg=330, calcium_mg=32, iron_mg=1.7,
        potassium_mg=280, magnesium_mg=38, zinc_mg=0.9, vitamin_a_mcg=32,
        vitamin_c_mg=3, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=64,
    ),
    "sambhar": _entry(
        "Sambhar", "dal", 84, 3.6, 11.2, 2.6, 1.02,
        fiber_g=2.9, sugar_g=2.4, sodium_mg=360, calcium_mg=38, iron_mg=1.2,
        potassium_mg=300, magnesium_mg=30, zinc_mg=0.6, vitamin_a_mcg=58,
        vitamin_c_mg=11, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=42,
    ),
    "chole_masala": _entry(
        "Chole Masala", "curry", 164, 7.4, 21.0, 5.6, 1.01,
        fiber_g=6.2, sugar_g=3.1, sodium_mg=410, calcium_mg=58, iron_mg=2.4,
        potassium_mg=350, magnesium_mg=48, zinc_mg=1.3, vitamin_a_mcg=26,
        vitamin_c_mg=8, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=96,
    ),
    "rajma_masala": _entry(
        "Rajma Masala", "curry", 148, 7.0, 20.4, 4.2, 1.02,
        fiber_g=6.8, sugar_g=2.4, sodium_mg=380, calcium_mg=52, iron_mg=2.2,
        potassium_mg=410, magnesium_mg=44, zinc_mg=1.1, vitamin_a_mcg=22,
        vitamin_c_mg=5, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=88,
    ),
    "aloo_gobi": _entry(
        "Aloo Gobi", "dry_sabzi", 118, 3.0, 14.6, 5.6, 0.78,
        fiber_g=3.2, sugar_g=2.8, sodium_mg=320, calcium_mg=34, iron_mg=1.0,
        potassium_mg=380, magnesium_mg=24, zinc_mg=0.5, vitamin_a_mcg=36,
        vitamin_c_mg=28, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=46,
    ),
    "bhindi_masala": _entry(
        "Bhindi Masala", "dry_sabzi", 112, 2.4, 9.8, 7.2, 0.72,
        fiber_g=3.6, sugar_g=2.2, sodium_mg=300, calcium_mg=78, iron_mg=0.8,
        potassium_mg=300, magnesium_mg=42, zinc_mg=0.6, vitamin_a_mcg=42,
        vitamin_c_mg=17, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=54,
    ),
    "mixed_veg_curry": _entry(
        "Mixed Vegetable Curry", "curry", 108, 3.2, 12.4, 5.0, 0.95,
        fiber_g=3.4, sugar_g=3.6, sodium_mg=340, calcium_mg=48, iron_mg=1.1,
        potassium_mg=330, magnesium_mg=28, zinc_mg=0.6, vitamin_a_mcg=210,
        vitamin_c_mg=22, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=44,
    ),
    "butter_chicken": _entry(
        "Butter Chicken", "curry", 240, 15.6, 6.2, 17.4, 1.03,
        fiber_g=1.1, sugar_g=3.8, sodium_mg=480, calcium_mg=68, iron_mg=1.3,
        potassium_mg=260, magnesium_mg=24, zinc_mg=1.4, vitamin_a_mcg=150,
        vitamin_c_mg=5, vitamin_d_mcg=0.3, vitamin_b12_mcg=0.6, folate_mcg=14,
    ),
    "chicken_curry": _entry(
        "Chicken Curry", "curry", 186, 16.4, 5.4, 11.2, 1.02,
        fiber_g=1.2, sugar_g=2.2, sodium_mg=440, calcium_mg=34, iron_mg=1.5,
        potassium_mg=280, magnesium_mg=26, zinc_mg=1.6, vitamin_a_mcg=54,
        vitamin_c_mg=6, vitamin_d_mcg=0.2, vitamin_b12_mcg=0.5, folate_mcg=12,
    ),
    "fish_curry": _entry(
        "Fish Curry", "curry", 152, 15.0, 5.0, 8.0, 1.02,
        fiber_g=1.0, sugar_g=2.0, sodium_mg=460, calcium_mg=52, iron_mg=1.2,
        potassium_mg=310, magnesium_mg=32, zinc_mg=0.9, vitamin_a_mcg=48,
        vitamin_c_mg=7, vitamin_d_mcg=2.6, vitamin_b12_mcg=1.8, folate_mcg=16,
    ),
    "egg_curry": _entry(
        "Egg Curry", "curry", 172, 9.8, 6.4, 12.2, 1.01,
        fiber_g=1.3, sugar_g=2.6, sodium_mg=420, calcium_mg=58, iron_mg=1.8,
        potassium_mg=240, magnesium_mg=22, zinc_mg=1.0, vitamin_a_mcg=126,
        vitamin_c_mg=6, vitamin_d_mcg=1.1, vitamin_b12_mcg=0.7, folate_mcg=32,
    ),
    "plain_rice": _entry(
        "Steamed Rice", "rice", 130, 2.7, 28.2, 0.3, 0.85,
        fiber_g=0.4, sugar_g=0.1, sodium_mg=2, calcium_mg=10, iron_mg=0.2,
        potassium_mg=36, magnesium_mg=12, zinc_mg=0.5, vitamin_a_mcg=0,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=3,
    ),
    "jeera_rice": _entry(
        "Jeera Rice", "rice", 168, 3.0, 28.8, 4.6, 0.82,
        fiber_g=0.8, sugar_g=0.2, sodium_mg=210, calcium_mg=18, iron_mg=0.6,
        potassium_mg=52, magnesium_mg=16, zinc_mg=0.5, vitamin_a_mcg=8,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=5,
    ),
    "veg_biryani": _entry(
        "Vegetable Biryani", "rice", 178, 4.2, 27.4, 5.8, 0.80,
        fiber_g=1.9, sugar_g=1.6, sodium_mg=360, calcium_mg=32, iron_mg=1.0,
        potassium_mg=160, magnesium_mg=22, zinc_mg=0.7, vitamin_a_mcg=96,
        vitamin_c_mg=6, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=22,
    ),
    "chicken_biryani": _entry(
        "Chicken Biryani", "rice", 198, 9.4, 25.6, 6.8, 0.83,
        fiber_g=1.4, sugar_g=1.2, sodium_mg=420, calcium_mg=28, iron_mg=1.2,
        potassium_mg=190, magnesium_mg=24, zinc_mg=1.1, vitamin_a_mcg=54,
        vitamin_c_mg=4, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.4, folate_mcg=18,
    ),
    "roti_chapati": _entry(
        "Roti / Chapati", "bread", 264, 8.4, 50.2, 3.4, 0.52,
        fiber_g=6.4, sugar_g=1.2, sodium_mg=180, calcium_mg=36, iron_mg=2.6,
        potassium_mg=180, magnesium_mg=82, zinc_mg=1.4, vitamin_a_mcg=0,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=24,
    ),
    "naan": _entry(
        "Naan", "bread", 310, 8.8, 51.0, 7.6, 0.48,
        fiber_g=2.2, sugar_g=3.4, sodium_mg=420, calcium_mg=64, iron_mg=2.2,
        potassium_mg=120, magnesium_mg=26, zinc_mg=0.8, vitamin_a_mcg=18,
        vitamin_c_mg=0, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.1, folate_mcg=52,
    ),
    "paratha": _entry(
        "Paratha", "bread", 326, 7.2, 44.6, 13.4, 0.55,
        fiber_g=4.8, sugar_g=1.4, sodium_mg=300, calcium_mg=34, iron_mg=2.4,
        potassium_mg=160, magnesium_mg=68, zinc_mg=1.2, vitamin_a_mcg=22,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=20,
    ),
    "poori": _entry(
        "Poori", "bread", 372, 6.8, 42.0, 19.8, 0.42,
        fiber_g=3.6, sugar_g=1.0, sodium_mg=220, calcium_mg=26, iron_mg=2.0,
        potassium_mg=140, magnesium_mg=58, zinc_mg=1.0, vitamin_a_mcg=0,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=18,
    ),
    "idli": _entry(
        "Idli", "steamed", 132, 4.2, 26.4, 0.6, 0.62,
        fiber_g=1.4, sugar_g=0.4, sodium_mg=190, calcium_mg=16, iron_mg=0.8,
        potassium_mg=70, magnesium_mg=22, zinc_mg=0.6, vitamin_a_mcg=0,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=14,
    ),
    "dosa": _entry(
        "Dosa", "bread", 216, 4.8, 32.4, 7.4, 0.50,
        fiber_g=1.6, sugar_g=0.6, sodium_mg=240, calcium_mg=18, iron_mg=1.0,
        potassium_mg=88, magnesium_mg=26, zinc_mg=0.7, vitamin_a_mcg=6,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=16,
    ),
    "medu_vada": _entry(
        "Medu Vada", "fried", 328, 8.2, 30.6, 18.6, 0.58,
        fiber_g=4.2, sugar_g=0.8, sodium_mg=330, calcium_mg=42, iron_mg=1.9,
        potassium_mg=210, magnesium_mg=48, zinc_mg=1.2, vitamin_a_mcg=8,
        vitamin_c_mg=2, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=38,
    ),
    "samosa": _entry(
        "Samosa", "fried", 308, 5.2, 32.4, 17.2, 0.60,
        fiber_g=2.8, sugar_g=1.6, sodium_mg=380, calcium_mg=24, iron_mg=1.3,
        potassium_mg=250, magnesium_mg=26, zinc_mg=0.6, vitamin_a_mcg=14,
        vitamin_c_mg=6, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=22,
    ),
    "pav_bhaji": _entry(
        "Pav Bhaji", "curry", 158, 3.6, 18.2, 7.8, 0.92,
        fiber_g=3.8, sugar_g=3.2, sodium_mg=470, calcium_mg=42, iron_mg=1.4,
        potassium_mg=340, magnesium_mg=26, zinc_mg=0.6, vitamin_a_mcg=180,
        vitamin_c_mg=32, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=38,
    ),
    "upma": _entry(
        "Upma", "grain", 148, 3.4, 22.6, 5.0, 0.72,
        fiber_g=1.8, sugar_g=1.2, sodium_mg=310, calcium_mg=22, iron_mg=1.0,
        potassium_mg=120, magnesium_mg=20, zinc_mg=0.5, vitamin_a_mcg=32,
        vitamin_c_mg=4, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=18,
    ),
    "poha": _entry(
        "Poha", "grain", 142, 2.6, 24.8, 3.8, 0.62,
        fiber_g=1.4, sugar_g=1.6, sodium_mg=290, calcium_mg=18, iron_mg=2.8,
        potassium_mg=130, magnesium_mg=18, zinc_mg=0.5, vitamin_a_mcg=24,
        vitamin_c_mg=6, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=14,
    ),
    "curd_yogurt": _entry(
        "Curd / Yogurt", "dairy", 62, 3.4, 4.8, 3.2, 1.03,
        fiber_g=0.0, sugar_g=4.6, sodium_mg=46, calcium_mg=125, iron_mg=0.1,
        potassium_mg=180, magnesium_mg=14, zinc_mg=0.6, vitamin_a_mcg=28,
        vitamin_c_mg=1, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.4, folate_mcg=8,
    ),
    "raita": _entry(
        "Raita", "dairy", 68, 3.0, 5.6, 3.4, 1.01,
        fiber_g=0.6, sugar_g=4.2, sodium_mg=210, calcium_mg=110, iron_mg=0.3,
        potassium_mg=200, magnesium_mg=16, zinc_mg=0.5, vitamin_a_mcg=30,
        vitamin_c_mg=4, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.3, folate_mcg=12,
    ),
    "coconut_chutney": _entry(
        "Coconut Chutney", "condiment", 184, 3.0, 8.4, 15.6, 0.94,
        fiber_g=4.6, sugar_g=2.4, sodium_mg=280, calcium_mg=24, iron_mg=1.4,
        potassium_mg=250, magnesium_mg=38, zinc_mg=0.7, vitamin_a_mcg=4,
        vitamin_c_mg=3, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=16,
    ),
    "green_salad": _entry(
        "Green Salad", "salad", 32, 1.4, 5.4, 0.4, 0.38,
        fiber_g=2.2, sugar_g=2.8, sodium_mg=18, calcium_mg=32, iron_mg=0.7,
        potassium_mg=260, magnesium_mg=18, zinc_mg=0.3, vitamin_a_mcg=320,
        vitamin_c_mg=24, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=42,
    ),
    "papad": _entry(
        "Papad", "fried", 372, 18.4, 52.0, 9.6, 0.30,
        fiber_g=8.2, sugar_g=1.0, sodium_mg=1200, calcium_mg=64, iron_mg=3.6,
        potassium_mg=540, magnesium_mg=96, zinc_mg=2.0, vitamin_a_mcg=0,
        vitamin_c_mg=0, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=44,
    ),
    "gulab_jamun": _entry(
        "Gulab Jamun", "dessert", 336, 4.6, 48.2, 14.2, 1.05,
        fiber_g=0.4, sugar_g=42.0, sodium_mg=86, calcium_mg=118, iron_mg=0.6,
        potassium_mg=110, magnesium_mg=14, zinc_mg=0.5, vitamin_a_mcg=64,
        vitamin_c_mg=0, vitamin_d_mcg=0.2, vitamin_b12_mcg=0.3, folate_mcg=8,
    ),
    "kheer": _entry(
        "Kheer", "dessert", 148, 3.8, 22.4, 4.8, 1.06,
        fiber_g=0.4, sugar_g=18.0, sodium_mg=64, calcium_mg=132, iron_mg=0.3,
        potassium_mg=170, magnesium_mg=16, zinc_mg=0.5, vitamin_a_mcg=52,
        vitamin_c_mg=1, vitamin_d_mcg=0.3, vitamin_b12_mcg=0.4, folate_mcg=10,
    ),
    "boiled_egg": _entry(
        "Boiled Egg", "protein", 155, 12.6, 1.1, 10.6, 1.03,
        fiber_g=0.0, sugar_g=1.1, sodium_mg=124, calcium_mg=50, iron_mg=1.2,
        potassium_mg=126, magnesium_mg=10, zinc_mg=1.1, vitamin_a_mcg=140,
        vitamin_c_mg=0, vitamin_d_mcg=2.0, vitamin_b12_mcg=1.1, folate_mcg=44,
    ),
    "grilled_chicken": _entry(
        "Grilled Chicken", "protein", 195, 29.6, 0.0, 7.8, 1.05,
        fiber_g=0.0, sugar_g=0.0, sodium_mg=320, calcium_mg=14, iron_mg=1.0,
        potassium_mg=280, magnesium_mg=28, zinc_mg=1.8, vitamin_a_mcg=12,
        vitamin_c_mg=0, vitamin_d_mcg=0.2, vitamin_b12_mcg=0.4, folate_mcg=6,
    ),
    "pasta_red_sauce": _entry(
        "Pasta in Red Sauce", "grain", 168, 5.6, 26.4, 4.6, 0.72,
        fiber_g=2.4, sugar_g=4.2, sodium_mg=380, calcium_mg=30, iron_mg=1.2,
        potassium_mg=240, magnesium_mg=26, zinc_mg=0.7, vitamin_a_mcg=48,
        vitamin_c_mg=9, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=42,
    ),
    "pizza_slice": _entry(
        "Pizza", "fast_food", 266, 11.0, 33.0, 10.0, 0.55,
        fiber_g=2.3, sugar_g=3.6, sodium_mg=598, calcium_mg=188, iron_mg=2.5,
        potassium_mg=184, magnesium_mg=24, zinc_mg=1.4, vitamin_a_mcg=68,
        vitamin_c_mg=1, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.4, folate_mcg=58,
    ),
    "french_fries": _entry(
        "French Fries", "fried", 312, 3.4, 41.4, 15.0, 0.45,
        fiber_g=3.8, sugar_g=0.3, sodium_mg=210, calcium_mg=18, iron_mg=0.8,
        potassium_mg=580, magnesium_mg=28, zinc_mg=0.4, vitamin_a_mcg=0,
        vitamin_c_mg=10, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=28,
    ),
    "banana": _entry(
        "Banana", "fruit", 89, 1.1, 22.8, 0.3, 0.94,
        fiber_g=2.6, sugar_g=12.2, sodium_mg=1, calcium_mg=5, iron_mg=0.3,
        potassium_mg=358, magnesium_mg=27, zinc_mg=0.2, vitamin_a_mcg=3,
        vitamin_c_mg=9, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=20,
    ),
    "apple": _entry(
        "Apple", "fruit", 52, 0.3, 13.8, 0.2, 0.85,
        fiber_g=2.4, sugar_g=10.4, sodium_mg=1, calcium_mg=6, iron_mg=0.1,
        potassium_mg=107, magnesium_mg=5, zinc_mg=0.0, vitamin_a_mcg=3,
        vitamin_c_mg=5, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=3,
    ),
    "momos": _entry(
        "Momos (Steamed)", "steamed", 168, 6.4, 24.8, 4.6, 0.58,
        fiber_g=1.8, sugar_g=1.4, sodium_mg=360, calcium_mg=22, iron_mg=1.2,
        potassium_mg=140, magnesium_mg=20, zinc_mg=0.7, vitamin_a_mcg=18,
        vitamin_c_mg=4, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=22,
    ),
    "green_chutney": _entry(
        "Green Chutney (Mint/Coriander)", "condiment", 68, 1.8, 6.2, 4.0, 0.96,
        fiber_g=2.4, sugar_g=3.2, sodium_mg=320, calcium_mg=52, iron_mg=1.4,
        potassium_mg=180, magnesium_mg=22, zinc_mg=0.4, vitamin_a_mcg=120,
        vitamin_c_mg=12, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=28,
    ),
    "tamarind_chutney": _entry(
        "Tamarind Chutney", "condiment", 228, 1.2, 52.0, 0.4, 1.02,
        fiber_g=2.8, sugar_g=44.0, sodium_mg=280, calcium_mg=34, iron_mg=1.8,
        potassium_mg=290, magnesium_mg=26, zinc_mg=0.3, vitamin_a_mcg=8,
        vitamin_c_mg=3, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=14,
    ),
    "manchurian": _entry(
        "Manchurian", "curry", 172, 4.8, 18.0, 8.6, 0.98,
        fiber_g=2.2, sugar_g=4.8, sodium_mg=520, calcium_mg=32, iron_mg=1.0,
        potassium_mg=160, magnesium_mg=18, zinc_mg=0.5, vitamin_a_mcg=24,
        vitamin_c_mg=8, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=16,
    ),
    "hakka_noodles": _entry(
        "Hakka Noodles", "grain", 182, 5.2, 28.0, 5.8, 0.52,
        fiber_g=2.4, sugar_g=2.6, sodium_mg=480, calcium_mg=24, iron_mg=1.2,
        potassium_mg=120, magnesium_mg=20, zinc_mg=0.6, vitamin_a_mcg=32,
        vitamin_c_mg=6, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=18,
    ),
    "fried_rice": _entry(
        "Fried Rice", "rice", 192, 4.6, 30.0, 5.6, 0.68,
        fiber_g=1.6, sugar_g=1.4, sodium_mg=460, calcium_mg=22, iron_mg=0.8,
        potassium_mg=100, magnesium_mg=18, zinc_mg=0.5, vitamin_a_mcg=28,
        vitamin_c_mg=4, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=14,
    ),
    "chole_bhature": _entry(
        "Chole Bhature", "fast_food", 310, 7.8, 34.0, 15.2, 0.72,
        fiber_g=5.4, sugar_g=3.6, sodium_mg=440, calcium_mg=54, iron_mg=2.2,
        potassium_mg=280, magnesium_mg=42, zinc_mg=1.0, vitamin_a_mcg=22,
        vitamin_c_mg=6, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=72,
    ),
    "paneer_tikka": _entry(
        "Paneer Tikka", "protein", 232, 14.8, 6.0, 16.8, 0.92,
        fiber_g=1.2, sugar_g=2.8, sodium_mg=420, calcium_mg=260, iron_mg=0.8,
        potassium_mg=180, magnesium_mg=22, zinc_mg=0.9, vitamin_a_mcg=120,
        vitamin_c_mg=8, vitamin_d_mcg=0.2, vitamin_b12_mcg=0.4, folate_mcg=12,
    ),
    "tandoori_chicken": _entry(
        "Tandoori Chicken", "protein", 210, 26.0, 4.0, 10.0, 1.04,
        fiber_g=0.8, sugar_g=2.0, sodium_mg=460, calcium_mg=22, iron_mg=1.2,
        potassium_mg=260, magnesium_mg=26, zinc_mg=1.4, vitamin_a_mcg=68,
        vitamin_c_mg=6, vitamin_d_mcg=0.3, vitamin_b12_mcg=0.5, folate_mcg=10,
    ),
    "dal_makhani": _entry(
        "Dal Makhani", "dal", 182, 7.2, 18.0, 8.8, 1.02,
        fiber_g=5.4, sugar_g=3.0, sodium_mg=380, calcium_mg=42, iron_mg=2.0,
        potassium_mg=310, magnesium_mg=42, zinc_mg=1.0, vitamin_a_mcg=38,
        vitamin_c_mg=4, vitamin_d_mcg=0.1, vitamin_b12_mcg=0.0, folate_mcg=68,
    ),
    "vada_pav": _entry(
        "Vada Pav", "fast_food", 290, 6.8, 36.0, 13.2, 0.55,
        fiber_g=3.2, sugar_g=3.0, sodium_mg=420, calcium_mg=28, iron_mg=1.4,
        potassium_mg=160, magnesium_mg=24, zinc_mg=0.6, vitamin_a_mcg=12,
        vitamin_c_mg=4, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=18,
    ),
    "masala_dosa": _entry(
        "Masala Dosa", "bread", 248, 5.0, 34.0, 10.2, 0.52,
        fiber_g=2.0, sugar_g=1.0, sodium_mg=360, calcium_mg=22, iron_mg=1.2,
        potassium_mg=110, magnesium_mg=30, zinc_mg=0.8, vitamin_a_mcg=14,
        vitamin_c_mg=4, vitamin_d_mcg=0.0, vitamin_b12_mcg=0.0, folate_mcg=20,
    ),
}

# Conservative per-category averages — used when a label resolves to nothing
# else, so the pipeline degrades instead of returning zeros (design.md §12.2).
CATEGORY_FALLBACK: dict[str, dict[str, Any]] = {
    "curry": _entry("Mixed Curry", "curry", 165, 6.0, 12.0, 10.0, 1.0, fiber_g=2.6, sodium_mg=400),
    "dal": _entry("Lentil Dish", "dal", 120, 6.0, 15.0, 4.0, 1.02, fiber_g=3.2, sodium_mg=340),
    "rice": _entry("Rice Dish", "rice", 150, 3.2, 28.0, 3.0, 0.83, fiber_g=0.8, sodium_mg=120),
    "bread": _entry("Flatbread", "bread", 285, 8.0, 48.0, 7.0, 0.50, fiber_g=4.0, sodium_mg=260),
    "dry_sabzi": _entry("Vegetable Sabzi", "dry_sabzi", 115, 3.0, 13.0, 6.0, 0.75, fiber_g=3.2),
    "salad": _entry("Salad", "salad", 40, 1.5, 6.0, 0.8, 0.40, fiber_g=2.2, vitamin_c_mg=20),
    "dairy": _entry("Dairy Side", "dairy", 65, 3.2, 5.0, 3.3, 1.02, calcium_mg=118),
    "fried": _entry("Fried Snack", "fried", 320, 6.0, 36.0, 16.0, 0.55, sodium_mg=340),
    "dessert": _entry("Indian Dessert", "dessert", 260, 4.0, 38.0, 10.0, 1.04, sugar_g=28),
    "protein": _entry("Protein Portion", "protein", 180, 22.0, 1.0, 9.0, 1.04),
    "fruit": _entry("Fruit", "fruit", 62, 0.7, 15.0, 0.3, 0.88, fiber_g=2.4, vitamin_c_mg=12),
    "grain": _entry("Grain Dish", "grain", 155, 4.0, 25.0, 4.5, 0.72, fiber_g=2.0),
    "condiment": _entry("Condiment", "condiment", 140, 2.4, 9.0, 10.0, 0.95),
    "fast_food": _entry("Fast Food", "fast_food", 270, 10.0, 32.0, 11.0, 0.58, sodium_mg=560),
    "steamed": _entry("Steamed Item", "steamed", 145, 4.4, 28.0, 1.2, 0.70, fiber_g=1.4, sodium_mg=200),
    "unknown": _entry("Unrecognized Item", "unknown", 150, 5.0, 18.0, 6.5, 0.85),
}

# Honest generic names for the coarse categories, used when the classifier's
# confidence is too low to assert a dish (design.md §12.2). Naming a plate
# "Vegetable Biryani" at 0.16 confidence is worse than admitting it is a
# yellow lentil dish of some kind, so these read as descriptions, not claims.
COARSE_FALLBACK: dict[str, dict[str, Any]] = {
    "rice_or_bread": _entry("Rice or Bread", "rice", 175, 4.4, 34.0, 3.2, 0.78, fiber_g=1.6, sodium_mg=180),
    "dal_or_yellow": _entry("Lentil or Yellow Dish", "dal", 130, 5.6, 17.0, 4.4, 1.0, fiber_g=3.0, sodium_mg=340),
    "gravy": _entry("Curry or Gravy", "curry", 165, 6.0, 12.0, 10.0, 1.0, fiber_g=2.6, sodium_mg=400),
    "red_dish": _entry("Tomato-based Dish", "curry", 158, 5.4, 13.0, 9.2, 1.0, fiber_g=2.8, sodium_mg=390),
    "brown_dish": _entry("Brown Dish", "curry", 172, 7.0, 14.0, 9.6, 0.95, fiber_g=2.6, sodium_mg=360),
    "vegetable_green": _entry("Green Vegetable", "dry_sabzi", 92, 3.0, 9.0, 4.4, 0.62, fiber_g=3.4, vitamin_c_mg=24),
    "dark_side": _entry("Dark Side Dish", "dry_sabzi", 130, 4.2, 14.0, 6.4, 0.8, fiber_g=3.0, sodium_mg=300),
    "mixed_dish": _entry("Mixed Dish", "unknown", 155, 5.4, 19.0, 6.6, 0.86, fiber_g=2.4, sodium_mg=320),
}

# Coarse detector labels (COCO / YOLO vocabulary) → catalog keys.
DETECTOR_ALIASES: dict[str, str] = {
    "bowl": "dal_tadka",
    "cup": "curd_yogurt",
    "pizza": "pizza_slice",
    "sandwich": "pizza_slice",
    "donut": "gulab_jamun",
    "cake": "kheer",
    "carrot": "green_salad",
    "broccoli": "mixed_veg_curry",
    "hot dog": "pizza_slice",
    "banana": "banana",
    "apple": "apple",
    "orange": "apple",
    "spoon": "curd_yogurt",
    "fork": "green_salad",
    "dining table": "mixed_veg_curry",
}


# Foods a person counts rather than ladles: `(grams per piece, cm² one piece
# covers when plated)`.
#
# The footprint is what lets a scan *guess* a count — four samosas in a basket
# come back from the detector as one blob, so `area ÷ footprint` is the only
# handle on "how many". It is a guess and the UI says so; the count is editable.
#
# A label belongs here only if a serving is naturally enumerated. Every curry,
# dal, rice, grain, dairy, salad, dry_sabzi and condiment is excluded because
# "two curries" describes dishes, not portions. `french_fries` is excluded for
# the same reason (a heap, not a count) and `kheer` because it is spooned.
#
# Piece weights are cooked weights for a home-kitchen portion. The footprints
# assume the piece is lying as it is normally served — a poori flat, a samosa on
# its side — and are the projected area, not the surface area.
PIECE_WEIGHTS: dict[str, tuple[float, float]] = {
    "samosa": (65.0, 42.0),
    "medu_vada": (45.0, 38.0),
    "papad": (12.0, 90.0),
    "poori": (35.0, 78.0),
    "roti_chapati": (45.0, 175.0),
    "naan": (90.0, 240.0),
    "paratha": (75.0, 200.0),
    "dosa": (110.0, 320.0),
    "idli": (50.0, 48.0),
    "gulab_jamun": (40.0, 24.0),
    "boiled_egg": (50.0, 30.0),
    "pizza_slice": (105.0, 150.0),
    "banana": (118.0, 95.0),
    "apple": (182.0, 66.0),
    "momos": (32.0, 28.0),
    "vada_pav": (90.0, 140.0),
}

# Fail at import rather than at request time if a piece weight names a food the
# composition table doesn't have — the two must describe the same 42 labels.
_orphan_pieces = sorted(set(PIECE_WEIGHTS) - set(COMPOSITION))
if _orphan_pieces:  # pragma: no cover - import-time guard
    raise RuntimeError(f"PIECE_WEIGHTS names foods missing from COMPOSITION: {_orphan_pieces}")


def _row_for(label: str) -> dict[str, Any] | None:
    """The best local composition row for a label: exact dish, then coarse group."""
    return COMPOSITION.get(label) or COARSE_FALLBACK.get(label)


def display_name(label: str) -> str:
    row = _row_for(label)
    if row:
        return str(row["display_name"])
    return label.replace("_", " ").title()


def density_for(label: str) -> float:
    """Food density in g/ml — used to turn volume into weight (design.md §7.3)."""
    row = _row_for(label)
    if row:
        return float(row["density_g_per_ml"])
    category = _category_of(label)
    return float(CATEGORY_FALLBACK.get(category, CATEGORY_FALLBACK["unknown"])["density_g_per_ml"])


def _category_of(label: str) -> str:
    row = _row_for(label)
    if row:
        return str(row["category"])
    return "unknown"


def category_of(label: str) -> str:
    """The catalog category for a label, or `"unknown"`.

    Public because the scan response carries it: the review UI groups rows by
    category, and it is the one field there that has nothing to do with
    nutrition numbers.
    """
    return _category_of(label)


def piece_weight_g(label: str) -> float | None:
    """Grams in one piece, or `None` when the food isn't countable.

    `None` is the signal the review UI uses to decide whether to offer a count
    field at all — a curry gets no "how many", because the question is wrong.
    """
    entry = PIECE_WEIGHTS.get(label)
    return float(entry[0]) if entry else None


def piece_footprint_cm2(label: str) -> float | None:
    """Plated area of one piece, or `None` when the food isn't countable."""
    entry = PIECE_WEIGHTS.get(label)
    return float(entry[1]) if entry else None


def nominal_portion_g(label: str) -> float:
    """A defensible weight for an item the photo can't measure.

    Only reached when the user *adds* an item the detector missed: there is no
    region, so no area and no depth, and geometry has nothing to work from.

    Rather than introduce a third portion table to disagree with the other two,
    this composes the constants already in use: the category's mean served depth
    over its nominal footprint gives a volume, density turns that into grams,
    and the category's own served-weight envelope clamps the result. A countable
    food answers with one piece instead, which is a sharper number.
    """
    # Imported here, not at module scope: `depth` pulls in scipy and the whole
    # detection stack, and `nutrition` is otherwise cheap enough to import in a
    # test or a script that has no interest in geometry. sys.modules caches it,
    # so the per-call cost is a dict lookup.
    from depth import NOMINAL_AREA_CM2, SERVING_DEPTH_CM, WEIGHT_BOUNDS

    single = piece_weight_g(label)
    if single is not None:
        return round(single, 1)

    category = _category_of(label)
    depth_cm = SERVING_DEPTH_CM.get(category, SERVING_DEPTH_CM["unknown"])
    grams = depth_cm * NOMINAL_AREA_CM2 * density_for(label)
    low, high = WEIGHT_BOUNDS.get(category, WEIGHT_BOUNDS["unknown"])
    return round(min(max(grams, low), high), 1)


def per_100g(label: str) -> dict[str, Any]:
    row = _row_for(label)
    if row:
        return dict(row)
    return dict(CATEGORY_FALLBACK.get(_category_of(label), CATEGORY_FALLBACK["unknown"]))


def scale_nutrients(base_per_100g: dict[str, Any], grams: float) -> dict[str, float]:
    """Scale a per-100 g composition row to `grams`.

    Pure function, unit-tested (backend/tests/test_nutrition_math.py). Rounding
    is applied once, at the end, so totals never drift from the per-item values
    by more than half a display unit.
    """
    if grams < 0:
        raise ValueError("grams must be non-negative")
    factor = grams / 100.0
    out: dict[str, float] = {
        "calories": round(float(base_per_100g.get("kcal", 0.0)) * factor, 1),
        "protein_g": round(float(base_per_100g.get("protein_g", 0.0)) * factor, 1),
        "carbs_g": round(float(base_per_100g.get("carbs_g", 0.0)) * factor, 1),
        "fat_g": round(float(base_per_100g.get("fat_g", 0.0)) * factor, 1),
    }
    for key in MICRO_KEYS:
        value = float(base_per_100g.get(key, 0.0)) * factor
        out[key] = round(value, 2 if key.endswith("_mcg") or "vitamin" in key else 1)
    return out


def sum_nutrients(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-item nutrient dicts into meal totals (design.md §7.6)."""
    keys = ("calories", "protein_g", "carbs_g", "fat_g", *MICRO_KEYS)
    totals = {key: 0.0 for key in keys}
    for row in rows:
        for key in keys:
            totals[key] += float(row.get(key, 0.0) or 0.0)
    return {key: round(value, 1) for key, value in totals.items()}


def daily_value_percent(nutrients: dict[str, float]) -> dict[str, float]:
    return {
        key: round(min(999.0, (float(nutrients.get(key, 0.0)) / dv) * 100.0), 1)
        for key, dv in DAILY_VALUES.items()
        if dv > 0
    }


def energy_from_macros(protein_g: float, carbs_g: float, fat_g: float) -> float:
    """Atwater factors — used to sanity-check reported calories."""
    return round(protein_g * 4.0 + carbs_g * 4.0 + fat_g * 9.0, 1)


# --------------------------------------------------------------------------
# USDA FoodData Central (design.md §7.5)
# --------------------------------------------------------------------------
_USDA_NUTRIENT_MAP: dict[int, str] = {
    1008: "kcal",
    1003: "protein_g",
    1005: "carbs_g",
    1004: "fat_g",
    1079: "fiber_g",
    2000: "sugar_g",
    1093: "sodium_mg",
    1087: "calcium_mg",
    1089: "iron_mg",
    1092: "potassium_mg",
    1090: "magnesium_mg",
    1095: "zinc_mg",
    1106: "vitamin_a_mcg",
    1162: "vitamin_c_mg",
    1114: "vitamin_d_mcg",
    1178: "vitamin_b12_mcg",
    1177: "folate_mcg",
}


def fetch_usda(label: str) -> dict[str, Any] | None:
    """Look a dish up in USDA FoodData Central. Returns a per-100 g row."""
    if not settings.usda_api_key:
        return None
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the stack
        return None

    query = label.replace("_", " ")
    try:
        with httpx.Client(timeout=settings.usda_timeout_s) as client:
            response = client.get(
                f"{settings.usda_base_url}/foods/search",
                params={
                    "api_key": settings.usda_api_key,
                    "query": query,
                    "pageSize": 1,
                    "dataType": "Survey (FNDDS),SR Legacy,Foundation",
                },
            )
            response.raise_for_status()
            foods = response.json().get("foods") or []
    except Exception as exc:  # network/quota/shape issues must never break a meal
        log.warning("USDA lookup failed for %r: %s", label, exc)
        return None

    if not foods:
        return None

    food = foods[0]
    row: dict[str, Any] = {
        "display_name": food.get("description") or display_name(label),
        "category": _category_of(label),
        "density_g_per_ml": density_for(label),
    }
    for key in ("kcal", "protein_g", "carbs_g", "fat_g", *MICRO_KEYS):
        row[key] = 0.0
    for nutrient in food.get("foodNutrients") or []:
        key = _USDA_NUTRIENT_MAP.get(int(nutrient.get("nutrientId") or 0))
        if key:
            row[key] = float(nutrient.get("value") or 0.0)
    if row["kcal"] <= 0:
        return None
    return row


def resolve_per_100g(session: Session | None, label: str) -> tuple[dict[str, Any], str]:
    """Return `(per_100g_row, source)` for a dish label.

    Cache → bundled table → USDA → category average.
    """
    from db import cached_nutrition, put_cached_nutrition  # local import avoids a cycle

    if session is not None:
        cached = cached_nutrition(session, label)
        if cached and cached.per_100g_json:
            return dict(cached.per_100g_json), f"cache:{cached.source}"

    if label in COMPOSITION:
        row = dict(COMPOSITION[label])
        if session is not None:
            put_cached_nutrition(session, label, "ifct", row)
        return row, "Indian Food Composition Table"

    if label in COARSE_FALLBACK:
        # A coarse group, not a dish — there is nothing meaningful to ask USDA
        # for, and the group average is already the honest answer.
        return dict(COARSE_FALLBACK[label]), "category-average (estimated)"

    usda = fetch_usda(label)
    if usda:
        if session is not None:
            put_cached_nutrition(session, label, "usda", usda)
        return usda, "USDA FoodData Central"

    return per_100g(label), "category-average (estimated)"


def catalog() -> list[dict[str, Any]]:
    """Everything the UI needs to render a correction picker."""
    return sorted(
        (
            {
                "label": key,
                "display_name": row["display_name"],
                "category": row["category"],
                "kcal_per_100g": row["kcal"],
                "protein_g": row["protein_g"],
                "carbs_g": row["carbs_g"],
                "fat_g": row["fat_g"],
                "density_g_per_ml": row["density_g_per_ml"],
                # Non-null means "offer a piece count for this one".
                "piece_weight_g": piece_weight_g(key),
                # The review page re-guesses a count when the user renames a
                # region, and has to land on the same number
                # `pipeline.guess_piece_count` would — otherwise the list shows
                # one count and the analysis quietly uses another.
                "piece_footprint_cm2": piece_footprint_cm2(key),
            }
            for key, row in COMPOSITION.items()
        ),
        key=lambda row: str(row["display_name"]),
    )
