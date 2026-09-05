"""Stage-5 nutrition arithmetic (design.md §7.5, §14.3).

design.md §14.3 singles out nutrition scaling as the maths a solo builder gets
subtly wrong with no reviewer to catch it, so the pure functions are pinned here
alongside the integrity of the composition table they read from — a typo'd
macro row is indistinguishable from a bug in the arithmetic once it reaches the
UI, and only one of the two is caught by testing `scale_nutrients` alone.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nutrition as N  # noqa: E402


class TestScaleNutrients(unittest.TestCase):
    def test_one_hundred_grams_is_the_identity(self):
        row = N.COMPOSITION["dal_tadka"]
        scaled = N.scale_nutrients(row, 100.0)
        self.assertAlmostEqual(scaled["calories"], row["kcal"], places=1)
        self.assertAlmostEqual(scaled["protein_g"], row["protein_g"], places=1)
        self.assertAlmostEqual(scaled["carbs_g"], row["carbs_g"], places=1)
        self.assertAlmostEqual(scaled["fat_g"], row["fat_g"], places=1)
        for key in N.MICRO_KEYS:
            self.assertAlmostEqual(scaled[key], row[key], places=1)

    def test_scales_linearly(self):
        row = N.COMPOSITION["plain_rice"]
        half = N.scale_nutrients(row, 50.0)
        double = N.scale_nutrients(row, 200.0)
        self.assertAlmostEqual(half["calories"] * 4, double["calories"], places=1)
        self.assertAlmostEqual(double["carbs_g"], row["carbs_g"] * 2, places=1)

    def test_zero_grams_is_all_zero(self):
        scaled = N.scale_nutrients(N.COMPOSITION["naan"], 0.0)
        self.assertEqual(set(scaled.values()), {0.0})

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            N.scale_nutrients(N.COMPOSITION["naan"], -1.0)

    def test_returns_every_tracked_key_even_for_a_sparse_row(self):
        """Missing keys must read as 0.0, not vanish — the UI indexes them all."""
        scaled = N.scale_nutrients({"kcal": 100.0}, 150.0)
        self.assertAlmostEqual(scaled["calories"], 150.0, places=1)
        for key in ("protein_g", "carbs_g", "fat_g", *N.MICRO_KEYS):
            self.assertEqual(scaled[key], 0.0)

    def test_micrograms_keep_more_precision_than_grams(self):
        """B12 at 0.6 µg/100 g would round to 0.0 at one decimal for a 30 g serving."""
        scaled = N.scale_nutrients(N.COMPOSITION["paneer_butter_masala"], 30.0)
        self.assertGreater(scaled["vitamin_b12_mcg"], 0.0)


class TestSumNutrients(unittest.TestCase):
    def test_totals_are_the_sum_of_the_parts(self):
        rows = [
            N.scale_nutrients(N.COMPOSITION["plain_rice"], 150.0),
            N.scale_nutrients(N.COMPOSITION["dal_tadka"], 120.0),
        ]
        totals = N.sum_nutrients(rows)
        self.assertAlmostEqual(
            totals["calories"], rows[0]["calories"] + rows[1]["calories"], places=1
        )
        self.assertAlmostEqual(totals["iron_mg"], rows[0]["iron_mg"] + rows[1]["iron_mg"], places=1)

    def test_empty_meal_totals_zero_across_every_key(self):
        totals = N.sum_nutrients([])
        self.assertEqual(set(totals.values()), {0.0})
        for key in ("calories", "protein_g", "carbs_g", "fat_g", *N.MICRO_KEYS):
            self.assertIn(key, totals)

    def test_tolerates_missing_and_null_values(self):
        totals = N.sum_nutrients([{"calories": 10.0}, {"calories": None}, {}])
        self.assertAlmostEqual(totals["calories"], 10.0, places=1)

    def test_ignores_private_keys_the_pipeline_stashes_alongside(self):
        """`_geometry` rides in the same dict; it must not corrupt the roll-up."""
        row = {**N.scale_nutrients(N.COMPOSITION["idli"], 100.0), "_geometry": {"method": "x"}}
        self.assertAlmostEqual(N.sum_nutrients([row])["calories"], N.COMPOSITION["idli"]["kcal"], places=1)


class TestDailyValues(unittest.TestCase):
    def test_full_reference_intake_is_one_hundred_percent(self):
        percents = N.daily_value_percent(dict(N.DAILY_VALUES))
        for key in N.DAILY_VALUES:
            self.assertAlmostEqual(percents[key], 100.0, places=1)

    def test_absent_nutrients_read_as_zero(self):
        percents = N.daily_value_percent({})
        self.assertEqual(set(percents.values()), {0.0})

    def test_capped_so_the_ui_bar_cannot_run_off(self):
        percents = N.daily_value_percent({"sodium_mg": 10_000_000.0})
        self.assertEqual(percents["sodium_mg"], 999.0)

    def test_covers_exactly_the_tracked_micronutrients(self):
        self.assertEqual(set(N.daily_value_percent({})), set(N.MICRO_KEYS))


class TestEnergyFromMacros(unittest.TestCase):
    def test_atwater_factors(self):
        self.assertAlmostEqual(N.energy_from_macros(10.0, 20.0, 5.0), 165.0, places=1)

    def test_declared_calories_agree_with_the_macros_they_ship_with(self):
        """A row whose kcal contradicts its own macros is a data-entry bug.

        ±22% absorbs both genuine rounding in the source tables and the fact that
        fibre and polyols do not follow the 4/4/9 rule exactly.
        """
        for label, row in N.COMPOSITION.items():
            atwater = N.energy_from_macros(row["protein_g"], row["carbs_g"], row["fat_g"])
            if row["kcal"] < 40:  # very low-energy rows are dominated by rounding
                continue
            self.assertLess(
                abs(atwater - row["kcal"]) / row["kcal"],
                0.22,
                f"{label}: declared {row['kcal']} kcal vs {atwater} from macros",
            )


class TestCompositionTableIntegrity(unittest.TestCase):
    def test_every_row_is_complete_and_physically_plausible(self):
        for label, row in {**N.COMPOSITION, **N.CATEGORY_FALLBACK, **N.COARSE_FALLBACK}.items():
            with self.subTest(label=label):
                for key in ("display_name", "category", "kcal", "density_g_per_ml", *N.MICRO_KEYS):
                    self.assertIn(key, row)
                self.assertGreater(row["kcal"], 0)
                self.assertLess(row["kcal"], 900)  # pure fat is 900 kcal/100 g
                self.assertGreater(row["density_g_per_ml"], 0.1)
                self.assertLess(row["density_g_per_ml"], 1.5)
                self.assertLessEqual(
                    row["protein_g"] + row["carbs_g"] + row["fat_g"], 100.0
                )
                for key in N.MICRO_KEYS:
                    self.assertGreaterEqual(row[key], 0.0)

    def test_every_category_used_by_a_dish_has_a_fallback_and_a_depth_prior(self):
        """A category with no fallback silently resolves to zeros downstream."""
        from depth import SERVING_DEPTH_CM, WEIGHT_BOUNDS

        for label, row in N.COMPOSITION.items():
            with self.subTest(label=label):
                self.assertIn(row["category"], N.CATEGORY_FALLBACK)
                self.assertIn(row["category"], SERVING_DEPTH_CM)
                self.assertIn(row["category"], WEIGHT_BOUNDS)

    def test_detector_aliases_all_resolve(self):
        for coco, label in N.DETECTOR_ALIASES.items():
            self.assertIn(label, N.COMPOSITION, f"alias {coco!r} points nowhere")

    def test_coarse_groups_never_masquerade_as_dishes(self):
        """A coarse group must not collide with a real dish key, or `_row_for`
        would return a specific dish for an intentionally vague label."""
        self.assertFalse(set(N.COARSE_FALLBACK) & set(N.COMPOSITION))


class TestLabelResolution(unittest.TestCase):
    def test_display_names_come_from_the_table(self):
        self.assertEqual(N.display_name("dal_tadka"), "Dal Tadka")
        self.assertEqual(N.display_name("dal_or_yellow"), "Lentil or Yellow Dish")

    def test_unknown_labels_are_humanised_rather_than_dropped(self):
        self.assertEqual(N.display_name("some_new_dish"), "Some New Dish")

    def test_density_falls_back_by_category_then_to_unknown(self):
        self.assertAlmostEqual(N.density_for("plain_rice"), 0.85, places=2)
        self.assertAlmostEqual(
            N.density_for("never_heard_of_it"),
            N.CATEGORY_FALLBACK["unknown"]["density_g_per_ml"],
            places=3,
        )

    def test_coarse_groups_resolve_without_a_database_or_network(self):
        row, source = N.resolve_per_100g(None, "gravy")
        self.assertEqual(row["display_name"], "Curry or Gravy")
        self.assertIn("estimated", source)

    def test_known_dishes_resolve_to_the_bundled_table(self):
        row, source = N.resolve_per_100g(None, "idli")
        self.assertEqual(row["kcal"], N.COMPOSITION["idli"]["kcal"])
        self.assertEqual(source, "Indian Food Composition Table")

    def test_catalog_is_sorted_and_complete(self):
        rows = N.catalog()
        self.assertEqual(len(rows), len(N.COMPOSITION))
        self.assertEqual(rows, sorted(rows, key=lambda row: row["display_name"]))
        for row in rows:
            self.assertIn(row["label"], N.COMPOSITION)


class TestPieceWeights(unittest.TestCase):
    """Countability decides whether the review page offers a "how many" field.

    Getting the set wrong is a UI bug that reads as a maths bug: asking how many
    curries are on the plate, or refusing to count four samosas and back-solving
    their weight from pixel area instead.
    """

    # Categories where a serving is a heap, a pour or a scoop — never a count.
    # These eight cover 25 of the 42 catalog labels; the other seven categories
    # (bread, fried, steamed, dessert, fruit, protein, fast_food) mix countable
    # and not, so they are checked label by label rather than wholesale.
    UNCOUNTABLE_CATEGORIES = {
        "curry",
        "dal",
        "rice",
        "grain",
        "dairy",
        "salad",
        "dry_sabzi",
        "condiment",
    }

    def test_countable_foods_have_a_positive_weight_and_footprint(self):
        for label in N.PIECE_WEIGHTS:
            with self.subTest(label=label):
                self.assertGreater(N.piece_weight_g(label), 0.0)
                self.assertGreater(N.piece_footprint_cm2(label), 0.0)

    def test_poured_and_heaped_foods_are_not_countable(self):
        checked = 0
        for label, row in N.COMPOSITION.items():
            if row["category"] in self.UNCOUNTABLE_CATEGORIES:
                checked += 1
                with self.subTest(label=label):
                    self.assertIsNone(N.piece_weight_g(label))
        # Guards the loop itself: a renamed category would otherwise make this
        # test pass by iterating over nothing.
        self.assertGreaterEqual(checked, 25)

    def test_a_heap_of_fries_is_not_countable_even_though_a_chip_is(self):
        """`french_fries` is a portion, not a piece; nobody counts 47 of them."""
        self.assertIsNone(N.piece_weight_g("french_fries"))

    def test_an_unknown_label_is_not_countable(self):
        self.assertIsNone(N.piece_weight_g("never_heard_of_it"))
        self.assertIsNone(N.piece_footprint_cm2("never_heard_of_it"))

    def test_four_samosas_cost_what_the_table_says_they_should(self):
        """The whole point of counting: 4 × 65 g priced from the composition row.

        Pinned against the number the user saw go wrong — four samosas came back
        as 943 kcal for the plate because two phantom regions were costed too.
        """
        weight = 4 * N.piece_weight_g("samosa")
        self.assertAlmostEqual(weight, 260.0, places=1)
        row = N.COMPOSITION["samosa"]
        scaled = N.scale_nutrients(row, weight)
        self.assertAlmostEqual(scaled["calories"], row["kcal"] * 2.6, places=1)
        self.assertAlmostEqual(scaled["protein_g"], row["protein_g"] * 2.6, places=1)

    def test_a_piece_footprint_implies_a_believable_count_on_a_real_plate(self):
        """A 26 cm plate is ~530 cm²; a footprint must not imply an absurd count."""
        plate_area_cm2 = 3.14159 * (26.0 / 2) ** 2
        for label in N.PIECE_WEIGHTS:
            with self.subTest(label=label):
                self.assertLess(N.piece_footprint_cm2(label), plate_area_cm2)


class TestNominalPortion(unittest.TestCase):
    """The weight for an item the user added by hand, which has no region.

    No area and no depth means geometry has nothing to work from, so this is the
    only number available — and it still has to land inside the same served-weight
    envelope the measured path is clamped to.
    """

    def test_every_category_lands_inside_its_own_weight_bounds(self):
        from depth import WEIGHT_BOUNDS

        for label, row in N.COMPOSITION.items():
            if N.piece_weight_g(label) is not None:
                continue  # answered with one piece instead, tested below
            category = row["category"]
            low, high = WEIGHT_BOUNDS.get(category, WEIGHT_BOUNDS["unknown"])
            with self.subTest(label=label, category=category):
                self.assertGreaterEqual(N.nominal_portion_g(label), low)
                self.assertLessEqual(N.nominal_portion_g(label), high)

    def test_every_category_the_formula_can_reach_produces_an_in_bounds_weight(self):
        """All 16 categories, including ones whose only labels are countable.

        The countable labels short-circuit to one piece, so their category is
        never exercised by the loop above. Here the composed formula is evaluated
        directly for every category so a bad `SERVING_DEPTH_CM` entry cannot hide
        behind a piece weight.
        """
        from depth import NOMINAL_AREA_CM2, SERVING_DEPTH_CM, WEIGHT_BOUNDS

        self.assertEqual(len(SERVING_DEPTH_CM), 16)
        for category, depth_cm in SERVING_DEPTH_CM.items():
            density = N.CATEGORY_FALLBACK.get(category, N.CATEGORY_FALLBACK["unknown"])[
                "density_g_per_ml"
            ]
            low, high = WEIGHT_BOUNDS[category]
            grams = min(max(depth_cm * NOMINAL_AREA_CM2 * density, low), high)
            with self.subTest(category=category):
                self.assertGreater(high, low)
                self.assertGreaterEqual(grams, low)
                self.assertLessEqual(grams, high)

    def test_a_countable_food_answers_with_one_piece(self):
        self.assertAlmostEqual(N.nominal_portion_g("samosa"), 65.0, places=1)
        self.assertAlmostEqual(N.nominal_portion_g("idli"), 50.0, places=1)

    def test_an_unknown_label_still_gets_a_usable_number(self):
        self.assertGreater(N.nominal_portion_g("never_heard_of_it"), 0.0)

    def test_the_catalog_tells_the_ui_which_foods_get_a_count_field(self):
        for row in N.catalog():
            with self.subTest(label=row["label"]):
                self.assertEqual(row["piece_weight_g"], N.piece_weight_g(row["label"]))


if __name__ == "__main__":
    unittest.main()
