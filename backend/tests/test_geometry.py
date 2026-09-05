"""Stage-3 geometry: elevation → volume → weight (design.md §7.3, §14.3).

These are the numbers a user actually reads, and they come out of pure functions
with no model in the loop — so they are exactly the thing worth pinning down.
The properties asserted here are the ones that broke during development:

* volume must be independent of how *peaked* the elevation field is
* a dome's mean elevation is 1/3 of its peak, so peak-height priors silently
  triple-discount every portion
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Before importing anything that reads config: the last test class drives the
# real pipeline, and the torch path would spend a minute loading a checkpoint to
# answer a question about pixel area. `setdefault` so the full-suite run, where
# test_api sets the same value first, is unaffected.
os.environ.setdefault("ENABLE_TORCH_MODELS", "false")

from depth import (  # noqa: E402
    MAX_DEPTH_CM,
    NOMINAL_AREA_CM2,
    SERVING_DEPTH_CM,
    WEIGHT_BOUNDS,
    clamp_weight,
    estimate_volume,
    integrate_volume,
    mask_for,
    mean_depth_cm,
    normalized_elevation,
    weight_from_volume,
)
from detection import Detection, PlateEstimate  # noqa: E402

import nutrition  # noqa: E402
from pipeline import remeasure_for_plate  # noqa: E402


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2


class TestIntegrateVolume(unittest.TestCase):
    def test_flat_slab_is_area_times_depth(self):
        """The simplest case must be exact: 100 px at 1 cm²/px and 2 cm deep."""
        elevation = np.ones((10, 10), dtype=np.float32)
        self.assertAlmostEqual(integrate_volume(elevation, 2.0, 1.0), 200.0, places=6)

    def test_volume_is_independent_of_shape_peakiness(self):
        """A dome and a slab of equal footprint and equal *mean* depth are equal.

        This is the invariant that the `mean_cm` formulation exists to provide.
        Before it, a dome came out at a third of the slab for the same prior.
        """
        mask = disc((80, 80), (39.5, 39.5), 30.0)
        slab = mask.astype(np.float32)
        dome = normalized_elevation(mask, None)
        self.assertGreater(dome.max() / max(dome.mean(), 1e-9), 2.0)  # genuinely peaked
        self.assertAlmostEqual(
            integrate_volume(dome, 3.0, 0.01),
            integrate_volume(slab, 3.0, 0.01),
            places=6,
        )

    def test_scales_linearly_with_depth_and_pixel_area(self):
        elevation = normalized_elevation(disc((60, 60), (29.5, 29.5), 20.0), None)
        base = integrate_volume(elevation, 2.0, 0.02)
        self.assertAlmostEqual(integrate_volume(elevation, 4.0, 0.02), base * 2.0, places=6)
        self.assertAlmostEqual(integrate_volume(elevation, 2.0, 0.04), base * 2.0, places=6)

    def test_degenerate_inputs_yield_zero(self):
        elevation = np.ones((4, 4), dtype=np.float32)
        self.assertEqual(integrate_volume(elevation, 0.0, 1.0), 0.0)
        self.assertEqual(integrate_volume(elevation, 2.0, 0.0), 0.0)
        self.assertEqual(integrate_volume(np.zeros((4, 4), dtype=np.float32), 2.0, 1.0), 0.0)


class TestNormalizedElevation(unittest.TestCase):
    def test_dome_mean_is_about_a_third_of_peak(self):
        """∫₀^R (1 − r/R)·2πr dr ÷ πR² = 1/3 — the cone-volume identity.

        Documented as a test because assuming otherwise is what pinned every
        portion to its clamp floor: a "peak height" prior applied to this field
        yields a third of the intended volume.
        """
        mask = disc((240, 240), (119.5, 119.5), 100.0)
        field = normalized_elevation(mask, None)
        ratio = field[mask].mean() / field.max()
        self.assertAlmostEqual(ratio, 1 / 3, delta=0.05)

    def test_bounded_to_unit_range_and_confined_to_mask(self):
        mask = disc((50, 50), (24.5, 24.5), 15.0)
        field = normalized_elevation(mask, None)
        self.assertTrue(np.all(field >= 0.0) and np.all(field <= 1.0))
        self.assertTrue(np.all(field[~mask] == 0.0))

    def test_empty_mask_is_all_zero(self):
        field = normalized_elevation(np.zeros((8, 8), dtype=bool), None)
        self.assertEqual(field.sum(), 0.0)

    def test_depth_map_shapes_the_field_without_escaping_the_mask(self):
        mask = disc((60, 60), (29.5, 29.5), 20.0)
        depth = np.zeros((60, 60), dtype=np.float32)
        depth[:30, :] = 1.0  # top half "closer" to the camera
        field = normalized_elevation(mask, depth)
        self.assertTrue(np.all(field[~mask] == 0.0))
        top = field[:30][mask[:30]].mean()
        bottom = field[30:][mask[30:]].mean()
        self.assertGreater(top, bottom)

    def test_flat_depth_map_falls_back_to_the_dome(self):
        mask = disc((60, 60), (29.5, 29.5), 20.0)
        flat = np.full((60, 60), 0.5, dtype=np.float32)
        np.testing.assert_allclose(
            normalized_elevation(mask, flat), normalized_elevation(mask, None), atol=1e-6
        )


class TestMeanDepthPrior(unittest.TestCase):
    def test_nominal_area_returns_the_category_prior_unchanged(self):
        for category, expected in SERVING_DEPTH_CM.items():
            self.assertAlmostEqual(mean_depth_cm(category, NOMINAL_AREA_CM2), expected, places=6)

    def test_unknown_category_uses_the_unknown_prior(self):
        self.assertAlmostEqual(
            mean_depth_cm("not-a-real-category", NOMINAL_AREA_CM2),
            SERVING_DEPTH_CM["unknown"],
            places=6,
        )

    def test_depth_rises_weakly_with_footprint(self):
        small = mean_depth_cm("curry", 20.0)
        large = mean_depth_cm("curry", 240.0)
        self.assertLess(small, large)
        # A 12× area increase must not double the depth, or weight would grow
        # super-quadratically with a mis-detected footprint.
        self.assertLess(large / small, 2.0)

    def test_extremes_stay_inside_the_physical_envelope(self):
        self.assertLessEqual(mean_depth_cm("dal", 10_000.0), MAX_DEPTH_CM)
        self.assertGreater(mean_depth_cm("bread", 0.01), 0.0)

    def test_flatbread_is_shallower_than_anything_bowl_served(self):
        area = NOMINAL_AREA_CM2
        self.assertLess(mean_depth_cm("bread", area), mean_depth_cm("dal", area))
        self.assertLess(mean_depth_cm("bread", area), mean_depth_cm("curry", area))


class TestWeightConversion(unittest.TestCase):
    def test_grams_are_millilitres_times_density(self):
        self.assertAlmostEqual(weight_from_volume(250.0, 1.03), 257.5, places=6)
        self.assertEqual(weight_from_volume(0.0, 0.85), 0.0)

    def test_rejects_impossible_inputs(self):
        with self.assertRaises(ValueError):
            weight_from_volume(-1.0, 1.0)
        with self.assertRaises(ValueError):
            weight_from_volume(1.0, 0.0)

    def test_clamp_reports_whether_it_intervened(self):
        low, high = WEIGHT_BOUNDS["rice"]
        inside, touched = clamp_weight((low + high) / 2, "rice")
        self.assertAlmostEqual(inside, (low + high) / 2)
        self.assertFalse(touched)

        under, touched = clamp_weight(low / 10, "rice")
        self.assertAlmostEqual(under, low)
        self.assertTrue(touched)

        over, touched = clamp_weight(high * 10, "rice")
        self.assertAlmostEqual(over, high)
        self.assertTrue(touched)

    def test_unknown_category_falls_back_to_unknown_bounds(self):
        low, _ = WEIGHT_BOUNDS["unknown"]
        value, touched = clamp_weight(0.0, "not-a-real-category")
        self.assertAlmostEqual(value, low)
        self.assertTrue(touched)


class TestMaskFor(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.food = disc(self.shape, (49.5, 49.5), 25.0)

    def test_prefers_the_detection_mask_when_present(self):
        own = disc(self.shape, (30.0, 30.0), 8.0)
        detection = Detection(bbox=(0, 0, 100, 100), label="x", confidence=1.0, mask=own)
        np.testing.assert_array_equal(mask_for(detection, self.food, self.shape), own)

    def test_intersects_the_box_with_the_global_food_mask(self):
        detection = Detection(bbox=(40, 40, 30, 30), label="x", confidence=1.0)
        mask = mask_for(detection, self.food, self.shape)
        self.assertTrue(mask.any())
        self.assertTrue(np.all(mask <= self.food))  # never claims non-food pixels
        self.assertFalse(mask[:40].any())  # never leaves the box

    def test_falls_back_to_an_inscribed_ellipse_off_the_food(self):
        detection = Detection(bbox=(0, 0, 20, 20), label="x", confidence=1.0)
        mask = mask_for(detection, np.zeros(self.shape, dtype=bool), self.shape)
        self.assertTrue(mask.any())
        self.assertFalse(mask[:, 20:].any())
        # An ellipse inscribed in a square covers ~π/4 of it.
        self.assertAlmostEqual(mask.sum() / (20 * 20), np.pi / 4, delta=0.06)

    def test_degenerate_box_yields_an_empty_mask(self):
        detection = Detection(bbox=(50, 50, 0, 0), label="x", confidence=1.0)
        self.assertFalse(mask_for(detection, self.food, self.shape).any())


class SyntheticPlate:
    """A plate whose pixel→cm scale is known exactly, so weights are checkable.

    Shared by the two classes below rather than inherited from one to the other:
    subclassing a `TestCase` for its fixtures re-runs all of its tests, which
    inflates the count without testing anything twice over.
    """

    def plate(self, diameter_cm: float = 26.0, radius_px: float = 200.0) -> PlateEstimate:
        shape = (480, 480)
        return PlateEstimate(
            center=(240.0, 240.0),
            radius_px=radius_px,
            mask=disc(shape, (239.5, 239.5), radius_px),
            detected=True,
            diameter_cm=diameter_cm,
        )

    def item(self, radius_px: float) -> Detection:
        mask = disc((480, 480), (239.5, 239.5), radius_px)
        size = int(radius_px * 2)
        return Detection(
            bbox=(240 - int(radius_px), 240 - int(radius_px), size, size),
            label="gravy",
            confidence=0.9,
            mask=mask,
            area_px=int(mask.sum()),
        )

    def estimate(self, detection: Detection, **kwargs):
        plate = kwargs.pop("plate", self.plate())
        return estimate_volume(
            detection,
            category=kwargs.pop("category", "curry"),
            density_g_per_ml=kwargs.pop("density_g_per_ml", 1.0),
            plate=plate,
            depth=kwargs.pop("depth", None),
            food_mask=kwargs.pop("food_mask", plate.mask),
            shape=(480, 480),
        )


class TestEstimateVolume(SyntheticPlate, unittest.TestCase):
    """End-to-end stage 3 on a synthetic plate with an exactly known scale."""

    def test_area_matches_the_plate_scale(self):
        """A blob of 40% the plate radius covers 0.4² of a 26 cm plate's area."""
        result = self.estimate(self.item(80.0))
        expected = np.pi * (26.0 / 2) ** 2 * 0.4**2
        self.assertAlmostEqual(result.area_cm2, expected, delta=expected * 0.03)

    def test_a_normal_serving_is_not_clamped(self):
        result = self.estimate(self.item(70.0))
        self.assertFalse(result.clamped)
        low, high = WEIGHT_BOUNDS["curry"]
        self.assertGreater(result.weight_g, low)
        self.assertLess(result.weight_g, high)

    def test_mean_height_is_volume_over_area(self):
        result = self.estimate(self.item(70.0))
        self.assertAlmostEqual(result.mean_height_cm, result.volume_ml / result.area_cm2, delta=0.02)

    def test_mean_height_tracks_the_category_prior(self):
        result = self.estimate(self.item(70.0))
        self.assertAlmostEqual(
            result.mean_height_cm, mean_depth_cm("curry", result.area_cm2), delta=0.15
        )

    def test_denser_food_weighs_more_for_the_same_shape(self):
        light = self.estimate(self.item(60.0), category="salad", density_g_per_ml=0.38)
        heavy = self.estimate(self.item(60.0), category="salad", density_g_per_ml=1.03)
        self.assertGreater(heavy.weight_g, light.weight_g)

    def test_volume_is_restated_consistently_after_clamping(self):
        """A clamped weight must not leave an inconsistent volume behind."""
        result = self.estimate(self.item(190.0), category="condiment", density_g_per_ml=0.95)
        self.assertTrue(result.clamped)
        self.assertAlmostEqual(result.volume_ml, result.weight_g / 0.95, delta=0.2)

    def test_larger_footprint_means_more_food(self):
        weights = [self.estimate(self.item(r)).weight_g for r in (40.0, 60.0, 80.0)]
        self.assertEqual(weights, sorted(weights))

    def test_bigger_plate_assumption_scales_weight_up(self):
        """The plate diameter is the only absolute scale in the system (§20 risk 1)."""
        small = self.estimate(self.item(70.0), plate=self.plate(diameter_cm=20.0))
        large = self.estimate(self.item(70.0), plate=self.plate(diameter_cm=32.0))
        self.assertGreater(large.weight_g, small.weight_g)

    def test_empty_region_degrades_to_a_flagged_portion(self):
        empty = Detection(bbox=(0, 0, 0, 0), label="gravy", confidence=0.5)
        result = self.estimate(empty, food_mask=np.zeros((480, 480), dtype=bool))
        self.assertEqual(result.method, "fallback-portion")
        self.assertTrue(result.clamped)
        self.assertAlmostEqual(result.weight_g, WEIGHT_BOUNDS["curry"][0], places=1)

    def test_method_names_the_depth_source(self):
        self.assertEqual(self.estimate(self.item(70.0)).method, "shape-prior+geometry")
        depth = np.zeros((480, 480), dtype=np.float32)
        depth[200:280, 200:280] = 1.0
        self.assertEqual(self.estimate(self.item(70.0), depth=depth).method, "midas+geometry")

    def test_reported_peak_exceeds_mean_but_stays_physical(self):
        result = self.estimate(self.item(70.0))
        self.assertGreater(result.peak_height_cm, result.mean_height_cm)
        self.assertLessEqual(result.peak_height_cm, MAX_DEPTH_CM * 2.5)


class TestRemeasureForPlate(SyntheticPlate, unittest.TestCase):
    """`pipeline.remeasure_for_plate` must be stage 3 with a different ruler.

    A user who corrects the plate size afterwards gets `PATCH /plate`, which
    cannot re-run MiDaS on a photo it no longer holds in memory — so it redoes
    the arithmetic from the item's stored geometry instead. That is only sound
    because `integrate_volume` normalises elevation to unit mean, making volume
    exactly footprint × mean depth: the shape of the mound cancels, so it does
    not need to be remembered.

    These tests pin that equivalence against the real `estimate_volume`, which
    is the only way to catch the two routes drifting apart again.
    """

    # A real label, because the recomputation resolves category and density from
    # the catalog rather than taking them on faith. `curry`, to match the
    # fixture's category.
    LABEL = "chole_masala"

    def geometry_of(self, estimate) -> dict:
        """The `_geometry` dict `analyze_scanned` would have persisted."""
        return {
            "area_cm2": estimate.area_cm2,
            "mean_height_cm": estimate.mean_height_cm,
            "peak_height_cm": estimate.peak_height_cm,
            "density_g_per_ml": round(estimate.density_g_per_ml, 3),
            "method": estimate.method,
        }

    def both_ways(self, radius_px: float, was: float, now: float, **kwargs):
        """Measure at `was` cm then correct to `now`, vs. measuring at `now`."""
        density = kwargs.setdefault("density_g_per_ml", nutrition.density_for(self.LABEL))
        first = self.estimate(self.item(radius_px), plate=self.plate(diameter_cm=was), **kwargs)
        corrected = remeasure_for_plate(
            self.LABEL,
            self.geometry_of(first),
            area_ratio=(now / was) ** 2,
            session=None,
        )
        direct = self.estimate(self.item(radius_px), plate=self.plate(diameter_cm=now), **kwargs)
        self.assertIsNotNone(corrected, "a measured item must be re-measurable")
        self.assertAlmostEqual(density, direct.density_g_per_ml, places=3)
        return corrected, direct

    def test_correcting_upward_matches_measuring_at_that_size(self):
        corrected, direct = self.both_ways(70.0, was=26.0, now=34.0)
        self.assertAlmostEqual(corrected.weight_g, direct.weight_g, delta=direct.weight_g * 0.005)
        self.assertAlmostEqual(corrected.geometry["area_cm2"], direct.area_cm2, delta=0.5)
        self.assertAlmostEqual(
            corrected.geometry["mean_height_cm"], direct.mean_height_cm, delta=0.02
        )

    def test_correcting_downward_matches_too(self):
        corrected, direct = self.both_ways(70.0, was=34.0, now=20.0)
        self.assertAlmostEqual(corrected.weight_g, direct.weight_g, delta=direct.weight_g * 0.005)

    def test_a_correction_that_changes_nothing_changes_nothing(self):
        corrected, direct = self.both_ways(70.0, was=26.0, now=26.0)
        self.assertAlmostEqual(corrected.weight_g, direct.weight_g, delta=0.2)

    def test_the_old_cube_rule_would_have_failed_this(self):
        """The regression this replaced, stated as a number rather than a story."""
        was, now, radius = 26.0, 34.0, 70.0
        first = self.estimate(self.item(radius), plate=self.plate(diameter_cm=was))
        corrected, _ = self.both_ways(radius, was=was, now=now)
        cubed = first.weight_g * (now / was) ** 3
        self.assertGreater(cubed, corrected.weight_g * 1.1, "d³ used to over-report by ~20%")

    def test_a_clamped_correction_stays_inside_the_envelope_and_says_so(self):
        low, high = WEIGHT_BOUNDS["curry"]
        corrected, direct = self.both_ways(190.0, was=26.0, now=44.0)
        self.assertTrue(direct.clamped)
        self.assertTrue(corrected.weight_estimated)
        self.assertAlmostEqual(corrected.weight_g, high, places=1)
        # A clamped weight must not leave an inconsistent volume behind, exactly
        # as the forward pass guarantees.
        self.assertAlmostEqual(
            corrected.volume_ml,
            corrected.weight_g / corrected.geometry["density_g_per_ml"],
            delta=0.3,
        )
        self.assertGreater(corrected.weight_g, low)

    def test_the_mound_shape_survives_the_change_of_ruler(self):
        """`peak / mean` is scale-free, so it is the one thing that must not move."""
        first = self.estimate(self.item(70.0), plate=self.plate(diameter_cm=26.0))
        corrected = remeasure_for_plate(
            self.LABEL, self.geometry_of(first), area_ratio=(34.0 / 26.0) ** 2, session=None
        )
        before = first.peak_height_cm / first.mean_height_cm
        after = corrected.geometry["peak_height_cm"] / corrected.geometry["mean_height_cm"]
        self.assertAlmostEqual(after, before, delta=0.05)

    def test_counted_and_hand_added_items_are_not_re_measured(self):
        """Neither weight ever passed through pixel area, so neither may move."""
        for method in ("piece-count", "nominal-portion"):
            with self.subTest(method=method):
                self.assertIsNone(
                    remeasure_for_plate(
                        "samosa",
                        {"area_cm2": 84.0, "mean_height_cm": 0.0, "method": method},
                        area_ratio=4.0,
                        session=None,
                    )
                )

    def test_an_item_with_no_footprint_is_not_re_measured(self):
        """Guards the `method` check: no area means nothing to scale, whatever it says."""
        self.assertIsNone(
            remeasure_for_plate(
                self.LABEL,
                {"area_cm2": 0.0, "mean_height_cm": 0.0, "method": "shape-prior+geometry"},
                area_ratio=4.0,
                session=None,
            )
        )

    def test_the_method_records_that_the_number_was_recalibrated(self):
        corrected, _ = self.both_ways(70.0, was=26.0, now=34.0)
        self.assertEqual(corrected.geometry["method"], "plate-recalibrated")


class TestAreaLessRegionsAreDropped(unittest.TestCase):
    """`scan_image` must reject a region before geometry ever sees it.

    This is the bug the two-phase flow exists to fix. A lemon wedge and a strip
    of basket weave were each detected, each resolved to no measurable area, and
    each still costed — `estimate_volume`'s `fallback-portion` branch handed back
    `WEIGHT_BOUNDS[category][0]`, so two non-foods arrived on the results page at
    exactly 45.0 g and ~142 kcal between them.

    The fallback branch is *not* the bug and stays: a user-added item genuinely
    has no region and still needs a number. What changed is that a *detection*
    now has to earn its area, and `scan_image` is the only place that can tell
    the two apart. So this pins the drop at the pipeline seam rather than
    re-testing `estimate_volume`.
    """

    @classmethod
    def setUpClass(cls):
        # Imported here, not at module scope: `pipeline` pulls in the whole
        # detection and classification stack, and the rest of this file is pure
        # geometry that should keep importing cheaply.
        import pipeline
        from classify import classifier
        from detection import detector

        # Normally the FastAPI lifespan does this. With torch off both fall
        # straight through to their heuristic backends, so it costs nothing.
        if not detector.ready:
            detector.load()
        if not classifier.ready:
            classifier.load()
        cls.pipeline = pipeline

    def setUp(self):
        self.shape = (480, 480)
        self.plate = PlateEstimate(
            center=(240.0, 240.0),
            radius_px=200.0,
            mask=disc(self.shape, (239.5, 239.5), 200.0),
            detected=True,
            diameter_cm=26.0,
        )

    def region(self, radius_px: float) -> Detection:
        """A detection carrying its own mask, so `mask_for` passes it straight through."""
        mask = disc(self.shape, (239.5, 239.5), radius_px)
        size = max(1, int(radius_px * 2))
        return Detection(
            bbox=(240 - int(radius_px), 240 - int(radius_px), size, size),
            label="gravy",
            confidence=0.35,  # the confidence the phantom "Fish Curry" came in at
            mask=mask,
            area_px=int(mask.sum()),
            meta={"source": "segmenter"},
        )

    def areas(self, detections: list[Detection]) -> list[float]:
        """Run the scan's drop rule over `detections` and return what survived."""
        kept = []
        for found in detections:
            resolved = mask_for(found, self.plate.mask, self.shape)
            area_px = int(resolved.sum())
            area_cm2 = area_px * self.plate.px_area_cm2
            if area_px == 0 or area_cm2 <= self.pipeline.MIN_ITEM_AREA_CM2:
                continue
            kept.append(area_cm2)
        return kept

    def test_the_drop_threshold_is_the_fallback_portion_trigger(self):
        """One constant, one condition — the two must not drift apart."""
        self.assertEqual(self.pipeline.MIN_ITEM_AREA_CM2, 0.5)

    def test_a_region_with_no_pixels_at_all_is_dropped(self):
        empty = Detection(bbox=(0, 0, 0, 0), label="gravy", confidence=0.35)
        self.assertEqual(self.areas([empty]), [])

    def test_a_sliver_too_small_to_weigh_is_dropped(self):
        """A 1 px radius blob on a 26 cm plate is well under half a cm²."""
        sliver = self.region(1.0)
        self.assertGreater(sliver.area_px, 0, "the sliver must have pixels to be interesting")
        self.assertEqual(self.areas([sliver]), [])

    def test_a_real_serving_survives(self):
        kept = self.areas([self.region(70.0)])
        self.assertEqual(len(kept), 1)
        self.assertGreater(kept[0], self.pipeline.MIN_ITEM_AREA_CM2)

    def test_only_the_phantoms_are_dropped_from_a_mixed_plate(self):
        food = self.region(70.0)
        phantoms = [Detection(bbox=(0, 0, 0, 0), label="gravy", confidence=0.24), self.region(1.0)]
        kept = self.areas([food, *phantoms])
        self.assertEqual(len(kept), 1)

    def test_scan_image_drops_them_and_says_so(self):
        """The rule wired up for real: stub the detector, keep everything else."""
        import detection as detection_module
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (600, 600), (246, 246, 243))
        draw = ImageDraw.Draw(image)
        draw.ellipse([40, 40, 560, 560], fill=(251, 251, 249))
        draw.ellipse([210, 210, 390, 390], fill=(188, 122, 44))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        payload = buffer.getvalue()

        real_detect = detection_module.detector.detect

        def detect_with_phantoms(prepared, plate):
            found = real_detect(prepared, plate)
            shape = (prepared.height, prepared.width)
            # A wedge of lemon the segmenter saw but that resolves to nothing,
            # and a strip of basket weave one pixel wide.
            return [
                *found,
                Detection(bbox=(0, 0, 0, 0), label="gravy", confidence=0.35),
                Detection(
                    bbox=(4, 4, 2, 2),
                    label="gravy",
                    confidence=0.24,
                    mask=disc(shape, (5.0, 5.0), 1.0),
                    area_px=int(disc(shape, (5.0, 5.0), 1.0).sum()),
                    meta={"source": "segmenter"},
                ),
            ]

        detection_module.detector.detect = detect_with_phantoms
        try:
            scan = self.pipeline.scan_image(payload, plate_diameter_cm=26.0)
        finally:
            detection_module.detector.detect = real_detect

        self.assertEqual(scan.dropped, 2)
        self.assertTrue(scan.items, "the real region should have survived")
        for item in scan.items:
            self.assertGreater(item.area_cm2, self.pipeline.MIN_ITEM_AREA_CM2)
        self.assertTrue(
            any("too small to measure" in warning for warning in scan.warnings),
            f"the user is never told what vanished: {scan.warnings}",
        )

    def test_a_plate_of_nothing_but_phantoms_is_not_a_meal(self):
        """Better to say "no food" than to serve a plate of minimum portions."""
        import detection as detection_module
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (600, 600), (250, 250, 248)).save(buffer, "JPEG", quality=92)
        real_detect = detection_module.detector.detect
        detection_module.detector.detect = lambda prepared, plate: [
            Detection(bbox=(0, 0, 0, 0), label="gravy", confidence=0.3)
        ]
        try:
            with self.assertRaises(self.pipeline.NoFoodDetectedError):
                self.pipeline.scan_image(buffer.getvalue(), plate_diameter_cm=26.0)
        finally:
            detection_module.detector.detect = real_detect


if __name__ == "__main__":
    unittest.main()
