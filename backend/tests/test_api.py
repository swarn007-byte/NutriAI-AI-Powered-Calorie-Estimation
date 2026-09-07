"""HTTP contract tests (design.md §10) against an isolated database.

Runs the real ASGI app in-process — routing, validation, multipart parsing,
auth, the five pipeline stages and JSON serialisation all execute for real. The
only thing not exercised is the TCP socket, which is uvicorn's concern.

The error contract from design.md §10 is the most valuable part to pin: a
pipeline that returns 200 with zeroes when it should have said 422 is a far
worse failure than one that crashes, because nothing downstream can tell.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urljoin

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Point config at throwaway storage *before* importing anything that reads it.
_TMP = tempfile.mkdtemp(prefix="nutriai-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP) / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(Path(_TMP) / "uploads")
os.environ["ENABLE_TORCH_MODELS"] = "false"  # keep the suite fast and offline
os.environ["JWT_SECRET"] = "test-secret"
# Off by default so every other test can keep assuming a new account owns
# nothing. TestWelcomeMeal re-enables it for the cases that are about the seed.
os.environ["SEED_WELCOME_MEAL"] = "false"

from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import seed  # noqa: E402

SAMPLES = BACKEND.parent / "frontend" / "samples"


def sample_bytes(name: str = "thali") -> bytes:
    path = SAMPLES / f"{name}.jpg"
    if not path.is_file():
        raise unittest.SkipTest(f"missing sample {path} — run tools/make_samples.py")
    return path.read_bytes()


def blank_image(colour: tuple[int, int, int] = (250, 250, 250), size: int = 420) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


class ApiTestCase(unittest.TestCase):
    """One client (and therefore one model warm-up) for the whole suite."""

    client: TestClient

    @classmethod
    def setUpClass(cls) -> None:
        cls._ctx = TestClient(main.app)
        cls.client = cls._ctx.__enter__()  # triggers lifespan → warm_models()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ctx.__exit__(None, None, None)

    def guest(self) -> tuple[str, dict[str, str]]:
        response = self.client.post("/api/auth/guest")
        self.assertEqual(response.status_code, 200)
        token = response.json()["token"]
        return token, {"Authorization": f"Bearer {token}"}

    def analyze(self, name: str = "thali", **data) -> dict:
        _, headers = self.guest()
        response = self.client.post(
            "/api/meals/analyze",
            files={"image": (f"{name}.jpg", sample_bytes(name), "image/jpeg")},
            data=data,
            headers=headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


class TestMeta(ApiTestCase):
    def test_health_reports_every_stage(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(set(body["models"]), {"detection", "depth", "classification", "nutrition"})
        for stage in body["models"].values():
            self.assertTrue(stage["ready"], stage)
        self.assertIn("max_upload_bytes", body["limits"])

    def test_health_engine_label_is_honest_about_the_fallbacks(self):
        """With torch disabled the engine must not claim to be the full stack."""
        self.assertEqual(self.client.get("/api/health").json()["engine"], "heuristic")

    def test_openapi_schema_builds(self):
        """A response_model that can't be generated breaks /docs, silently."""
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_catalog_powers_the_correction_picker(self):
        rows = self.client.get("/api/nutrition/catalog").json()
        self.assertGreater(len(rows), 30)
        self.assertEqual(set(rows[0]) >= {"label", "display_name", "kcal_per_100g"}, True)

    def test_lookup_normalises_free_text(self):
        body = self.client.get("/api/nutrition/lookup", params={"food": "Dal Tadka"}).json()
        self.assertEqual(body["food"], "dal_tadka")
        self.assertEqual(body["display_name"], "Dal Tadka")
        self.assertEqual(body["per_100g"]["kcal"], 128)
        self.assertGreater(body["density_g_per_ml"], 0)

    def test_lookup_of_an_unknown_dish_degrades_instead_of_failing(self):
        body = self.client.get("/api/nutrition/lookup", params={"food": "quinoa surprise"}).json()
        self.assertIn("estimated", body["source"])
        self.assertGreater(body["per_100g"]["kcal"], 0)

    def test_lookup_rejects_an_empty_query(self):
        self.assertEqual(self.client.get("/api/nutrition/lookup", params={"food": ""}).status_code, 422)


class TestAuth(ApiTestCase):
    def test_guest_needs_no_credentials(self):
        body = self.client.post("/api/auth/guest").json()
        self.assertTrue(body["user"]["is_guest"])
        self.assertIsNone(body["user"]["email"])
        self.assertTrue(body["token"])

    def test_me_requires_a_token(self):
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(
            self.client.get("/api/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code,
            401,
        )

    def test_me_returns_the_token_holder(self):
        token, headers = self.guest()
        body = self.client.get("/api/auth/me", headers=headers).json()
        self.assertTrue(body["is_guest"])

    def test_register_then_login(self):
        payload = {"email": "cook@example.com", "password": "correct-horse-battery", "name": "Cook"}
        created = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(created.status_code, 200, created.text)
        self.assertFalse(created.json()["user"]["is_guest"])

        duplicate = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(duplicate.status_code, 409)

        ok = self.client.post(
            "/api/auth/login", json={"email": payload["email"], "password": payload["password"]}
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["user"]["email"], payload["email"])

        wrong = self.client.post(
            "/api/auth/login", json={"email": payload["email"], "password": "not-it"}
        )
        self.assertEqual(wrong.status_code, 401)

    def test_registering_over_a_guest_token_keeps_that_guest_s_meals(self):
        """The no-login-wall promise (§13.1) is worthless if signing up loses work."""
        _, headers = self.guest()
        meal = self.client.post(
            "/api/meals/analyze",
            files={"image": ("thali.jpg", sample_bytes(), "image/jpeg")},
            headers=headers,
        ).json()

        upgraded = self.client.post(
            "/api/auth/register",
            json={"email": "upgrade@example.com", "password": "a-long-enough-password"},
            headers=headers,
        )
        self.assertEqual(upgraded.status_code, 200, upgraded.text)
        self.assertFalse(upgraded.json()["user"]["is_guest"])

        owned = self.client.get(
            f"/api/meals/{meal['meal_id']}",
            headers={"Authorization": f"Bearer {upgraded.json()['token']}"},
        )
        self.assertEqual(owned.status_code, 200)

    def test_password_must_be_long_enough(self):
        response = self.client.post(
            "/api/auth/register", json={"email": "short@example.com", "password": "abc"}
        )
        self.assertEqual(response.status_code, 422)

    def test_preferences_round_trip(self):
        _, headers = self.guest()
        response = self.client.patch(
            "/api/users/me/preferences",
            json={"calorie_goal": 2400, "plate_diameter_cm": 28.5},
            headers=headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        preferences = response.json()["preferences"]
        self.assertEqual(preferences["calorie_goal"], 2400)
        self.assertAlmostEqual(preferences["plate_diameter_cm"], 28.5)

    def test_preferences_partial_update_keeps_the_rest(self):
        _, headers = self.guest()
        self.client.patch("/api/users/me/preferences", json={"calorie_goal": 1800}, headers=headers)
        body = self.client.patch(
            "/api/users/me/preferences", json={"plate_diameter_cm": 24.0}, headers=headers
        ).json()
        self.assertEqual(body["preferences"]["calorie_goal"], 1800)


class TestAnalyze(ApiTestCase):
    def test_analyze_without_a_token_mints_a_guest(self):
        """design.md §13.1 — no login wall before the first try."""
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("thali.jpg", sample_bytes(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["user"]["is_guest"])
        self.assertTrue(body["token"])

    def test_result_shape_is_complete(self):
        body = self.analyze()
        self.assertGreater(len(body["items"]), 0)
        self.assertLessEqual(len(body["items"]), 6)
        for key in ("meal_id", "items", "totals", "micronutrients", "daily_values", "timings_ms"):
            self.assertIn(key, body)
        item = body["items"][0]
        for key in (
            "display_name", "confidence", "estimated_weight_g", "calories",
            "bbox", "alternatives", "nutrition_source", "geometry",
        ):
            self.assertIn(key, item)

    def test_totals_equal_the_sum_of_the_items(self):
        body = self.analyze()
        self.assertAlmostEqual(
            body["totals"]["calories"], sum(item["calories"] for item in body["items"]), delta=0.5
        )
        self.assertAlmostEqual(
            body["totals"]["protein_g"], sum(item["protein_g"] for item in body["items"]), delta=0.5
        )

    def test_numbers_are_physically_plausible(self):
        for name in ("thali", "dosa", "breakfast", "curry-bowl"):
            with self.subTest(sample=name):
                body = self.analyze(name)
                self.assertGreater(body["totals"]["calories"], 80)
                self.assertLess(body["totals"]["calories"], 2500)
                for item in body["items"]:
                    self.assertGreater(item["estimated_weight_g"], 0)
                    self.assertLess(item["estimated_weight_g"], 600)
                    self.assertGreater(item["estimated_volume_ml"], 0)

    def test_bounding_boxes_are_normalised_fractions(self):
        for item in self.analyze()["items"]:
            box = item["bbox"]
            self.assertTrue(0.0 <= box["x"] <= 1.0 and 0.0 <= box["y"] <= 1.0, box)
            self.assertTrue(0.0 < box["w"] <= 1.0 and 0.0 < box["h"] <= 1.0, box)

    def test_geometry_explains_where_the_weight_came_from(self):
        """§18 asks the app to show its reasoning, so the numbers must travel."""
        geometry = self.analyze()["items"][0]["geometry"]
        for key in ("area_cm2", "mean_height_cm", "density_g_per_ml", "method"):
            self.assertIn(key, geometry)
        self.assertGreater(geometry["area_cm2"], 0)

    def test_low_confidence_items_carry_alternatives_to_choose_from(self):
        body = self.analyze()
        for item in body["items"]:
            if item["low_confidence"]:
                self.assertTrue(item["alternatives"], "a flagged item with no alternatives is a dead end")

    def test_timings_are_reported_per_stage(self):
        timings = self.analyze()["timings_ms"]
        for stage in ("input", "detection", "depth", "classification", "volume", "nutrition", "total"):
            self.assertIn(stage, timings)
        self.assertGreater(timings["total"], 0)

    def test_plate_diameter_form_field_changes_the_portions(self):
        small = self.analyze(plate_diameter_cm=20.0)
        large = self.analyze(plate_diameter_cm=32.0)
        self.assertGreater(large["totals"]["calories"], small["totals"]["calories"])
        self.assertAlmostEqual(large["plate_diameter_cm"], 32.0)

    def test_notes_are_stored(self):
        self.assertEqual(self.analyze(notes="  lunch at home  ")["notes"], "lunch at home")

    def test_image_urls_are_capability_scoped(self):
        body = self.analyze()
        self.assertIn("?t=", body["image_url"])
        self.assertEqual(self.client.get(body["image_url"]).status_code, 200)
        self.assertEqual(self.client.get(body["thumb_url"]).status_code, 200)
        bare = body["image_url"].split("?")[0]
        self.assertEqual(self.client.get(bare).status_code, 403)
        self.assertEqual(self.client.get(f"{bare}?t=forged").status_code, 403)

    # ---- error contract, design.md §10 ---------------------------------

    def test_rejects_a_non_image_content_type(self):
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("notes.txt", b"hello", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_an_empty_upload(self):
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_a_file_that_is_not_a_decodable_image(self):
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("fake.jpg", b"\xff\xd8not-a-jpeg", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_an_oversized_upload(self):
        from config import settings

        payload = b"\xff\xd8" + b"\x00" * (settings.max_upload_bytes + 1024)
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("huge.jpg", payload, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("MB", response.json()["detail"])

    def test_missing_file_field_is_a_validation_error(self):
        self.assertEqual(self.client.post("/api/meals/analyze").status_code, 422)

    def test_an_empty_plate_is_422_not_a_meal_of_zero_calories(self):
        response = self.client.post(
            "/api/meals/analyze", files={"image": ("blank.jpg", blank_image(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("food", response.json()["detail"].lower())


class TestTwoPhaseFlow(ApiTestCase):
    """upload → scan → review → deep pass.

    The point of the split is that the user gets to strike a wrong item off the
    list *before* the app spends a depth pass on it and shows them a number they
    will remember. So the first thing worth asserting is the absence of numbers.
    """

    def setUp(self):
        self.token, self.headers = self.guest()
        self.draft = self.scan()
        self.draft_id = self.draft["draft_id"]

    def scan(self, name: str = "thali", **data) -> dict:
        response = self.client.post(
            "/api/meals/scan",
            files={"image": (f"{name}.jpg", sample_bytes(name), "image/jpeg")},
            data=data,
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def deep(self, plate_diameter_cm: float = 26.0, items: list | None = None, **extra) -> dict:
        response = self.client.post(
            f"/api/meals/{self.draft_id}/analyze",
            json={"plate_diameter_cm": plate_diameter_cm, "items": items or [], **extra},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    # ---- phase one ------------------------------------------------------

    def test_a_scan_names_items_without_costing_them(self):
        self.assertTrue(self.draft["items"])
        for item in self.draft["items"]:
            self.assertNotIn("calories", item)
            self.assertNotIn("protein_g", item)
            self.assertNotIn("estimated_weight_g", item)
            self.assertTrue(item["display_name"])
            self.assertGreater(item["area_cm2"], 0.0)

    def test_a_scan_skips_the_depth_pass(self):
        """Depth is only ever consumed by volume, and volume needs a plate size."""
        self.assertNotIn("depth", self.draft["timings_ms"])
        self.assertNotIn("volume", self.draft["timings_ms"])
        self.assertNotIn("nutrition", self.draft["timings_ms"])
        self.assertIn("detection", self.draft["timings_ms"])
        self.assertIn("classification", self.draft["timings_ms"])

    def test_a_scan_indexes_items_so_edits_cannot_drift(self):
        indexes = [item["index"] for item in self.draft["items"]]
        self.assertEqual(indexes, list(range(len(indexes))))

    def test_a_scan_mints_a_guest_like_analyze_does(self):
        response = self.client.post(
            "/api/meals/scan",
            files={"image": ("thali.jpg", sample_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["user"]["is_guest"])
        self.assertTrue(body["token"])

    def test_a_scan_image_is_capability_scoped(self):
        url = self.draft["image_url"]
        self.assertRegex(url, r"^/media/[0-9a-f-]{36}\?t=[0-9a-f]{32}$")
        self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(self.client.get(url.split("?")[0] + "?t=wrong").status_code, 403)

    def test_countable_items_carry_a_count_and_say_it_is_a_guess(self):
        countable = [row for row in self.draft["items"] if row["piece_weight_g"] is not None]
        for item in countable:
            self.assertGreaterEqual(item["piece_count"], 1)
            self.assertTrue(item["piece_count_estimated"])
        for item in self.draft["items"]:
            if item["piece_weight_g"] is None:
                self.assertIsNone(item["piece_count"])

    def test_an_empty_plate_is_422_before_anything_is_costed(self):
        response = self.client.post(
            "/api/meals/scan",
            files={"image": ("blank.jpg", blank_image(), "image/jpeg")},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 422)

    def test_scan_rejects_a_non_image_content_type(self):
        response = self.client.post(
            "/api/meals/scan",
            files={"image": ("notes.txt", b"hello", "text/plain")},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)

    # ---- phase two ------------------------------------------------------

    def test_the_deep_pass_costs_the_reviewed_list(self):
        meal = self.deep()
        self.assertEqual(len(meal["items"]), len(self.draft["items"]))
        self.assertGreater(meal["totals"]["calories"], 0.0)
        self.assertAlmostEqual(
            meal["totals"]["calories"], sum(row["calories"] for row in meal["items"]), delta=0.5
        )
        self.assertIn("depth", meal["timings_ms"])
        self.assertIn("nutrition", meal["timings_ms"])

    def test_removing_an_item_removes_it_from_the_totals(self):
        victim = self.draft["items"][0]
        meal = self.deep(items=[{"index": victim["index"], "removed": True}])
        self.assertEqual(len(meal["items"]), len(self.draft["items"]) - 1)
        self.assertNotIn(victim["label"], [row["classified_label"] for row in meal["items"]])

    def test_removing_every_item_is_422_not_a_meal_of_zero_calories(self):
        response = self.client.post(
            f"/api/meals/{self.draft_id}/analyze",
            json={
                "plate_diameter_cm": 26.0,
                "items": [{"index": row["index"], "removed": True} for row in self.draft["items"]],
            },
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 422)

    def count_first_item(self, count: int, label: str = "samosa") -> tuple[dict, dict]:
        """Rename the first scanned row to a countable dish and give it a count.

        The rename is what makes this reachable in the test environment at all:
        `ENABLE_TORCH_MODELS=false` leaves the heuristic classifier, which emits
        coarse groups like `gravy` and never a countable dish. Renaming is also
        exactly what a user does on the review page, so nothing here is contrived.
        """
        target = self.draft["items"][0]
        meal = self.deep(items=[{"index": target["index"], "label": label, "piece_count": count}])
        row = next(item for item in meal["items"] if item["classified_label"] == label)
        return meal, row

    def test_a_piece_count_drives_the_weight_and_halving_it_halves_the_calories(self):
        _, four = self.count_first_item(4)
        self.assertAlmostEqual(four["estimated_weight_g"], 4 * 65.0, delta=0.5)
        self.assertEqual(four["piece_count"], 4)
        self.assertEqual(four["piece_weight_g"], 65.0)
        self.assertEqual(four["geometry"]["method"], "piece-count")
        # A count the user typed is not an estimate.
        self.assertFalse(four["weight_estimated"])

        self.draft = self.scan()
        self.draft_id = self.draft["draft_id"]
        _, two = self.count_first_item(2)
        self.assertAlmostEqual(two["calories"], four["calories"] / 2, delta=1.0)

    def test_a_guessed_count_is_flagged_as_estimated(self):
        """The area guess is a guess; only a count the user set is not."""
        target = self.draft["items"][0]
        meal = self.deep(items=[{"index": target["index"], "label": "samosa"}])
        row = next(item for item in meal["items"] if item["classified_label"] == "samosa")
        self.assertGreaterEqual(row["piece_count"], 1)
        self.assertTrue(row["weight_estimated"])
        self.assertTrue(row["geometry"]["piece_count_estimated"])

    def test_a_non_countable_item_gets_its_weight_from_the_photo(self):
        meal = self.deep()
        measured = [row for row in meal["items"] if row["piece_count"] is None]
        self.assertTrue(measured, "expected at least one item measured from geometry")
        for row in measured:
            self.assertIn(row["geometry"]["method"], {"shape-prior+geometry", "midas+geometry"})
            self.assertGreater(row["geometry"]["area_cm2"], 0.0)

    def test_a_counted_items_footprint_is_measured_at_the_users_plate_size(self):
        """The footprint of a counted item must use the plate width the user gave.

        The weight of a counted item is `count × grams per piece`, which needs no
        plate scale at all — so nothing about the calories catches it if the
        reported footprint is still the one measured against the provisional
        width during the scan. The results page prints that footprint next to
        the geometry-measured items' own, and at two different scales the
        smaller number can describe the larger region.
        """
        target = self.draft["items"][0]
        edit = [{"index": target["index"], "label": "samosa", "piece_count": 4}]
        narrow_meal = self.deep(plate_diameter_cm=13.0, items=edit)
        narrow = next(r for r in narrow_meal["items"] if r["classified_label"] == "samosa")

        self.draft = self.scan()
        self.draft_id = self.draft["draft_id"]
        target = self.draft["items"][0]
        edit = [{"index": target["index"], "label": "samosa", "piece_count": 4}]
        wide_meal = self.deep(plate_diameter_cm=26.0, items=edit)
        wide_row = next(r for r in wide_meal["items"] if r["classified_label"] == "samosa")

        # Doubling the diameter quadruples every cm² on the plate.
        self.assertAlmostEqual(
            wide_row["geometry"]["area_cm2"],
            narrow["geometry"]["area_cm2"] * 4,
            delta=max(1.0, narrow["geometry"]["area_cm2"] * 0.02),
        )
        # ...and leaves the weight alone, because four samosas are four samosas.
        self.assertAlmostEqual(
            wide_row["estimated_weight_g"], narrow["estimated_weight_g"], delta=0.5
        )

    def test_renaming_an_item_changes_its_nutrition(self):
        target = self.draft["items"][0]
        meal = self.deep(items=[{"index": target["index"], "label": "gulab_jamun"}])
        renamed = next(row for row in meal["items"] if row["classified_label"] == "gulab_jamun")
        self.assertEqual(renamed["display_name"], "Gulab Jamun")
        self.assertFalse(renamed["low_confidence"])
        self.assertAlmostEqual(
            renamed["calories"], 336 * renamed["estimated_weight_g"] / 100, delta=1.0
        )

    def test_a_rename_to_a_countable_food_gets_a_count(self):
        target = self.draft["items"][0]
        meal = self.deep(items=[{"index": target["index"], "label": "samosa", "piece_count": 3}])
        row = next(item for item in meal["items"] if item["classified_label"] == "samosa")
        self.assertEqual(row["piece_count"], 3)
        self.assertAlmostEqual(row["estimated_weight_g"], 3 * 65.0, delta=0.5)

    def test_a_hand_added_item_gets_a_nominal_portion(self):
        meal = self.deep(items=[{"index": -1, "label": "dal_tadka"}])
        added = next(row for row in meal["items"] if row["classified_label"] == "dal_tadka")
        self.assertEqual(added["geometry"]["method"], "nominal-portion")
        self.assertEqual(added["detected_label"], "user_added")
        self.assertTrue(added["weight_estimated"])
        self.assertGreater(added["estimated_weight_g"], 0.0)
        self.assertIsNone(added["bbox"])

    def test_the_plate_size_the_user_enters_is_the_one_that_is_used(self):
        meal = self.deep(plate_diameter_cm=32.0)
        self.assertAlmostEqual(meal["plate_diameter_cm"], 32.0, places=1)

    def test_a_bigger_plate_means_bigger_measured_portions(self):
        """The plate is the only ruler in the photo, so it sets every measured weight.

        Both diameters are deliberately mid-range. Below roughly 22 cm this
        sample's regions fall under `WEIGHT_BOUNDS`' lower edge and every weight
        clamps to the same floor, which would compare two constants and prove
        nothing.
        """
        small = self.deep(plate_diameter_cm=26.0)
        self.draft = self.scan()
        self.draft_id = self.draft["draft_id"]
        large = self.deep(plate_diameter_cm=34.0)

        def measured_weight(meal: dict) -> float:
            return sum(
                row["estimated_weight_g"]
                for row in meal["items"]
                if row["piece_count"] is None and not row["weight_estimated"]
            )

        self.assertGreater(measured_weight(small), 0.0, "nothing was measured at 26 cm")
        self.assertGreater(measured_weight(large), measured_weight(small))

    def test_notes_survive_the_review(self):
        self.draft = self.scan(notes="from the scan")
        self.draft_id = self.draft["draft_id"]
        self.assertEqual(self.draft["notes"], "from the scan")
        self.assertEqual(self.deep()["notes"], "from the scan")
        self.draft = self.scan(notes="from the scan")
        self.draft_id = self.draft["draft_id"]
        self.assertEqual(self.deep(notes="edited at review")["notes"], "edited at review")

    def test_a_draft_is_private_to_its_owner(self):
        _, other = self.client.post("/api/auth/guest").json()["token"], None
        _, other_headers = self.guest()
        response = self.client.post(
            f"/api/meals/{self.draft_id}/analyze",
            json={"plate_diameter_cm": 26.0, "items": []},
            headers=other_headers,
        )
        self.assertEqual(response.status_code, 404)

    def test_an_unknown_draft_is_404(self):
        response = self.client.post(
            "/api/meals/does-not-exist/analyze",
            json={"plate_diameter_cm": 26.0, "items": []},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 404)

    def test_a_draft_is_consumed_by_its_analysis(self):
        self.deep()
        response = self.client.post(
            f"/api/meals/{self.draft_id}/analyze",
            json={"plate_diameter_cm": 26.0, "items": []},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 404)

    def test_the_reviewed_meal_is_readable_afterwards(self):
        meal = self.deep()
        body = self.client.get(f"/api/meals/{meal['meal_id']}", headers=self.headers).json()
        self.assertEqual(body["meal_id"], meal["meal_id"])
        self.assertEqual(len(body["items"]), len(meal["items"]))
        self.assertIsNotNone(body["image_url"])

    def test_the_draft_plate_size_is_range_checked(self):
        for diameter in (2.0, 400.0):
            response = self.client.post(
                f"/api/meals/{self.draft_id}/analyze",
                json={"plate_diameter_cm": diameter, "items": []},
                headers=self.headers,
            )
            self.assertEqual(response.status_code, 422, diameter)

    def test_a_counted_item_ignores_plate_recalibration(self):
        """Four samosas weigh the same on any plate — the count wasn't measured."""
        meal, before = self.count_first_item(3)

        body = self.client.patch(
            f"/api/meals/{meal['meal_id']}/plate",
            json={"plate_diameter_cm": 34.0},
            headers=self.headers,
        ).json()
        after = next(row for row in body["items"] if row["id"] == before["id"])
        self.assertAlmostEqual(after["estimated_weight_g"], before["estimated_weight_g"], delta=0.1)
        self.assertEqual(after["piece_count"], 3)

    def test_correcting_a_counted_item_s_weight_drops_the_count(self):
        """A hand-set weight and a piece count cannot both be true; the count goes."""
        meal, target = self.count_first_item(4)

        body = self.client.patch(
            f"/api/meals/{meal['meal_id']}/items/{target['id']}",
            json={"estimated_weight_g": 180.0},
            headers=self.headers,
        ).json()
        corrected = next(row for row in body["items"] if row["id"] == target["id"])
        self.assertAlmostEqual(corrected["estimated_weight_g"], 180.0)
        self.assertIsNone(corrected["piece_count"])


class TestMealLifecycle(ApiTestCase):
    def setUp(self):
        self.token, self.headers = self.guest()
        self.meal = self.client.post(
            "/api/meals/analyze",
            files={"image": ("thali.jpg", sample_bytes(), "image/jpeg")},
            headers=self.headers,
        ).json()
        self.meal_id = self.meal["meal_id"]

    def test_read_back_matches_what_analyze_returned(self):
        body = self.client.get(f"/api/meals/{self.meal_id}", headers=self.headers).json()
        self.assertEqual(body["meal_id"], self.meal_id)
        self.assertEqual(len(body["items"]), len(self.meal["items"]))
        self.assertAlmostEqual(body["totals"]["calories"], self.meal["totals"]["calories"], delta=0.5)

    def test_a_meal_is_private_to_its_owner(self):
        _, other = self.guest()
        self.assertEqual(self.client.get(f"/api/meals/{self.meal_id}", headers=other).status_code, 404)
        self.assertEqual(self.client.get(f"/api/meals/{self.meal_id}").status_code, 401)

    def test_unknown_meal_is_404(self):
        self.assertEqual(self.client.get("/api/meals/does-not-exist", headers=self.headers).status_code, 404)

    def test_correcting_a_label_recomputes_the_nutrition(self):
        item = self.meal["items"][0]
        response = self.client.patch(
            f"/api/meals/{self.meal_id}/items/{item['id']}",
            json={"classified_label": "gulab_jamun"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        corrected = next(row for row in response.json()["items"] if row["id"] == item["id"])
        self.assertEqual(corrected["display_name"], "Gulab Jamun")
        self.assertTrue(corrected["user_corrected"])
        self.assertFalse(corrected["low_confidence"])
        self.assertEqual(corrected["confidence"], 1.0)
        # 336 kcal/100 g at the item's unchanged weight.
        self.assertAlmostEqual(
            corrected["calories"], 336 * corrected["estimated_weight_g"] / 100, delta=1.0
        )

    def test_correcting_a_weight_rescales_only_that_item(self):
        items = self.meal["items"]
        target, other = items[0], items[-1]
        body = self.client.patch(
            f"/api/meals/{self.meal_id}/items/{target['id']}",
            json={"estimated_weight_g": 200.0},
            headers=self.headers,
        ).json()
        changed = next(row for row in body["items"] if row["id"] == target["id"])
        untouched = next(row for row in body["items"] if row["id"] == other["id"])
        self.assertAlmostEqual(changed["estimated_weight_g"], 200.0)
        self.assertAlmostEqual(untouched["calories"], other["calories"], delta=0.5)

    def test_a_correction_updates_the_meal_totals(self):
        item = self.meal["items"][0]
        body = self.client.patch(
            f"/api/meals/{self.meal_id}/items/{item['id']}",
            json={"estimated_weight_g": 400.0},
            headers=self.headers,
        ).json()
        self.assertAlmostEqual(
            body["totals"]["calories"], sum(row["calories"] for row in body["items"]), delta=0.5
        )
        self.assertGreater(body["totals"]["calories"], self.meal["totals"]["calories"])

    def test_an_empty_correction_is_rejected(self):
        item = self.meal["items"][0]
        response = self.client.patch(
            f"/api/meals/{self.meal_id}/items/{item['id']}", json={}, headers=self.headers
        )
        self.assertEqual(response.status_code, 400)

    def test_correcting_someone_else_s_item_is_404(self):
        _, other = self.guest()
        response = self.client.patch(
            f"/api/meals/{self.meal_id}/items/{self.meal['items'][0]['id']}",
            json={"estimated_weight_g": 100.0},
            headers=other,
        )
        self.assertEqual(response.status_code, 404)

    def test_correcting_an_item_from_another_meal_is_404(self):
        response = self.client.patch(
            f"/api/meals/{self.meal_id}/items/not-in-this-meal",
            json={"estimated_weight_g": 100.0},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 404)

    def test_plate_recalibration_agrees_with_analysing_at_that_plate_size(self):
        """One photo, one stated plate size, one answer — whichever route got there.

        A plate correction and a fresh analysis at the corrected size are the
        same measurement, so they have to produce the same weight. They did not:
        this handler used to scale finished weights by the cube of the diameter,
        while the forward pipeline damps the footprint→depth prior
        (`depth.AREA_DEPTH_EXPONENT`) and clips it, which lands nearer d^2.3. On
        a real plate at 26 → 34 cm that gap was 20%, and which number the user
        saw depended on whether they had set the plate size before or after the
        analysis.
        """
        target = 34.0
        patched = self.client.patch(
            f"/api/meals/{self.meal_id}/plate",
            json={"plate_diameter_cm": target},
            headers=self.headers,
        ).json()
        self.assertAlmostEqual(patched["plate_diameter_cm"], target, places=1)
        fresh = self.analyze(plate_diameter_cm=target)

        # Same detector, same deterministic seeds, so the item lists line up.
        self.assertEqual(len(patched["items"]), len(fresh["items"]))
        for was, now in zip(patched["items"], fresh["items"]):
            self.assertEqual(was["classified_label"], now["classified_label"])
            self.assertAlmostEqual(
                was["estimated_weight_g"],
                now["estimated_weight_g"],
                delta=max(0.5, now["estimated_weight_g"] * 0.01),
                msg=f"{now['classified_label']}: recalibrated vs re-analysed",
            )
        self.assertAlmostEqual(
            patched["totals"]["calories"],
            fresh["totals"]["calories"],
            delta=max(1.0, fresh["totals"]["calories"] * 0.01),
        )

    def test_plate_recalibration_grows_portions_but_slower_than_the_cube(self):
        """A wider plate means more food — sub-linearly in depth, so under d³.

        Worth pinning rather than leaving to the agreement test above: that one
        would still pass if both routes became wrong in the same way.
        """
        previous = float(self.meal["plate_diameter_cm"])
        ratio = 1.3
        body = self.client.patch(
            f"/api/meals/{self.meal_id}/plate",
            json={"plate_diameter_cm": previous * ratio},
            headers=self.headers,
        ).json()

        before = {row["id"]: row for row in self.meal["items"]}
        compared = 0
        for row in body["items"]:
            was = before[row["id"]]["estimated_weight_g"]
            if row["weight_estimated"] or was <= 0:
                continue  # clamped, counted or hand-added: no ratio to check
            grew = row["estimated_weight_g"] / was
            self.assertGreater(grew, ratio**2.0, msg=row["classified_label"])
            self.assertLess(grew, ratio**3.0, msg=row["classified_label"])
            compared += 1
        self.assertGreater(compared, 0, "no measured item survived to be compared")

    def test_plate_recalibration_restates_whether_the_weight_was_clamped(self):
        """A stale `weight_estimated` is a lie about a number that just changed.

        The old handler scaled the weight and left the flag where it was, so an
        item squeezed onto a tiny plate until it hit the floor of its
        served-weight envelope still reported itself as measured. The UI reads
        that flag to decide whether to offer the weight slider, so the one case
        that most needs correcting was the one that hid.
        """
        shrunk = self.client.patch(
            f"/api/meals/{self.meal_id}/plate",
            json={"plate_diameter_cm": 12.0},
            headers=self.headers,
        ).json()
        floors = {
            row["id"]: row
            for row in shrunk["items"]
            if row["geometry"]["method"] == "plate-recalibrated"
        }
        self.assertTrue(floors, "nothing was re-measured")
        # A 12 cm plate is small enough to bottom every measured item out.
        self.assertTrue(
            all(row["weight_estimated"] for row in floors.values()),
            {r["classified_label"]: r["estimated_weight_g"] for r in floors.values()},
        )

        # ...and widening it again clears the flag for anything back in range.
        widened = self.client.patch(
            f"/api/meals/{self.meal_id}/plate",
            json={"plate_diameter_cm": 26.0},
            headers=self.headers,
        ).json()
        recovered = [row for row in widened["items"] if row["id"] in floors]
        self.assertTrue(
            any(not row["weight_estimated"] for row in recovered),
            "no item came back inside its envelope",
        )

    def test_plate_recalibration_leaves_user_corrections_alone(self):
        item = self.meal["items"][0]
        self.client.patch(
            f"/api/meals/{self.meal_id}/items/{item['id']}",
            json={"estimated_weight_g": 175.0},
            headers=self.headers,
        )
        body = self.client.patch(
            f"/api/meals/{self.meal_id}/plate",
            json={"plate_diameter_cm": 34.0},
            headers=self.headers,
        ).json()
        kept = next(row for row in body["items"] if row["id"] == item["id"])
        self.assertAlmostEqual(kept["estimated_weight_g"], 175.0)

    def test_plate_diameter_is_range_checked(self):
        for diameter in (2.0, 400.0):
            response = self.client.patch(
                f"/api/meals/{self.meal_id}/plate",
                json={"plate_diameter_cm": diameter},
                headers=self.headers,
            )
            self.assertEqual(response.status_code, 422, diameter)

    def test_delete_removes_the_meal_and_its_image(self):
        image_url = self.meal["image_url"]
        self.assertEqual(self.client.get(image_url).status_code, 200)

        response = self.client.delete(f"/api/meals/{self.meal_id}", headers=self.headers)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get(f"/api/meals/{self.meal_id}", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get(image_url).status_code, 404)

    def test_delete_is_owner_only(self):
        _, other = self.guest()
        self.assertEqual(self.client.delete(f"/api/meals/{self.meal_id}", headers=other).status_code, 404)


class TestHistoryAndSummary(ApiTestCase):
    def setUp(self):
        self.token, self.headers = self.guest()
        for name in ("thali", "dosa"):
            self.client.post(
                "/api/meals/analyze",
                files={"image": (f"{name}.jpg", sample_bytes(name), "image/jpeg")},
                headers=self.headers,
            )

    def test_history_lists_the_user_s_meals_newest_first(self):
        body = self.client.get("/api/users/me/history", headers=self.headers).json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(len(body["meals"]), 2)
        stamps = [row["captured_at"] for row in body["meals"]]
        self.assertEqual(stamps, sorted(stamps, reverse=True))
        entry = body["meals"][0]
        self.assertTrue(entry["thumb_url"])
        self.assertGreater(entry["item_count"], 0)
        self.assertTrue(entry["top_items"])

    def test_history_paginates(self):
        first = self.client.get(
            "/api/users/me/history", params={"limit": 1}, headers=self.headers
        ).json()
        second = self.client.get(
            "/api/users/me/history", params={"limit": 1, "offset": 1}, headers=self.headers
        ).json()
        self.assertEqual(first["total"], 2)
        self.assertEqual(len(first["meals"]), 1)
        self.assertNotEqual(first["meals"][0]["meal_id"], second["meals"][0]["meal_id"])

    def test_history_is_scoped_to_the_caller(self):
        _, other = self.guest()
        self.assertEqual(self.client.get("/api/users/me/history", headers=other).json()["total"], 0)

    def test_reading_another_user_s_history_is_forbidden(self):
        _, other = self.guest()
        me = self.client.get("/api/auth/me", headers=self.headers).json()
        response = self.client.get(f"/api/users/{me['id']}/history", headers=other)
        self.assertEqual(response.status_code, 403)

    def test_history_pagination_bounds_are_validated(self):
        self.assertEqual(
            self.client.get("/api/users/me/history", params={"limit": 0}, headers=self.headers).status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/api/users/me/history", params={"limit": 500}, headers=self.headers).status_code,
            422,
        )

    def test_summary_has_today_a_trend_and_a_goal(self):
        body = self.client.get(
            "/api/users/me/summary", params={"days": 7}, headers=self.headers
        ).json()
        self.assertEqual(len(body["trend"]), 7)
        self.assertEqual(body["trend"][-1]["date"], body["date"])
        self.assertEqual(body["meals_logged"], 2)
        self.assertEqual(body["today"]["meals"], 2)
        self.assertGreater(body["today"]["calories"], 0)
        self.assertEqual(body["streak_days"], 1)
        self.assertGreater(body["goal"]["calories"], 0)
        self.assertEqual(set(body["daily_values"]) >= {"iron_mg", "fiber_g"}, True)

    def test_summary_today_equals_the_sum_of_today_s_meals(self):
        history = self.client.get("/api/users/me/history", headers=self.headers).json()
        expected = sum(row["total_calories"] for row in history["meals"])
        body = self.client.get("/api/users/me/summary", headers=self.headers).json()
        self.assertAlmostEqual(body["today"]["calories"], expected, delta=1.0)

    def test_summary_respects_the_calorie_goal_preference(self):
        self.client.patch("/api/users/me/preferences", json={"calorie_goal": 2600}, headers=self.headers)
        body = self.client.get("/api/users/me/summary", headers=self.headers).json()
        self.assertAlmostEqual(body["goal"]["calories"], 2600.0)
        # Macro goals are derived from the calorie goal when not set explicitly.
        self.assertAlmostEqual(body["goal"]["protein_g"], round(2600 * 0.25 / 4), delta=1)

    def test_summary_timezone_offset_is_range_checked(self):
        self.assertEqual(
            self.client.get(
                "/api/users/me/summary", params={"tz_offset": 2000}, headers=self.headers
            ).status_code,
            422,
        )

    def test_a_fresh_account_summarises_to_zero_rather_than_erroring(self):
        _, other = self.guest()
        body = self.client.get("/api/users/me/summary", headers=other).json()
        self.assertEqual(body["today"]["calories"], 0.0)
        self.assertEqual(body["streak_days"], 0)
        self.assertIsNone(body["best_day"])


class TestFrontendServing(ApiTestCase):
    def test_modules_are_never_heuristically_cached(self):
        """Nothing in the module graph may be served without `no-cache`.

        Filenames carry no content hash, so a browser that guesses a freshness
        window can pair a just-fetched index.html with a module it cached hours
        ago and run a combination that was never released. The ETag still makes
        the common case a 304; what this forbids is the guess.
        """
        for path in ("/", "/src/main.js", "/src/store.js", "/src/styles/base.css"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertEqual(response.headers.get("cache-control"), "no-cache", path)

    def test_unknown_paths_fall_through_to_the_spa_shell(self):
        """History-API routing needs every client route to return index.html."""
        for path in ("/", "/history", "/results/abc", "/settings", "/review"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("text/html", response.headers["content-type"])

    def test_api_routes_are_not_swallowed_by_the_catch_all(self):
        self.assertIn("application/json", self.client.get("/api/health").headers["content-type"])

    def test_the_catch_all_cannot_escape_the_frontend_directory(self):
        response = self.client.get("/../backend/config.py")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_every_module_in_the_import_graph_is_actually_served(self):
        """Crawl index.html → main.js → every static and lazy import.

        A route registered against a page module that does not exist fails only
        when a user navigates there, and the SPA catch-all makes it fail *quietly*
        — the missing module comes back as index.html with a 200 and an HTML
        content type, so the browser rejects it as a module and the view simply
        never appears. Walking the graph over HTTP is what turns that into a
        test failure here instead of a blank screen there.
        """
        shell = self.client.get("/")
        self.assertEqual(shell.status_code, 200)

        entries = re.findall(r'<script[^>]+type="module"[^>]+src="([^"]+)"', shell.text)
        self.assertTrue(entries, "index.html declares no module entry point")

        seen: set[str] = set()
        queue = [urljoin("/", src) for src in entries]
        while queue:
            path = queue.pop()
            if path in seen:
                continue
            seen.add(path)

            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"{path} is imported but not served")
            self.assertIn(
                "javascript",
                response.headers["content-type"],
                f"{path} is imported as a module but served as "
                f"{response.headers['content-type']} — the catch-all swallowed it",
            )

            for spec in re.findall(r'(?:from|import)\s*\(?\s*["\'](\.[^"\']+)["\']', response.text):
                queue.append(urljoin(path, spec))

        # The crawl is only meaningful if it reached the lazily-imported pages,
        # which are the ones a broken route would hide. `analyzing` and `review`
        # are the two-phase flow's middle, and a user cannot reach the results at
        # all if either of them fails to load.
        for page in (
            "home",
            "analyzing",
            "review",
            "results",
            "today",
            "history",
            "auth",
            "settings",
            "method",
            "notfound",
        ):
            self.assertIn(f"/src/pages/{page}.js", seen, f"{page}.js was never reached")

    def test_every_stylesheet_the_shell_asks_for_is_served(self):
        shell = self.client.get("/")
        sheets = re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"', shell.text)
        self.assertTrue(sheets, "index.html links no stylesheet")

        for href in sheets:
            path = urljoin("/", href)
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("css", response.headers["content-type"], path)

            for spec in re.findall(r'@import\s+(?:url\()?["\']([^"\']+)["\']', response.text):
                nested = self.client.get(urljoin(path, spec))
                self.assertEqual(nested.status_code, 200, spec)
                self.assertIn("css", nested.headers["content-type"], spec)


class TestWelcomeMeal(ApiTestCase):
    """The sample meal a brand-new account is given (backend/seed.py).

    The rest of the suite runs with seeding off, so these cases flip the setting
    on around themselves rather than relying on the process default.
    """

    def setUp(self):
        self._previous = main.settings.seed_welcome_meal
        main.settings.seed_welcome_meal = True

    def tearDown(self):
        main.settings.seed_welcome_meal = self._previous

    def test_a_new_guest_starts_with_one_meal_in_history(self):
        _, headers = self.guest()
        body = self.client.get("/api/users/me/history", headers=headers).json()
        self.assertEqual(body["total"], 1)
        entry = body["meals"][0]
        self.assertTrue(entry["thumb_url"])
        self.assertEqual(entry["item_count"], len(seed.WELCOME_ITEMS))
        self.assertTrue(entry["top_items"])
        self.assertFalse(entry["has_low_confidence"])

    def test_the_seeded_thumbnail_is_actually_served(self):
        _, headers = self.guest()
        entry = self.client.get("/api/users/me/history", headers=headers).json()["meals"][0]
        response = self.client.get(entry["thumb_url"])
        self.assertEqual(response.status_code, 200)
        self.assertIn("image", response.headers["content-type"])

    def test_seeded_totals_are_the_sum_of_its_items(self):
        _, headers = self.guest()
        meal_id = self.client.get("/api/users/me/history", headers=headers).json()["meals"][0]["meal_id"]
        meal = self.client.get(f"/api/meals/{meal_id}", headers=headers).json()
        self.assertAlmostEqual(
            meal["totals"]["calories"],
            sum(item["calories"] for item in meal["items"]),
            delta=1.0,
        )
        for item in meal["items"]:
            self.assertGreater(item["calories"], 0)
            self.assertTrue(item["display_name"])

    def test_the_seeded_meal_does_not_inflate_today(self):
        """It is dated yesterday, so a new user's daily ring still reads zero."""
        _, headers = self.guest()
        body = self.client.get("/api/users/me/summary", headers=headers).json()
        self.assertEqual(body["today"]["calories"], 0)

    def test_registering_fresh_seeds_a_meal(self):
        response = self.client.post(
            "/api/auth/register",
            json={"email": "seeded@example.com", "password": "hunter2hunter2", "name": "Seed"},
        )
        self.assertEqual(response.status_code, 200)
        headers = {"Authorization": f"Bearer {response.json()['token']}"}
        self.assertEqual(self.client.get("/api/users/me/history", headers=headers).json()["total"], 1)

    def test_upgrading_a_guest_does_not_seed_a_second_meal(self):
        _, headers = self.guest()
        response = self.client.post(
            "/api/auth/register",
            json={"email": "upgraded@example.com", "password": "hunter2hunter2"},
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        upgraded = {"Authorization": f"Bearer {response.json()['token']}"}
        self.assertEqual(self.client.get("/api/users/me/history", headers=upgraded).json()["total"], 1)

    def test_seeding_can_be_switched_off(self):
        main.settings.seed_welcome_meal = False
        _, headers = self.guest()
        self.assertEqual(self.client.get("/api/users/me/history", headers=headers).json()["total"], 0)

    def test_a_missing_sample_image_does_not_block_account_creation(self):
        original = seed.SAMPLE_IMAGE
        seed.SAMPLE_IMAGE = original.with_name("does-not-exist.jpg")
        try:
            _, headers = self.guest()
            self.assertEqual(self.client.get("/api/users/me/history", headers=headers).json()["total"], 0)
        finally:
            seed.SAMPLE_IMAGE = original


if __name__ == "__main__":
    unittest.main()
