"""Contract tests between the backend and the hosted classifier service.

`model_api/` deploys from its own directory and shares no imports with
`backend/`, which is what lets the API run without torch. The cost of that
independence is two places that have to agree, and neither would raise if they
drifted:

* **Preprocessing.** A different resample filter or channel order shifts the
  input distribution off what the model was trained on. It shows up as quietly
  worse accuracy, never as an error.
* **The wire format.** A renamed field means `RemoteClassifier` reads `None`
  where a label should be, or worse, silently attaches labels to the wrong
  items.

So the first half of this file asserts the arithmetic is numerically identical,
and the second half runs the real service in-process over ASGI and has the real
client parse its response. The third half checks that every way the remote can
fail ends in a fallback rather than an error, because that is the whole reason
the fallback chain exists.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import classify  # noqa: E402
from config import settings  # noqa: E402


def _load_model_api():
    """Import `model_api/app.py` by path.

    Not on sys.path and deliberately not a package: it is a deploy unit, not a
    library. Importing it by location here is the same act as copying it into a
    Space, which is the point — the thing under test is the file that ships.
    """
    path = PROJECT_DIR / "model_api" / "app.py"
    spec = importlib.util.spec_from_file_location("_model_api_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


model_api = _load_model_api()


def _image(seed: int, size: tuple[int, int] = (137, 211)) -> Image.Image:
    """A non-square image on purpose: a square one would hide an axis swap."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


class TinyNet:
    """Stand-in for EfficientNet-B3, as in test_training.

    Repeated rather than imported because this file must be able to stand in for
    the service without pulling timm, which is not installed locally and is the
    reason the real backbone cannot be built here.
    """

    def __new__(cls, num_classes: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 6, 3, stride=8, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(6, num_classes)

            def forward(self, x):
                return self.classifier(self.features(x).flatten(1))

        return _Net()


class TestPreprocessingParity(unittest.TestCase):
    """The duplicated arithmetic must be bit-identical, not merely similar."""

    def test_constants_agree(self):
        self.assertEqual(model_api.INPUT_RESOLUTION, classify.INPUT_RESOLUTION)
        np.testing.assert_array_equal(model_api.IMAGENET_MEAN, classify._IMAGENET_MEAN)
        np.testing.assert_array_equal(model_api.IMAGENET_STD, classify._IMAGENET_STD)

    def test_tensors_are_identical(self):
        import torch

        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                image = _image(seed)
                theirs = model_api.preprocess(image)
                ours = classify._preprocess(image)
                self.assertEqual(tuple(theirs.shape), (3, classify.INPUT_RESOLUTION, classify.INPUT_RESOLUTION))
                self.assertTrue(
                    torch.equal(theirs, ours),
                    f"max abs difference {float((theirs - ours).abs().max()):.3e}",
                )

    def test_grayscale_and_rgba_inputs_are_handled_the_same(self):
        """Uploads are not all 3-channel JPEGs; a PNG with alpha is common."""
        import torch

        base = _image(7)
        for mode in ("L", "RGBA", "P"):
            with self.subTest(mode=mode):
                converted = base.convert(mode)
                self.assertTrue(
                    torch.equal(model_api.preprocess(converted), classify._preprocess(converted))
                )

    def test_client_side_resize_does_not_change_the_arithmetic(self):
        """`RemoteClassifier` sends crops already at the model's input size.

        That is only safe because PIL short-circuits a same-size resize, so the
        service's own resize becomes a no-op. If that ever stopped being true,
        every remote prediction would be made on a doubly-resampled image while
        every local one was not — a silent accuracy gap between two engines that
        are supposed to be interchangeable.
        """
        import torch

        crop = _image(11)
        pre = crop.convert("RGB").resize(
            (classify.INPUT_RESOLUTION, classify.INPUT_RESOLUTION), Image.BICUBIC
        )
        self.assertTrue(torch.equal(model_api.preprocess(pre), classify._preprocess(crop)))


class ServiceHarness(unittest.TestCase):
    """Runs `model_api` in-process with a tiny checkpoint behind it."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.tmp = Path(tempfile.mkdtemp(prefix="nutriai-modelapi-"))
        cls.labels = ["dal_tadka", "naan", "samosa"]
        net = TinyNet(len(cls.labels))
        cls.checkpoint = cls.tmp / "tiny.pt"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "classes": cls.labels,
                "version": "tiny1",
                "input_resolution": classify.INPUT_RESOLUTION,
            },
            cls.checkpoint,
        )

        # The service builds its backbone through this hook; swapping it is what
        # lets the real app.py run without timm installed.
        cls._original_backbone = model_api.build_backbone
        model_api.build_backbone = lambda num_classes: TinyNet(num_classes)
        import os

        os.environ["CHECKPOINT_PATH"] = str(cls.checkpoint)
        model_api.classifier = model_api.Classifier()
        model_api.classifier.load()
        assert model_api.classifier.ready, model_api.classifier.error

    @classmethod
    def tearDownClass(cls):
        import os
        import shutil

        model_api.build_backbone = cls._original_backbone
        os.environ.pop("CHECKPOINT_PATH", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def remote(self, **kwargs) -> classify.RemoteClassifier:
        """A `RemoteClassifier` whose transport is the service's ASGI app.

        Real multipart encoding, real routing, real JSON — everything except a
        socket. A mocked response object would have let a field-name change
        through, which is the failure this file exists to catch.

        `TestClient` rather than `httpx.ASGITransport` because the latter is
        async-only, and `RemoteClassifier` is deliberately synchronous: the
        pipeline it serves is sync, and uvicorn already runs it in a threadpool.
        `TestClient` is an `httpx.Client` that bridges to ASGI, so it drops
        straight into the same slot the real client occupies.
        """
        from fastapi.testclient import TestClient

        remote = classify.RemoteClassifier("http://model-api", **kwargs)
        remote._client = TestClient(model_api.app)
        self.addCleanup(remote.close)
        return remote


class TestWireFormat(ServiceHarness):
    def test_health_probe_reads_the_service_metadata(self):
        remote = self.remote()
        self.assertTrue(remote.probe())
        self.assertEqual(remote.version, "tiny1")
        self.assertEqual(list(remote.classes), self.labels)
        self.assertEqual(remote.input_resolution, classify.INPUT_RESOLUTION)
        self.assertIsNone(remote.last_error)

    def test_health_shape_matches_the_backend_status_block(self):
        """Both sides are read side by side when something looks wrong."""
        body = model_api.health()
        for key in (
            "status", "ready", "engine", "version", "classes", "input_resolution",
            "tta_passes", "error",
        ):
            self.assertIn(key, body)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["engine"], "efficientnet_b3")

    def test_a_plate_of_crops_comes_back_aligned_and_parsed(self):
        remote = self.remote()
        crops = [_image(seed) for seed in range(4)]
        predictions = remote.predict(crops, top_k=4)

        self.assertEqual(len(predictions), len(crops))
        for prediction in predictions:
            self.assertIn(prediction.label, self.labels)
            self.assertEqual(prediction.engine, "efficientnet_b3@remote")
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
            # top_k counts the winner, so alternatives holds the rest.
            self.assertEqual(len(prediction.alternatives), min(3, len(self.labels) - 1))
            labels = [prediction.label] + [a["label"] for a in prediction.alternatives]
            self.assertEqual(len(labels), len(set(labels)), "a label was ranked twice")
            self.assertEqual(
                [a["confidence"] for a in prediction.alternatives],
                sorted((a["confidence"] for a in prediction.alternatives), reverse=True),
                "alternatives must be ranked descending",
            )

    def test_remote_agrees_with_the_same_weights_loaded_locally(self):
        """The end-to-end point of the whole split.

        One checkpoint, two engines — hosted and in-process. If they disagree,
        then which machine happened to serve a photo changes what the user is
        told, and the fallback chain would be swapping in a different model
        rather than the same one.

        What "agree" can mean here is bounded by transport. `classify.py` encodes
        each crop as JPEG q92 before sending it (JPEG_QUALITY), a deliberate
        trade: a crop out of a 1600 px photo is ~500 KB as PNG and ~20 KB this
        way. So the remote model does not see the same pixels the local one does,
        and demanding identical labels asks for something the design never
        promised. This asserts what is actually promised: the two engines produce
        the same probability distribution to within what quantisation costs.

        Measured, not guessed. Across 8 randomly-initialised TinyNets x 200 crops
        the worst per-label confidence deviation was 0.0263, so TOLERANCE is
        0.05 — a little under 2x the observed worst case. Argmax flips happen
        only inside the band the tolerance describes: 7 of those 8 models flipped
        nothing at all, the eighth flipped 6 of 200, and the largest local top-2
        margin that ever flipped was 0.0086. Hence labels are compared only where
        the local margin clears the tolerance, ~6x that figure.

        This does not weaken the preprocessing guarantee.
        `TestPreprocessingParity.test_tensors_are_identical` asserts bit-exact
        tensors on the same input, and it is deterministic: a BILINEAR resize, a
        one-digit typo in IMAGENET_STD, a 288 px resize and a channel swap were
        each checked against it, and it catches all four — by 0.31 to 4.76 in
        tensor space, or an outright shape mismatch. That test guards the
        arithmetic; this one guards that the arithmetic is being fed the same
        weights and read back in the same class order.

        TinyNet is left unseeded on purpose. The claim is that *any* checkpoint
        behaves this way, so drawing a fresh model each run tests more of the
        space than pinning one would — and the assertions below have to hold for
        all of them, not for a lucky one.
        """
        TOLERANCE = 0.05

        self._use_local_checkpoint()
        original = classify._build_backbone
        classify._build_backbone = lambda num_classes, pretrained=True: TinyNet(num_classes)
        self.addCleanup(setattr, classify, "_build_backbone", original)

        local = classify.DishClassifier()
        local._load_checkpoint(self.checkpoint)
        remote = self.remote()

        def distribution(prediction) -> dict[str, float]:
            scores = {prediction.label: prediction.confidence}
            for alternative in prediction.alternatives:
                scores[alternative["label"]] = alternative["confidence"]
            return scores

        width = len(self.labels)
        crops = [_image(seed) for seed in range(20, 26)]
        remote_predictions = remote.predict(crops, top_k=width)
        local_predictions = local._predict_torch(crops, top_k=width)

        self.assertEqual(len(remote_predictions), len(crops))
        for index, (hosted_prediction, local_prediction) in enumerate(
            zip(remote_predictions, local_predictions)
        ):
            hosted = distribution(hosted_prediction)
            in_process = distribution(local_prediction)

            # Same class list, in whatever order each engine ranked it. A
            # checkpoint loaded with a different class ordering surfaces here.
            self.assertEqual(
                sorted(hosted), sorted(in_process), f"crop {index}: different class lists"
            )

            for label in in_process:
                self.assertAlmostEqual(
                    hosted[label],
                    in_process[label],
                    delta=TOLERANCE,
                    msg=(
                        f"crop {index}: '{label}' scored {hosted[label]:.4f} remotely and "
                        f"{in_process[label]:.4f} locally. JPEG q92 transport costs about "
                        f"0.026; this is past {TOLERANCE}, so the two engines are not "
                        f"running the same weights."
                    ),
                )

            ranked = sorted(in_process.values(), reverse=True)
            margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
            if margin > TOLERANCE:
                self.assertEqual(
                    hosted_prediction.label,
                    local_prediction.label,
                    msg=(
                        f"crop {index}: the local engine preferred "
                        f"'{local_prediction.label}' by {margin:.4f} — well clear of "
                        f"quantisation noise — yet the service answered "
                        f"'{hosted_prediction.label}'."
                    ),
                )

    def _use_local_checkpoint(self) -> None:
        for attribute, value in (
            ("classifier_checkpoint", str(self.checkpoint)),
            ("enable_torch_models", True),
        ):
            self.addCleanup(setattr, settings, attribute, getattr(settings, attribute))
            setattr(settings, attribute, value)

    def test_metadata_is_learned_late_when_the_probe_missed_it(self):
        """The Space-was-asleep-at-boot case, which is the normal one.

        `load()` used to copy the version and class list off the remote once. A
        failed startup probe therefore pinned `/api/health` to "remote" and all
        42 signature classes for the life of the process, while real predictions
        came back tagged with the checkpoint's actual version — health reporting
        one thing and the results another.
        """
        self._configure(classifier_url="http://model-api/classify", enable_torch_models=False)
        model = classify.DishClassifier()
        original = classify.RemoteClassifier.probe
        classify.RemoteClassifier.probe = lambda self: False
        model.load()
        classify.RemoteClassifier.probe = original

        # Nothing known yet, and it says so rather than guessing.
        self.assertEqual(model.version, "remote")
        self.assertEqual(len(model.classes), len(classify.CLASS_LIST))

        from fastapi.testclient import TestClient

        assert model._remote is not None
        model._remote._client = TestClient(model_api.app)
        self.addCleanup(model._remote.close)

        predictions = model.predict_crops(
            [_image(0)], coarse_labels=["dal_or_yellow"], area_fracs=[0.3]
        )
        self.assertEqual(predictions[0].engine, "efficientnet_b3@remote")
        self.assertEqual(model.version, "tiny1", "the first success should fix the reported version")
        self.assertEqual(list(model.classes), self.labels)

    def test_a_local_checkpoint_behind_a_remote_is_reported_as_a_fallback(self):
        """Both engines loaded: the remote answers, and health says so."""
        self._configure(
            classifier_url="http://model-api/classify",
            classifier_checkpoint=str(self.checkpoint),
            enable_torch_models=True,
        )
        original_backbone = classify._build_backbone
        classify._build_backbone = lambda num_classes, pretrained=True: TinyNet(num_classes)
        self.addCleanup(setattr, classify, "_build_backbone", original_backbone)
        original_probe = classify.RemoteClassifier.probe
        classify.RemoteClassifier.probe = lambda self: True
        self.addCleanup(setattr, classify.RemoteClassifier, "probe", original_probe)

        model = classify.DishClassifier()
        model.load()
        self.assertEqual(model.backend, "efficientnet_b3@remote")
        self.assertEqual(model.fallbacks, ["efficientnet_b3", "signature"])
        self.assertIsNotNone(model._model, "the local checkpoint should still be loaded")

    def _configure(self, **overrides) -> None:
        for attribute, value in overrides.items():
            self.addCleanup(setattr, settings, attribute, getattr(settings, attribute))
            setattr(settings, attribute, value)

    def test_top_k_of_one_still_yields_a_usable_prediction(self):
        remote = self.remote()
        prediction = remote.predict([_image(3)], top_k=1)[0]
        self.assertIn(prediction.label, self.labels)
        self.assertEqual(prediction.alternatives, [])

    def test_no_crops_is_not_a_request(self):
        """An empty plate must not cost a round trip — or a 422."""
        remote = self.remote()
        remote._client = None  # any HTTP attempt would now raise
        self.assertEqual(remote.predict([]), [])


class TestServiceLimits(ServiceHarness):
    """The limits `RemoteClassifier` relies on being enforced far away."""

    def client(self):
        from fastapi.testclient import TestClient

        return TestClient(model_api.app)

    @staticmethod
    def _jpeg(seed: int = 0, size: tuple[int, int] = (64, 64)) -> bytes:
        buffer = io.BytesIO()
        _image(seed, size).save(buffer, "JPEG")
        return buffer.getvalue()

    def test_too_many_images_is_rejected(self):
        files = [
            ("images", (f"c{index}.jpg", self._jpeg(index), "image/jpeg"))
            for index in range(model_api.MAX_IMAGES_PER_REQUEST + 1)
        ]
        response = self.client().post("/classify", files=files)
        self.assertEqual(response.status_code, 413)

    def test_the_limit_is_above_what_a_plate_can_hold(self):
        """Otherwise a legitimately busy plate gets a 413 and no answer."""
        self.assertGreaterEqual(model_api.MAX_IMAGES_PER_REQUEST, settings.max_items_per_plate)

    def test_a_file_that_is_not_an_image_is_a_422(self):
        response = self.client().post(
            "/classify", files=[("images", ("x.jpg", b"not a jpeg", "image/jpeg"))]
        )
        self.assertEqual(response.status_code, 422)

    def test_an_oversized_image_is_rejected(self):
        original = model_api.MAX_BYTES_PER_IMAGE
        model_api.MAX_BYTES_PER_IMAGE = 128
        self.addCleanup(setattr, model_api, "MAX_BYTES_PER_IMAGE", original)
        response = self.client().post(
            "/classify", files=[("images", ("big.jpg", self._jpeg(1), "image/jpeg"))]
        )
        self.assertEqual(response.status_code, 413)

    def test_a_token_is_enforced_when_set(self):
        original = model_api.API_TOKEN
        model_api.API_TOKEN = "s3cret"
        self.addCleanup(setattr, model_api, "API_TOKEN", original)
        files = [("images", ("c.jpg", self._jpeg(2), "image/jpeg"))]

        client = self.client()
        self.assertEqual(client.post("/classify", files=files).status_code, 401)
        self.assertEqual(
            client.post("/classify", files=files, headers={"Authorization": "Bearer wrong"}).status_code,
            401,
        )
        self.assertEqual(
            client.post("/classify", files=files, headers={"Authorization": "Bearer s3cret"}).status_code,
            200,
        )

    def test_an_unloaded_service_says_retry_rather_than_error(self):
        """503 is the contract: the caller falls back now and may succeed later.

        A 500 would read as "this request was malformed" and a 404 as "wrong
        URL", and neither is true — the weights simply are not there yet.
        """
        loaded = model_api.classifier
        model_api.classifier = model_api.Classifier()
        self.addCleanup(setattr, model_api, "classifier", loaded)
        response = self.client().post(
            "/classify", files=[("images", ("c.jpg", self._jpeg(3), "image/jpeg"))]
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client().get("/health").json()["status"], "degraded")


class TestRemoteFailureHandling(unittest.TestCase):
    """Every remote failure has to end in a fallback, never in an exception."""

    def _remote(self, handler) -> classify.RemoteClassifier:
        import httpx

        remote = classify.RemoteClassifier("http://model-api/classify", timeout_s=0.5)
        remote._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(remote.close)
        return remote

    def test_either_url_form_resolves_the_same(self):
        """People paste the Space root and people paste the endpoint."""
        for given in (
            "http://host/classify",
            "http://host",
            "http://host/",
            "http://host/classify/",
        ):
            with self.subTest(given=given):
                remote = classify.RemoteClassifier(given)
                self.assertEqual(remote.classify_url, "http://host/classify")
                self.assertEqual(remote.health_url, "http://host/health")

    def test_a_timeout_raises_remote_unavailable(self):
        import httpx

        def handler(request):
            raise httpx.ReadTimeout("too slow", request=request)

        with self.assertRaises(classify.RemoteUnavailable):
            self._remote(handler).predict([_image(0)])

    def test_a_401_raises_remote_unavailable(self):
        import httpx

        remote = self._remote(lambda request: httpx.Response(401, json={"detail": "nope"}))
        with self.assertRaises(classify.RemoteUnavailable):
            remote.predict([_image(0)])
        self.assertIn("401", remote.last_error or "")

    def test_a_result_count_mismatch_is_refused(self):
        """Fewer results than crops would attach labels to the wrong items.

        Silently wrong is worse than absent here: a plate of dal and naan coming
        back as naan and nothing looks like a plausible answer.
        """
        import httpx

        def handler(request):
            return httpx.Response(200, json={"results": [{"label": "naan", "confidence": 0.9}]})

        with self.assertRaises(classify.RemoteUnavailable):
            self._remote(handler).predict([_image(0), _image(1)])

    def test_the_breaker_opens_after_repeated_failures(self):
        import httpx

        calls = []

        def handler(request):
            calls.append(request.url)
            raise httpx.ConnectError("refused", request=request)

        remote = self._remote(handler)
        for _ in range(classify.RemoteClassifier.FAILURE_THRESHOLD):
            with self.assertRaises(classify.RemoteUnavailable):
                remote.predict([_image(0)])

        attempts = len(calls)
        self.assertFalse(remote.available, "the breaker should be open")
        with self.assertRaises(classify.RemoteUnavailable):
            remote.predict([_image(0)])
        self.assertEqual(len(calls), attempts, "an open breaker must not touch the network")

    def test_a_success_closes_the_breaker(self):
        import httpx

        state = {"fail": True}

        def handler(request):
            if state["fail"]:
                raise httpx.ConnectError("refused", request=request)
            if request.url.path == "/health":
                return httpx.Response(
                    200, json={"ready": True, "version": "v2", "classes": ["naan"]}
                )
            return httpx.Response(
                200,
                json={"version": "v2", "results": [{"label": "naan", "confidence": 0.9, "alternatives": []}]},
            )

        remote = self._remote(handler)
        with self.assertRaises(classify.RemoteUnavailable):
            remote.predict([_image(0)])
        self.assertEqual(remote._failures, 1)

        state["fail"] = False
        remote.predict([_image(0)])
        self.assertEqual(remote._failures, 0)
        self.assertIsNone(remote.last_error)
        self.assertEqual(remote.version, "v2", "a rolled-out checkpoint should be picked up")

    def test_late_metadata_lookup_cannot_mark_a_working_remote_as_broken(self):
        """A successful prediction followed by an unreachable /health.

        The opportunistic metadata read must stay bookkeeping. Letting it write
        `last_error` meant a remote that had just answered correctly reported
        itself as failing, and `/api/health` said the engine was in trouble on
        the strength of a request whose only job was to count classes.
        """
        import httpx

        def handler(request):
            if request.url.path == "/health":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                200,
                json={"version": "v3", "results": [{"label": "naan", "confidence": 0.9, "alternatives": []}]},
            )

        remote = self._remote(handler)
        predictions = remote.predict([_image(0)])
        self.assertEqual(predictions[0].label, "naan")
        self.assertIsNone(remote.last_error)
        self.assertEqual(remote._failures, 0)
        self.assertTrue(remote.available)

    def test_a_failed_probe_does_not_disable_the_remote(self):
        """A free Space sleeps; the probe is the request that wakes it.

        Treating an unreachable probe as permanent would mean a restart during
        an idle period silently downgraded the engine until the next restart.
        """
        import httpx

        remote = self._remote(lambda request: httpx.Response(503, text="sleeping"))
        self.assertFalse(remote.probe())
        self.assertTrue(remote.available)

    def test_reachable_but_unweighted_is_reported_not_retried(self):
        import httpx

        remote = self._remote(
            lambda request: httpx.Response(
                200, json={"ready": False, "error": "No checkpoint.", "version": "unloaded"}
            )
        )
        self.assertFalse(remote.probe())
        self.assertEqual(remote.last_error, "No checkpoint.")


class TestFallbackChain(unittest.TestCase):
    """`DishClassifier` composition: what answers, and what it admits to."""

    def _configure(self, **overrides) -> None:
        for attribute, value in overrides.items():
            self.addCleanup(setattr, settings, attribute, getattr(settings, attribute))
            setattr(settings, attribute, value)

    def test_a_configured_remote_becomes_the_primary_engine(self):
        import httpx

        self._configure(classifier_url="http://model-api/classify", classifier_token="t")
        model = classify.DishClassifier()
        original = classify.RemoteClassifier.probe
        classify.RemoteClassifier.probe = lambda self: True
        self.addCleanup(setattr, classify.RemoteClassifier, "probe", original)
        model.load()

        self.assertEqual(model.backend, "efficientnet_b3@remote")
        self.assertTrue(model.is_trained_model)
        self.assertEqual(model.fallbacks, ["signature"])
        assert model._remote is not None
        self.assertEqual(model._remote._headers, {"Authorization": "Bearer t"})

    def test_a_dead_remote_falls_through_to_the_signature_prior(self):
        """The user-facing promise: a photo always gets an answer."""
        import httpx

        self._configure(classifier_url="http://model-api/classify")
        model = classify.DishClassifier()
        original = classify.RemoteClassifier.probe
        classify.RemoteClassifier.probe = lambda self: False
        self.addCleanup(setattr, classify.RemoteClassifier, "probe", original)
        model.load()

        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        assert model._remote is not None
        model._remote._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(model._remote.close)

        predictions = model.predict_crops(
            [_image(0), _image(1)],
            coarse_labels=["dal_or_yellow", "flatbread"],
            area_fracs=[0.3, 0.2],
        )
        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertEqual(prediction.engine, "signature")
            self.assertIn(prediction.label, classify.CLASS_LIST)

    def test_no_url_means_no_remote_at_all(self):
        self._configure(classifier_url="", enable_torch_models=False)
        model = classify.DishClassifier()
        model.load()
        self.assertIsNone(model._remote)
        self.assertEqual(model.backend, "signature")
        self.assertEqual(model.fallbacks, [])
        self.assertFalse(model.is_trained_model)

    def test_predict_crop_and_predict_crops_agree(self):
        """The single-crop wrapper is used by callers that only have one item."""
        self._configure(classifier_url="", enable_torch_models=False)
        model = classify.DishClassifier()
        model.load()
        crop = _image(5)
        one = model.predict_crop(crop, coarse_label="flatbread", area_frac=0.25)
        many = model.predict_crops(
            [crop], coarse_labels=["flatbread"], area_fracs=[0.25]
        )[0]
        self.assertEqual(one.label, many.label)
        self.assertEqual(one.confidence, many.confidence)

    def test_an_empty_plate_returns_no_predictions(self):
        self._configure(classifier_url="", enable_torch_models=False)
        model = classify.DishClassifier()
        model.load()
        self.assertEqual(model.predict_crops([], coarse_labels=[], area_fracs=[]), [])


if __name__ == "__main__":
    unittest.main()
