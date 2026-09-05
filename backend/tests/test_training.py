"""Training pipeline and dataset-assembly tests.

The expensive parts of `classify.train()` are the backbone and the data. Neither
is what can break silently: EfficientNet-B3 either builds or raises. What *can*
break silently is the contract around it — the checkpoint payload shape, whether
a trained checkpoint actually displaces the signature prior, whether the label
map still points at classes the nutrition layer can resolve, and whether the
threaded loader stays deterministic. Those are what these tests pin down, with a
tiny stand-in network so the whole file runs on CPU in seconds.
"""

from __future__ import annotations

import importlib.util
import contextlib
import io
import itertools
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import classify  # noqa: E402
import nutrition  # noqa: E402
from config import settings  # noqa: E402


def _load_tool(name: str):
    """Import a script from tools/ — not a package, so no plain import."""
    path = BACKEND_DIR / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_kaggle = _load_tool("train_kaggle")


def _write_image(path: Path, seed: int, size: int = 48) -> None:
    """A small deterministic image. Content is irrelevant; only decodability is."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(path, "JPEG", quality=80)


class TinyNet:
    """Stand-in for EfficientNet-B3.

    Same interface, ~200 parameters instead of 12 million. It has a `classifier`
    attribute because `_freeze_early_layers` keys off that name, so the freezing
    path gets exercised rather than bypassed.
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


class TestLabelMap(unittest.TestCase):
    """The alias tables are the one place a typo silently produces a model whose
    labels nothing downstream can resolve."""

    def test_every_alias_target_is_a_real_class(self):
        for table_name in ("ALIASES", "LOOSE_ALIASES", "LOOSE_PREFIXES"):
            table = getattr(train_kaggle, table_name)
            for source, target in table.items():
                with self.subTest(table=table_name, source=source):
                    self.assertIn(target, classify.CLASS_LIST)

    def test_every_alias_target_has_nutrition_data(self):
        """A class the model can predict but nutrition cannot price is useless:
        the pipeline would resolve it and then have nothing to scale."""
        for table_name in ("ALIASES", "LOOSE_ALIASES", "LOOSE_PREFIXES"):
            for source, target in getattr(train_kaggle, table_name).items():
                with self.subTest(source=source):
                    self.assertIn(target, nutrition.COMPOSITION)

    def test_normalise_folds_real_world_folder_names(self):
        cases = {
            "Butter Naan": "butter_naan",
            "butter-naan": "butter_naan",
            "  Chana   Masala  ": "chana_masala",
            "Apple Braeburn": "apple_braeburn",
            "masala_dosa": "masala_dosa",
        }
        for raw, expected in cases.items():
            self.assertEqual(train_kaggle.normalise(raw), expected)

    def test_exact_and_alias_resolution(self):
        self.assertEqual(train_kaggle.resolve_label("samosa", loose=False)[0], "samosa")
        self.assertEqual(train_kaggle.resolve_label("Chana Masala", loose=False)[0], "chole_masala")
        self.assertEqual(train_kaggle.resolve_label("pizza", loose=False)[0], "pizza_slice")
        self.assertEqual(train_kaggle.resolve_label("rajma_curry", loose=False)[0], "rajma_masala")
        self.assertEqual(train_kaggle.resolve_label("pavbhaji", loose=False)[0], "pav_bhaji")
        self.assertEqual(
            train_kaggle.resolve_label("spaghetti_bolognese", loose=False)[0], "pasta_red_sauce"
        )

    def test_excluded_labels_stay_excluded_even_when_loose(self):
        """`biryani` cannot be split into veg vs chicken from a folder name, so it
        must never train either class — including under --loose."""
        for name in ("biryani", "fried_rice", "chole_bhature"):
            for loose in (False, True):
                label, reason = train_kaggle.resolve_label(name, loose=loose)
                with self.subTest(name=name, loose=loose):
                    self.assertIsNone(label)
                    self.assertIn("excluded", reason)

    def test_one_plate_cannot_resolve_two_ways_because_of_its_spelling(self):
        """Every romanisation of chole bhature is excluded, not just the one spelling.

        `chana_batura` used to sit in ALIASES pointing at `chole_masala` while
        `chole_bhature` was excluded as a two-dish plate. Both names describe the
        same plate, so which verdict applied came down to the romanisation a
        dataset happened to pick — and the mapped spelling taught the fried bread
        as part of the curry, which is what the exclusion exists to prevent. The
        table-overlap test could not see this: the name was in one table only.
        """
        for name in ("chole_bhature", "chana_batura", "chana_bhatura", "chole_batura"):
            for loose in (False, True):
                with self.subTest(name=name, loose=loose):
                    label, reason = train_kaggle.resolve_label(name, loose=loose)
                    self.assertIsNone(label, f"{name} resolved to {label}")
                    self.assertIn("excluded", reason)
        # The bread still has a home of its own, so excluding the combined plate
        # costs no bread images.
        self.assertEqual(train_kaggle.resolve_label("bhatura", loose=False)[0], "poori")

    def test_loose_aliases_are_gated(self):
        strict_label, strict_reason = train_kaggle.resolve_label("dal_makhani", loose=False)
        self.assertIsNone(strict_label)
        self.assertIn("--loose", strict_reason)
        self.assertEqual(train_kaggle.resolve_label("dal_makhani", loose=True)[0], "dal_tadka")

    def test_near_aliases_need_no_flag(self):
        """A near alias is not a trade, so it must not sit behind --loose.

        `--loose` means "I accept a known calorie error". These entries carry
        none worth flagging, so gating them would push people toward the flag
        that also enables the -82% ones.
        """
        for folder, expected in train_kaggle.NEAR_ALIASES.items():
            with self.subTest(folder=folder):
                label, reason = train_kaggle.resolve_label(folder, loose=False)
                self.assertEqual(label, expected)
                self.assertEqual(reason, "near alias")

    def test_lossy_aliases_stay_behind_the_flag(self):
        """The entries whose calorie error is large must keep needing --loose."""
        for folder, expected in train_kaggle.LOOSE_ALIASES.items():
            with self.subTest(folder=folder):
                self.assertIsNone(train_kaggle.resolve_label(folder, loose=False)[0])
                self.assertEqual(train_kaggle.resolve_label(folder, loose=True)[0], expected)

    def test_alias_tables_do_not_overlap(self):
        """One folder name, one route. Overlap would make order decide the answer."""
        tables = {
            "ALIASES": set(train_kaggle.ALIASES),
            "NEAR_ALIASES": set(train_kaggle.NEAR_ALIASES),
            "LOOSE_ALIASES": set(train_kaggle.LOOSE_ALIASES),
            "EXCLUDED": set(train_kaggle.EXCLUDED),
        }
        for a, b in itertools.combinations(tables, 2):
            with self.subTest(pair=(a, b)):
                self.assertEqual(tables[a] & tables[b], set())

    def test_the_expensive_loose_aliases_are_all_accounted_for(self):
        """The four that would most distort a calorie report must be gated.

        Named individually rather than checked by rule, so that moving any one of
        them into NEAR_ALIASES has to be a deliberate edit to this test.
        """
        for folder in ("caesar_salad", "greek_salad", "misti_doi", "rasgulla",
                       "dal_makhani", "chicken_wings"):
            with self.subTest(folder=folder):
                self.assertIn(folder, train_kaggle.LOOSE_ALIASES)

    def test_loose_prefix_only_applies_when_loose(self):
        self.assertIsNone(train_kaggle.resolve_label("Apple Braeburn", loose=False)[0])
        self.assertEqual(train_kaggle.resolve_label("Apple Braeburn", loose=True)[0], "apple")

    def test_loose_prefix_collapses_cultivars(self):
        """A cultivar name after the fruit is still that fruit."""
        for name in ("Apple Braeburn", "Apple Golden 1", "Apple Granny Smith",
                     "Apple Red 1", "Banana Lady Finger", "Banana Red"):
            with self.subTest(name=name):
                label, reason = train_kaggle.resolve_label(name, loose=True)
                self.assertEqual(label, "apple" if name.startswith("Apple") else "banana")
                self.assertEqual(reason, "loose prefix")

    def test_loose_prefix_refuses_preparations(self):
        """`apple_pie` must not become `apple`.

        This is the expensive kind of mistake: Food-101 ships ~1000 apple_pie
        images, so an unbounded prefix would both teach the classifier that
        pastry is fruit and report every pie at 52 kcal/100 g instead of ~265.
        """
        for name in ("apple_pie", "apple_crumble", "apple_juice", "banana_bread",
                     "banana_split", "banana_ice_cream", "Apple Tart"):
            with self.subTest(name=name):
                label, reason = train_kaggle.resolve_label(name, loose=True)
                self.assertIsNone(label, f"{name} leaked into {label}")
                self.assertIn("not the fruit", reason)

    def test_prefix_dish_tails_do_not_shadow_a_real_class(self):
        """The deny-list must not block a folder that has a legitimate route."""
        for tail in train_kaggle.PREFIX_DISH_TAILS:
            with self.subTest(tail=tail):
                label, _ = train_kaggle.resolve_label(tail, loose=True)
                if label is not None:
                    # Fine — it resolved on its own name, before the prefix rule.
                    self.assertNotIn(label, set(train_kaggle.LOOSE_PREFIXES.values()))


class TestDatasetAssembly(unittest.TestCase):
    """Discovery has to survive the layout variety of real Kaggle datasets."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="nutriai-ds-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        layout = {
            # Food-101 style: images/<class>/
            "food101/images/samosa": 6,
            "food101/images/pizza": 6,
            # split folders that must merge into one class, not become classes
            "indian/Food Classification/train/butter_naan": 4,
            "indian/Food Classification/test/butter_naan": 3,
            # a class that should be dropped for being too small
            "indian/Food Classification/train/palak_paneer": 2,
            # an excluded name
            "indian/Food Classification/train/biryani": 5,
        }
        seed = 0
        for relative, count in layout.items():
            for index in range(count):
                _write_image(self.root / relative / f"{index}.jpg", seed)
                seed += 1

    def test_split_folders_merge_into_one_class(self):
        buckets, _ = train_kaggle.collect([self.root], loose=False)
        self.assertEqual(len(buckets["naan"]), 7, "train/ and test/ should merge, not split")
        self.assertNotIn("train", buckets)
        self.assertNotIn("test", buckets)

    def test_excluded_source_is_reported_with_a_reason(self):
        _, unmapped = train_kaggle.collect([self.root], loose=False)
        reasons = {name: reason for name, _, reason in unmapped}
        self.assertIn("biryani", reasons)
        self.assertIn("ambiguous", reasons["biryani"])

    def test_balance_drops_thin_classes_and_caps_fat_ones(self):
        buckets, _ = train_kaggle.collect([self.root], loose=False)
        kept, dropped = train_kaggle.balance(buckets, minimum=4, maximum=5, seed=1)
        self.assertEqual(dropped, {"palak_paneer": 2})
        self.assertEqual(len(kept["samosa"]), 5, "should be capped at maximum")
        self.assertEqual(len(kept["naan"]), 5)

    def test_balance_is_deterministic_for_a_seed(self):
        buckets, _ = train_kaggle.collect([self.root], loose=False)
        first, _ = train_kaggle.balance(buckets, minimum=1, maximum=3, seed=99)
        second, _ = train_kaggle.balance(buckets, minimum=1, maximum=3, seed=99)
        self.assertEqual(first, second)

    def test_build_tree_produces_a_discoverable_imagefolder(self):
        buckets, _ = train_kaggle.collect([self.root], loose=False)
        kept, _ = train_kaggle.balance(buckets, minimum=4, maximum=99, seed=1)
        out = self.root / "processed"
        linked = train_kaggle.build_tree(kept, out)

        self.assertEqual(linked, sum(len(v) for v in kept.values()))
        samples, class_names = classify._discover_dataset(out)
        self.assertEqual(class_names, sorted(kept))
        self.assertEqual(len(samples), linked)
        # Links must resolve to readable images, or training dies one epoch in.
        for path, _ in samples[:5]:
            with Image.open(path) as handle:
                handle.verify()

    def test_build_tree_replaces_a_previous_run(self):
        buckets, _ = train_kaggle.collect([self.root], loose=False)
        kept, _ = train_kaggle.balance(buckets, minimum=4, maximum=99, seed=1)
        out = self.root / "processed"
        train_kaggle.build_tree(kept, out)
        stale = out / "ghost_class"
        stale.mkdir()
        (stale / "x.jpg").touch()
        train_kaggle.build_tree(kept, out)
        self.assertFalse(stale.exists(), "a stale class folder would become a phantom class")

    def test_build_tree_resolves_relative_source_paths(self):
        """CLI paths are often relative to the project root."""
        source = self.root / "source.jpg"
        _write_image(source, 123)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            out = self.root / "relative-processed"
            linked = train_kaggle.build_tree({"samosa": [Path("source.jpg")]}, out)
            self.assertEqual(linked, 1)
            target = out / "samosa" / "00000.jpg"
            self.assertTrue(target.exists())
            with Image.open(target) as image:
                image.verify()
        finally:
            os.chdir(previous)


class TestLoaderDeterminism(unittest.TestCase):
    """The threaded loader must not make a run depend on thread scheduling."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="nutriai-loader-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.rows = []
        for index in range(9):
            path = self.root / f"{index}.jpg"
            _write_image(path, seed=index)
            self.rows.append((path, index % 3))

    def _stack(self, workers: int):
        import torch

        batches = list(
            classify._batches(self.rows, 4, augment=True, seed=5, workers=workers)
        )
        return torch.cat([inputs for inputs, _ in batches]), [
            label for _, targets in batches for label in targets.tolist()
        ]

    def test_augmentation_is_independent_of_worker_count(self):
        import torch

        single, labels_single = self._stack(1)
        many, labels_many = self._stack(6)
        self.assertEqual(labels_single, labels_many)
        self.assertTrue(
            torch.allclose(single, many),
            "per-sample RNG should make augmentation reproducible regardless of pool size",
        )

    def test_unreadable_images_are_skipped_not_fatal(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        rows = self.rows + [(broken, 0)]
        total = sum(int(inputs.shape[0]) for inputs, _ in classify._batches(rows, 4, augment=False, seed=1))
        self.assertEqual(total, len(self.rows))

    def test_batches_cover_every_row_exactly_once(self):
        seen = sum(
            int(inputs.shape[0]) for inputs, _ in classify._batches(self.rows, 4, augment=False, seed=1)
        )
        self.assertEqual(seen, len(self.rows))


class TestParamGroups(unittest.TestCase):
    """v1 froze 70% of the network and underfit. These pin the replacement."""

    def groups(self, **kwargs):
        defaults = {"base_lr": 1e-3, "weight_decay": 0.01, "layer_decay": 0.75}
        return classify._param_groups(TinyNet(3), **{**defaults, **kwargs})

    def test_every_trainable_parameter_gets_exactly_one_group(self):
        model = TinyNet(3)
        groups = classify._param_groups(
            model, base_lr=1e-3, weight_decay=0.01, layer_decay=0.75
        )
        listed = [param for group in groups for param in group["params"]]
        expected = [param for param in model.parameters() if param.requires_grad]
        self.assertEqual(len(listed), len(expected))
        # Identity, not equality: a parameter counted twice would be updated twice.
        self.assertEqual({id(p) for p in listed}, {id(p) for p in expected})

    def test_shallow_layers_get_a_smaller_rate_than_the_head(self):
        groups = self.groups()
        self.assertLess(groups[0]["lr"], groups[-1]["lr"])
        self.assertAlmostEqual(groups[-1]["lr"], 1e-3)

    def test_layer_decay_of_one_gives_every_group_the_same_rate(self):
        rates = {group["lr"] for group in self.groups(layer_decay=1.0)}
        self.assertEqual(rates, {1e-3})

    def test_norms_and_biases_are_spared_the_weight_decay(self):
        """A bias has no scale-invariance for decay to exploit, so shrinking it
        does not regularise — it just pulls the layer off centre."""
        for group in self.groups():
            with self.subTest(param=group["name"]):
                one_dimensional = group["params"][0].ndim <= 1
                self.assertEqual(group["weight_decay"], 0.0 if one_dimensional else 0.01)

    def test_freezing_keeps_parameters_out_of_the_optimiser(self):
        model = TinyNet(3)
        classify._freeze_early_layers(model, keep_fraction=0.5)
        groups = classify._param_groups(
            model, base_lr=1e-3, weight_decay=0.01, layer_decay=0.75
        )
        listed = {id(param) for group in groups for param in group["params"]}
        for name, param in model.named_parameters():
            with self.subTest(param=name):
                self.assertEqual(param.requires_grad, id(param) in listed)

    def test_the_default_freeze_fraction_trains_everything(self):
        """`freeze_fraction=0.0` is the whole point of the change, so it is pinned
        here rather than left to the caller to get right."""
        model = TinyNet(3)
        frozen, trainable = classify._freeze_early_layers(model, keep_fraction=1.0)
        self.assertEqual(frozen, 0)
        self.assertEqual(trainable, len(list(model.parameters())))

    def test_a_fully_frozen_model_is_refused_rather_than_trained(self):
        model = TinyNet(3)
        for param in model.parameters():
            param.requires_grad = False
        with self.assertRaises(ValueError):
            classify._param_groups(model, base_lr=1e-3, weight_decay=0.01, layer_decay=0.75)


class TestMixBatch(unittest.TestCase):
    """Mixup and CutMix, and the `lam` the loss is weighted by."""

    def batch(self, size=4, side=8):
        import torch

        # One constant value per sample, so any pixel can be traced to its source.
        inputs = torch.stack(
            [torch.full((3, side, side), float(index)) for index in range(size)]
        )
        return inputs, torch.arange(size)

    def test_a_batch_is_left_alone_when_the_probability_is_zero(self):
        import torch

        inputs, targets = self.batch()
        out, a, b, lam = classify._mix_batch(
            inputs, targets, mixup_alpha=0.2, cutmix_alpha=1.0,
            mix_prob=0.0, rng=random.Random(0),
        )
        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(out, inputs))
        self.assertTrue(torch.equal(a, b))

    def test_a_single_example_batch_cannot_be_mixed(self):
        inputs, targets = self.batch(size=1)
        _, _, _, lam = classify._mix_batch(
            inputs, targets, mixup_alpha=0.2, cutmix_alpha=1.0,
            mix_prob=1.0, rng=random.Random(0),
        )
        self.assertEqual(lam, 1.0)

    def test_mixup_leaves_every_pixel_between_its_two_sources(self):
        import torch

        inputs, targets = self.batch()
        out, _, _, lam = classify._mix_batch(
            inputs, targets, mixup_alpha=0.4, cutmix_alpha=0.0,
            mix_prob=1.0, rng=random.Random(3),
        )
        self.assertTrue(0.0 <= lam <= 1.0)
        low, high = inputs.min(), inputs.max()
        self.assertGreaterEqual(out.min().item(), low.item() - 1e-5)
        self.assertLessEqual(out.max().item(), high.item() + 1e-5)
        # A convex blend of the batch cannot introduce a value outside it, and
        # cannot leave the mean of the batch behind either.
        self.assertAlmostEqual(out.mean().item(), inputs.mean().item(), places=4)
        # Seed 3 is load-bearing: it draws lam=0.732 and a permutation that moves
        # two of the four samples. A seed drawing the identity permutation, or a lam
        # within float32 epsilon of 1.0, would blend to a copy of the input and fail
        # here through no fault of the code.
        self.assertFalse(torch.equal(out, inputs), "nothing was actually mixed")

    def test_the_same_seed_mixes_the_same_way_twice(self):
        """Every draw comes from the injected rng, so the seed fixes the outcome.

        This was briefly untrue: the pairing permutation came from torch's global
        generator while lam and the rectangle came from `rng`. The suite caught it
        as a 1-in-24 flake — the identity permutation blending each sample with
        itself — but the real cost was silent. Two runs of one recipe would mix
        differently, so their loss curves could not be compared, which is the only
        reason to hold a seed fixed at all.
        """
        import torch

        for use_cutmix in (False, True):
            first, second = [
                classify._mix_batch(
                    *self.batch(),
                    mixup_alpha=0.0 if use_cutmix else 0.4,
                    cutmix_alpha=1.0 if use_cutmix else 0.0,
                    mix_prob=1.0,
                    rng=random.Random(11),
                )
                for _ in range(2)
            ]
            self.assertEqual(first[3], second[3], "lam drifted between runs")
            self.assertTrue(torch.equal(first[0], second[0]), "pixels drifted")
            self.assertTrue(torch.equal(first[2], second[2]), "pairing drifted")

    def test_cutmix_reports_the_area_it_really_pasted(self):
        """`lam` is recomputed from the clipped rectangle, not the requested one.

        The rectangle is centred on a random pixel and then clipped to the frame,
        so a corner draw delivers far less area than the beta sample asked for.
        Weighting the loss by the requested area would then credit the model for
        a label whose evidence was mostly cropped away.
        """
        import torch

        side = 8
        for seed in range(40):
            inputs, targets = self.batch(side=side)
            out, _, _, lam = classify._mix_batch(
                inputs, targets, mixup_alpha=0.0, cutmix_alpha=1.0,
                mix_prob=1.0, rng=random.Random(seed),
            )
            with self.subTest(seed=seed):
                self.assertTrue(0.0 <= lam <= 1.0)
                expected = round((1.0 - lam) * side * side)
                for index in range(inputs.size(0)):
                    changed = int((out[index] != inputs[index]).any(dim=0).sum())
                    # 0 when the shuffle paired a sample with itself.
                    self.assertIn(changed, (0, expected))

    def test_cutmix_only_ever_moves_whole_pixels(self):
        """CutMix copies; unlike mixup it must not produce a blended value."""
        import torch

        inputs, targets = self.batch()
        out, _, _, _ = classify._mix_batch(
            inputs, targets, mixup_alpha=0.0, cutmix_alpha=1.0,
            mix_prob=1.0, rng=random.Random(11),
        )
        self.assertTrue(torch.isin(out, inputs.unique()).all())


class TestEMA(unittest.TestCase):
    """The averaged weights are an evaluation aid, so they must not leak."""

    def test_the_average_moves_towards_the_weights_it_is_fed(self):
        import torch

        model = TinyNet(3)
        ema = classify._EMA(model, decay=0.5)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
        start = {key: value.clone() for key, value in ema.shadow.items()}
        for _ in range(4):
            ema.update(model)
        for key, value in ema.shadow.items():
            if start[key].numel():
                with self.subTest(tensor=key):
                    # Closer to 1.0 than it started, but not all the way there:
                    # that is what makes it an average and not a copy.
                    self.assertLess((value - 1.0).abs().max(), (start[key] - 1.0).abs().max() + 1e-9)

    def test_applying_the_average_is_reversible(self):
        import torch

        model = TinyNet(3)
        ema = classify._EMA(model, decay=0.9)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(2.0)
        ema.update(model)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with ema.applied_to(model):
            inside = {key: value.clone() for key, value in model.state_dict().items()}
        after = model.state_dict()
        for key in before:
            with self.subTest(tensor=key):
                self.assertTrue(torch.equal(before[key], after[key]))
        self.assertTrue(
            any(not torch.equal(before[key], inside[key]) for key in before),
            "the averaged weights were identical to the live ones, so nothing was tested",
        )

    def test_the_live_weights_come_back_even_if_the_block_raises(self):
        """Leaving the average installed would make the next epoch train from it —
        an evaluation aid quietly turning into a second optimiser."""
        import torch

        model = TinyNet(3)
        ema = classify._EMA(model, decay=0.9)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(3.0)
        ema.update(model)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with ema.applied_to(model):
                raise RuntimeError("evaluation blew up")
        for key, value in model.state_dict().items():
            with self.subTest(tensor=key):
                self.assertTrue(torch.equal(before[key], value))

    def test_integer_buffers_are_not_averaged(self):
        """A decayed average of a batch counter is not a counter."""
        import torch
        import torch.nn as nn

        model = nn.Sequential(nn.BatchNorm2d(2))
        ema = classify._EMA(model, decay=0.9)
        self.assertNotIn("0.num_batches_tracked", ema.shadow)
        self.assertIn("0.running_mean", ema.shadow)


class TestWarmStart(unittest.TestCase):
    """Continuing v1 rather than restarting, across a changed class list."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="nutriai-warm-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write_checkpoint(self, classes, *, version="v1", test_top1=0.7846):
        import torch

        donor = TinyNet(len(classes))
        with torch.no_grad():
            for index in range(len(classes)):
                donor.classifier.weight[index].fill_(float(index) + 1.0)
                donor.classifier.bias[index].fill_(float(index) + 1.0)
        path = self.root / f"efficientnet_{version}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in donor.state_dict().items()},
                "classes": list(classes),
                "version": version,
                "test_top1": test_top1,
            },
            path,
        )
        return path, donor

    def test_the_shared_body_is_loaded_verbatim(self):
        import torch

        path, donor = self.write_checkpoint(["a", "b", "c"])
        model = TinyNet(3)
        report = classify._warm_start(model, path, ["a", "b", "c"])
        self.assertEqual(report["from_version"], "v1")
        self.assertEqual(report["from_test_top1"], 0.7846)
        self.assertEqual(report["tensors_skipped"], [])
        self.assertTrue(
            torch.equal(model.features[0].weight, donor.features[0].weight),
            "the convolution is class-count independent and should have loaded exactly",
        )

    def test_an_unchanged_class_list_reports_every_row_as_reused(self):
        """The head keeps its shape then, so it arrives through the verbatim branch
        and never the transplant one. Counting only transplants reported "0 of 3
        rows reused" for the case where all three were — and that log line is the
        one thing a warm start gets read to check."""
        path, _ = self.write_checkpoint(["a", "b", "c"])
        model = TinyNet(3)
        report = classify._warm_start(model, path, ["a", "b", "c"])
        self.assertEqual(report["head_rows_reused"], 3)
        self.assertEqual(report["head_rows_new"], [])

    def test_head_rows_follow_their_class_name_not_their_index(self):
        """The point of the whole exercise.

        Widening the dataset inserts labels, which shifts every index after the
        insertion. Loading the head by position would then quietly relabel most of
        what v1 learned — `samosa`'s row landing on `sambhar` — and present as a
        mysteriously bad initialisation rather than as a bug.
        """
        import torch

        path, _ = self.write_checkpoint(["a", "b", "c"])
        # "z" is new and sorts first; a, b, c all move.
        new_classes = ["z", "a", "b", "c"]
        model = TinyNet(len(new_classes))
        fresh = model.classifier.weight[0].clone()
        report = classify._warm_start(model, path, new_classes)

        self.assertEqual(report["head_rows_reused"], 3)
        self.assertEqual(report["head_rows_new"], ["z"])
        for old_index, name in enumerate(["a", "b", "c"]):
            with self.subTest(dish=name):
                row = model.classifier.weight[new_classes.index(name)]
                self.assertTrue(torch.allclose(row, torch.full_like(row, float(old_index) + 1.0)))
        self.assertTrue(
            torch.equal(model.classifier.weight[0], fresh),
            "a label v1 never saw should keep its fresh initialisation",
        )

    def test_a_dropped_class_simply_does_not_arrive(self):
        import torch

        path, _ = self.write_checkpoint(["a", "b", "c"])
        model = TinyNet(2)
        report = classify._warm_start(model, path, ["a", "c"])
        self.assertEqual(report["head_rows_reused"], 2)
        self.assertEqual(report["head_rows_new"], [])
        for expected, name in ((1.0, "a"), (3.0, "c")):
            row = model.classifier.weight[["a", "c"].index(name)]
            with self.subTest(dish=name):
                self.assertTrue(torch.allclose(row, torch.full_like(row, expected)))

    def test_a_body_that_does_not_fit_is_reported_not_fatal(self):
        """Run B changes the backbone. The head is meant to be rebuilt then, but a
        silent all-tensors-skipped load would look exactly like a successful one."""
        import torch

        path, _ = self.write_checkpoint(["a", "b", "c"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["state_dict"]["features.0.weight"] = torch.zeros(9, 3, 3, 3)
        torch.save(payload, path)

        model = TinyNet(3)
        untouched = model.features[0].weight.clone()
        report = classify._warm_start(model, path, ["a", "b", "c"])
        self.assertIn("features.0.weight", report["tensors_skipped"])
        self.assertTrue(torch.equal(model.features[0].weight, untouched))


class TestTrainToServeRoundTrip(unittest.TestCase):
    """The load-bearing contract: what `train()` writes, `DishClassifier` reads.

    This is the test that would have caught a renamed payload key, and it is the
    reason the checkpoint format is not just documented but pinned.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="nutriai-train-"))
        data = cls.root / "processed"
        cls.labels = ["dal_tadka", "naan", "samosa"]
        seed = 0
        for label in cls.labels:
            for index in range(12):
                _write_image(data / label / f"{index}.jpg", seed)
                seed += 1

        cls._original_backbone = classify._build_backbone
        cls._original_model_dir = classify.MODEL_DIR
        cls._original_project_dir = classify.PROJECT_DIR
        classify._build_backbone = lambda num_classes, pretrained=True: TinyNet(num_classes)
        classify.MODEL_DIR = cls.root / "models"
        classify.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        classify.PROJECT_DIR = cls.root

        cls.summary = classify.train(
            data, epochs=2, batch_size=6, patience=5, version="test1", device="cpu", workers=2
        )

    @classmethod
    def tearDownClass(cls):
        classify._build_backbone = cls._original_backbone
        classify.MODEL_DIR = cls._original_model_dir
        classify.PROJECT_DIR = cls._original_project_dir
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_summary_reports_what_it_trained_on(self):
        self.assertEqual(self.summary["classes"], 3)
        self.assertEqual(self.summary["images"], 36)
        self.assertEqual(self.summary["epochs_run"], 2)
        self.assertEqual(self.summary["device"], "cpu")
        self.assertEqual(set(self.summary["per_class_f1"]), set(self.labels))

    def test_checkpoint_payload_shape_is_what_inference_expects(self):
        import torch

        payload = torch.load(self.summary["checkpoint"], map_location="cpu", weights_only=False)
        for key in ("state_dict", "classes", "version", "input_resolution"):
            self.assertIn(key, payload)
        self.assertEqual(payload["classes"], self.labels)
        self.assertEqual(payload["input_resolution"], classify.INPUT_RESOLUTION)
        self.assertEqual(payload["version"], "test1")

    def test_checkpoint_tensors_are_on_cpu(self):
        """Serving hosts have no GPU; a CUDA-tensor checkpoint would need one."""
        import torch

        payload = torch.load(self.summary["checkpoint"], map_location="cpu", weights_only=False)
        for name, tensor in payload["state_dict"].items():
            with self.subTest(tensor=name):
                self.assertEqual(tensor.device.type, "cpu")

    def test_training_log_entry_is_appended(self):
        log_path = self.root / "TRAINING_LOG.md"
        self.assertTrue(log_path.is_file())
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("## test1", text)
        self.assertIn("Test top-1", text)

    def _use_checkpoint(self, path) -> None:
        """Point the settings at `path` for one test, then restore.

        `enable_torch_models` is forced on rather than assumed: `test_api` sets
        `ENABLE_TORCH_MODELS=false` in the environment at import time to keep
        itself fast, and that lands on the whole process. Inheriting it here
        would skip the checkpoint branch entirely and quietly turn both of the
        tests below into assertions about the signature prior.
        """
        for attribute, value in (
            ("classifier_checkpoint", str(path)),
            ("enable_torch_models", True),
        ):
            self.addCleanup(setattr, settings, attribute, getattr(settings, attribute))
            setattr(settings, attribute, value)

    def test_a_trained_checkpoint_displaces_the_signature_prior(self):
        self._use_checkpoint(self.summary["checkpoint"])
        model = classify.DishClassifier()
        model.load()
        self.assertEqual(model.backend, "efficientnet_b3")
        self.assertTrue(model.is_trained_model)
        self.assertEqual(model.version, "test1")
        self.assertEqual(list(model.classes), self.labels)

        crop = Image.fromarray(
            np.random.default_rng(3).integers(0, 255, (40, 40, 3), dtype=np.uint8)
        )
        prediction = model.predict_crop(crop, coarse_label="dal_or_yellow", area_frac=0.3)
        self.assertEqual(prediction.engine, "efficientnet_b3")
        self.assertIn(prediction.label, self.labels)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_a_corrupt_checkpoint_falls_back_instead_of_crashing(self):
        """A truncated download must degrade to the prior, not take the API down."""
        broken = self.root / "models" / "broken.pt"
        broken.write_bytes(b"\x00\x01\x02 not a torch file")
        self._use_checkpoint(broken)
        model = classify.DishClassifier()
        model.load()
        self.assertEqual(model.backend, "signature")
        self.assertFalse(model.is_trained_model)
        self.assertTrue(model.ready)


class TestTrainingCLI(unittest.TestCase):
    """`train()` grew knobs faster than the CLIs that drive it.

    `--device` was the expensive one: it was accepted by `train()` and not by
    `classify.py --train`, so a Kaggle run launched that way silently trained on
    CPU. Nothing raises for that — it just takes twenty hours instead of one.
    """

    def _captured(self, argv: list[str]) -> dict:
        seen: dict = {}

        def fake_train(data_dir, **kwargs):
            seen["data_dir"] = data_dir
            seen.update(kwargs)
            return {"history": [], "checkpoint": "x", "version": "x"}

        original = classify.train
        self.addCleanup(setattr, classify, "train", original)
        classify.train = fake_train
        with contextlib.redirect_stdout(io.StringIO()):  # _main prints the summary
            self.assertEqual(classify._main(argv), 0)
        return seen

    def test_every_train_keyword_is_reachable_from_the_cli(self):
        """Introspected rather than listed, so a new keyword fails here first."""
        import inspect

        keywords = {
            name
            for name, parameter in inspect.signature(classify.train).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        forwarded = set(self._captured(["--train", "--epochs", "1"]))
        self.assertEqual(keywords - forwarded, set(), "CLI does not forward these to train()")

    def test_flags_arrive_with_the_values_that_were_typed(self):
        seen = self._captured(
            [
                "--train", "--data", "/tmp/tree", "--epochs", "3", "--batch-size", "8",
                "--learning-rate", "5e-5", "--patience", "2", "--workers", "6",
                "--device", "cuda", "--version", "v9",
            ]
        )
        self.assertEqual(str(seen["data_dir"]), "/tmp/tree")
        self.assertEqual(seen["epochs"], 3)
        self.assertEqual(seen["batch_size"], 8)
        self.assertAlmostEqual(seen["learning_rate"], 5e-5)
        self.assertEqual(seen["patience"], 2)
        self.assertEqual(seen["workers"], 6)
        self.assertEqual(seen["device"], "cuda")
        self.assertEqual(seen["version"], "v9")

    def test_lr_is_still_accepted_as_a_short_form(self):
        """The old spelling stays valid — it is in notes and shell history."""
        self.assertAlmostEqual(
            self._captured(["--train", "--lr", "2e-4"])["learning_rate"], 2e-4
        )

    def test_the_two_entry_points_speak_the_same_language(self):
        """One vocabulary for both front doors.

        `tools/train_kaggle.py` assembles a dataset then trains; `classify.py
        --train` trains an already-built tree. A flag that means something in one
        should mean the same thing in the other, or a command copied from the
        readme works in one place and errors in the other.
        """
        shared = ["--epochs", "--batch-size", "--learning-rate", "--patience",
                  "--workers", "--device", "--version"]
        # Plus every recipe flag. Read off the table both parsers are built from,
        # rather than restated here: a new knob then has to appear in both front
        # doors without anyone remembering to extend this list.
        shared += [flag for flag, _ in classify.RECIPE_ARGUMENTS]
        helps = {}
        for name, runner in (("classify", classify._main), ("train_kaggle", train_kaggle.main)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    runner(["--help"])
            helps[name] = out.getvalue()
        for flag in shared:
            for name, text in helps.items():
                with self.subTest(flag=flag, entry=name):
                    self.assertIn(flag, text)

    def test_not_passing_train_prints_help_instead_of_training(self):
        """A bare `python classify.py` must not start a 20-hour job."""
        called = []
        original = classify.train
        self.addCleanup(setattr, classify, "train", original)
        classify.train = lambda *a, **k: called.append(1)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(classify._main([]), 0)
        self.assertEqual(called, [])
        self.assertIn("--train", out.getvalue())


class TestAutocastSelection(unittest.TestCase):
    def test_cpu_gets_a_no_op_context(self):
        import contextlib

        self.assertIs(classify._autocast(classify._device("cpu")), contextlib.nullcontext)

    def test_cuda_gets_a_real_autocast_factory(self):
        import torch

        factory = classify._autocast(torch.device("cuda"))
        self.assertTrue(callable(factory))
        self.assertIsNot(factory, __import__("contextlib").nullcontext)


if __name__ == "__main__":
    unittest.main()
