#!/usr/bin/env python3
"""Assemble a training set from attached Kaggle datasets, then fine-tune the classifier.

Why this file exists
--------------------
`classify.train()` already does the training. What it needs and cannot supply is
an ImageFolder tree whose folder names are *our* 42 class labels. Public food
datasets do not use our names: Food-101 calls it `spaghetti_bolognese`, we call
it `pasta_red_sauce`; the Indian sets call it `chana_masala`, we call it
`chole_masala`. This script is the translation layer, and nothing more.

It does four things:
  1. finds every ImageFolder-ish class directory under the attached datasets,
  2. maps the folder names onto `classify.CLASS_LIST` through an explicit table,
  3. builds a balanced symlink tree, printing exactly which classes it could and
     could not cover,
  4. hands that tree to `classify.train()`.

Run it on Kaggle with the GPU accelerator on:

    !pip install -q timm
    !cp -r /kaggle/input/nutriai-backend/backend /kaggle/working/backend
    %env MODEL_DIR=/kaggle/working/models
    !python /kaggle/working/backend/tools/train_kaggle.py --epochs 24

Use `--dry-run` first. It prints the coverage table without touching the GPU, and
that table is the thing worth looking at before spending an hour of quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import classify as classify_module  # noqa: E402  (after sys.path fix)
from classify import CLASS_LIST  # noqa: E402

log = logging.getLogger("nutriai.train_kaggle")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Directory names that are structure, not labels. A tree like
# `Food Classification/train/samosa/*.jpg` must yield "samosa", never "train".
STRUCTURAL_DIRS = {
    "train", "training", "test", "testing", "val", "valid", "validation",
    "images", "image", "img", "data", "dataset", "food", "input",
}


# --------------------------------------------------------------- Label mapping

# Same dish, different spelling or a strict subtype. Safe to train on directly.
ALIASES: dict[str, str] = {
    # -- Food-101 -------------------------------------------------------
    "pizza": "pizza_slice",
    "spaghetti_bolognese": "pasta_red_sauce",
    # -- breads ---------------------------------------------------------
    "butter_naan": "naan",
    "plain_naan": "naan",
    "chapati": "roti_chapati",
    "chapathi": "roti_chapati",
    "roti": "roti_chapati",
    "phulka": "roti_chapati",
    "tandoori_roti": "roti_chapati",
    "aloo_paratha": "paratha",
    "parantha": "paratha",
    "prantha": "paratha",
    "puri": "poori",
    "luchi": "poori",
    # -- south Indian ---------------------------------------------------
    "masala_dosa": "dosa",
    "plain_dosa": "dosa",
    "sada_dosa": "dosa",
    "idly": "idli",
    "vada": "medu_vada",
    "medhu_vada": "medu_vada",
    "uttapam": "dosa",
    "sambar": "sambhar",
    "saambar": "sambhar",
    "coconut_chutney": "coconut_chutney",
    # -- curries and gravies -------------------------------------------
    "chana_masala": "chole_masala",
    "chole": "chole_masala",
    "rajma": "rajma_masala",
    "rajma_curry": "rajma_masala",
    "dal_fry": "dal_tadka",
    "daal_tadka": "dal_tadka",
    "dal_tarka": "dal_tadka",
    "murgh_makhani": "butter_chicken",
    "maach_jhol": "fish_curry",
    "machher_jhol": "fish_curry",
    "anda_curry": "egg_curry",
    "egg_masala": "egg_curry",
    # -- rice -----------------------------------------------------------
    "steamed_rice": "plain_rice",
    "white_rice": "plain_rice",
    "boiled_rice": "plain_rice",
    "cumin_rice": "jeera_rice",
    "vegetable_biryani": "veg_biryani",
    "veg_pulao": "veg_biryani",
    "pavbhaji": "pav_bhaji",
    # -- sides and sweets ----------------------------------------------
    "papadum": "papad",
    "appalam": "papad",
    "curd": "curd_yogurt",
    "dahi": "curd_yogurt",
    "yogurt": "curd_yogurt",
    "phirni": "kheer",
    "chak_hao_kheer": "kheer",
    "payasam": "kheer",
    "payasa": "kheer",
    "gulab_jamoon": "gulab_jamun",
    "gulaab_jamun": "gulab_jamun",
    # -- protein --------------------------------------------------------
    "hard_boiled_egg": "boiled_egg",
    "egg_boiled": "boiled_egg",
}

# Different dish, same family, and the composition table agrees. These are not
# synonyms — nobody calls a bhatura a poori — so they are not in ALIASES. But they
# are not a trade either, which is what `--loose` exists to flag, so they do not
# belong there.
#
# An entry earns a place here only by passing both tests independently:
#   1. same dish family — a Ledikeni *is* a gulab-jamun-family fried milk sweet,
#      a chicken tikka *is* grilled marinated chicken. The label shown to the
#      user has to still be true.
#   2. composition within ±15% of the target's row. The volume-from-one-photo
#      estimate upstream carries more error than that, so below this line the
#      alias is not what limits accuracy.
#
# Anything failing either test stays in LOOSE_ALIASES with its cost written down.
# The ±15% figure is checked against published composition values, which are
# approximate — hence requiring the family test to agree rather than trusting a
# number alone.
NEAR_ALIASES: dict[str, str] = {
    # -- paneer: creamy tomato gravies, same base ------------------------
    "shahi_paneer": "paneer_butter_masala",          # -3%
    "paneer_tikka_masala": "paneer_butter_masala",   # +12%
    # -- chicken ---------------------------------------------------------
    "chicken_tikka_masala": "butter_chicken",        # +7%
    "chicken_tikka": "grilled_chicken",              # 0%
    "tandoori_chicken": "grilled_chicken",           # +11%
    # -- fried breads ----------------------------------------------------
    "bhatura": "poori",                              # -7%
    "daal_puri": "poori",                            # +9%
    # -- griddle breads --------------------------------------------------
    "misi_roti": "roti_chapati",                     # -6%
    "missi_roti": "roti_chapati",                    # -6%
    # -- vegetable dishes ------------------------------------------------
    "aloo_matar": "mixed_veg_curry",                 # -10%
    "karela_bharta": "mixed_veg_curry",              # -2%
    # -- gulab-jamun family: fried milk solids in syrup ------------------
    "ledikeni": "gulab_jamun",                       # +8%
    "lyangcha": "gulab_jamun",                       # +2%
}

# Close enough to be useful, wrong enough to be worth naming. Off by default;
# `--loose` turns them on. Every entry costs some label accuracy — dal makhani
# carries cream and butter that dal tadka does not, so its calories will read
# low — which is a fair trade for a class that would otherwise have no data at
# all, but only if it is a deliberate trade.
#
# The percentage on each line is how the app would misreport that food once it is
# taught under the target's label. Negative underreports, which is the worse
# direction for a calorie tracker: the plate looks cheaper than it was.
LOOSE_ALIASES: dict[str, str] = {
    "dal": "dal_tadka",                              # generic lentil dish
    "dal_makhani": "dal_tadka",                        # -44%  cream and butter
    "daal_makhani": "dal_tadka",                       # -44%
    "kadai_paneer": "paneer_butter_masala",            # +33%
    "matar_paneer": "paneer_butter_masala",            # +47%
    "chicken_wings": "grilled_chicken",                # -33%  skin and fat
    "makki_di_roti_sarson_da_saag": "roti_chapati",    # +20%, and a two-part plate
    "aloo_methi": "mixed_veg_curry",                   # -26%
    "aloo_shimla_mirch": "mixed_veg_curry",            # -17%
    "navrattan_korma": "mixed_veg_curry",              # -43%  cream, nuts, paneer
    "dum_aloo": "mixed_veg_curry",                     # -35%
    "greek_salad": "green_salad",                      # -70%  oil, feta, olives
    "caesar_salad": "green_salad",                     # -82%  dressing and cheese
    "misti_doi": "curd_yogurt",                        # -57%  it is sweetened
    "rasgulla": "gulab_jamun",                         # +81%  poached, not fried
}

# Names that look mappable and must not be. Listed so the omission reads as a
# decision rather than an oversight, and so the coverage report can say why.
EXCLUDED: dict[str, str] = {
    "biryani": "ambiguous — cannot tell veg_biryani from chicken_biryani by folder name",
    "cholebhature": "two dishes on one plate — a single label would teach the model both as one",
    "fried_rice": "no matching class; not jeera_rice (egg/soy/vegetables change the composition)",
    "chole_bhature": "two dishes on one plate — a single label would teach the model both as one",
    # The same plate, transliterated differently. `chana_batura` sat in ALIASES
    # for a while, which meant the rule above could be dodged by spelling: one
    # dataset's folder was rejected and another's was taught as pure chole_masala,
    # fried bread and all. The app would then read a poori as part of a chickpea
    # curry and cost the plate twice over. Listing the variants keeps the decision
    # from depending on which romanisation a dataset happened to pick.
    "chana_batura": "same plate as chole_bhature",
    "chana_bhatura": "same plate as chole_bhature",
    "chole_batura": "same plate as chole_bhature",
    "daal_baati_churma": "three-component thali, same problem",
    "litti_chokha": "two-component plate",
    "spaghetti_carbonara": "cream and egg sauce, not pasta_red_sauce",
    "burger": "no matching class",
    "momos": "no matching class",
    "chai": "a drink",
    "lassi": "a drink",
}

# Fruits-360 and friends ship per-cultivar folders ("Apple Braeburn", "Banana
# Lady Finger"). The prefix collapses them, but only under --loose: those images
# are single fruits on a white studio background, so a model trained on them
# learns the background as much as the fruit.
LOOSE_PREFIXES: dict[str, str] = {"apple": "apple", "banana": "banana"}

# The prefix rule is a cultivar collapser, and it has to be told where a cultivar
# name ends and a recipe begins. Without this, Food-101's `apple_pie` — a thousand
# images of pastry at roughly 265 kcal/100 g — resolves to `apple` at 52, which is
# wrong twice over: the classifier learns pie as fruit, and every pie the app ever
# sees is then reported at a fifth of its calories. A cultivar is a proper noun
# ("braeburn", "lady_finger", "red_1"); anything below is a preparation.
PREFIX_DISH_TAILS: frozenset[str] = frozenset({
    "pie", "tart", "cake", "bread", "muffin", "pudding", "crumble", "cobbler",
    "strudel", "turnover", "fritter", "pancake", "pancakes", "waffle", "waffles",
    "donut", "donuts", "chips", "crisps", "juice", "cider", "smoothie", "shake",
    "milkshake", "split", "sauce", "jam", "sorbet", "ice_cream", "custard",
    "halwa", "kheer", "sheera", "chutney", "curry", "raita", "salad", "fry",
    "fries", "cutlet", "tikki", "paratha", "roll", "toast", "foster", "flambe",
})


def normalise(name: str) -> str:
    """Fold a folder name to a comparable key: lowercase, underscore-separated."""
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", key).strip("_")


def resolve_label(folder: str, *, loose: bool) -> tuple[str | None, str]:
    """Map one source folder name to one of our classes.

    Returns `(our_class_or_None, reason)`, where the reason explains a miss so the
    coverage report can be specific instead of just dropping images on the floor.
    """
    key = normalise(folder)
    if not key:
        return None, "empty name"
    if key in CLASS_SET:
        return key, "exact"
    if key in ALIASES:
        return ALIASES[key], "alias"
    if key in NEAR_ALIASES:
        return NEAR_ALIASES[key], "near alias"
    if key in EXCLUDED:
        return None, f"excluded: {EXCLUDED[key]}"
    if key in LOOSE_ALIASES:
        if loose:
            return LOOSE_ALIASES[key], "loose alias"
        return None, f"approximate match to {LOOSE_ALIASES[key]} — pass --loose to include"
    if loose:
        head, _, tail = key.partition("_")
        if head in LOOSE_PREFIXES:
            dish = next((t for t in (tail, *tail.split("_")) if t in PREFIX_DISH_TAILS), None)
            if dish:
                return None, f"'{key}' is {dish}, not the fruit — prefix rule declined"
            return LOOSE_PREFIXES[head], "loose prefix"
    return None, "no mapping"


CLASS_SET = set(CLASS_LIST)

# Fail loudly at import if the tables drift away from the model's class list —
# a typo here would silently create a 43rd class folder and a model whose labels
# the nutrition layer cannot resolve.
_unknown = {
    target
    for table in (ALIASES, NEAR_ALIASES, LOOSE_ALIASES, LOOSE_PREFIXES)
    for target in table.values()
    if target not in CLASS_SET
}
if _unknown:
    raise SystemExit(f"Alias tables point at labels the classifier does not have: {sorted(_unknown)}")

# A name must resolve one way only. Overlap between the tables would make the
# result depend on the order the checks happen to run in, which is exactly the
# kind of bug that shows up as a class quietly missing 300 images.
_overlap = {
    name
    for a, b in ((ALIASES, NEAR_ALIASES), (ALIASES, LOOSE_ALIASES), (NEAR_ALIASES, LOOSE_ALIASES))
    for name in set(a) & set(b)
}
if _overlap:
    raise SystemExit(f"Folder names listed in more than one alias table: {sorted(_overlap)}")


# ------------------------------------------------------------------ Discovery


def find_class_dirs(root: Path) -> list[Path]:
    """Every directory that directly holds images.

    Structure-agnostic on purpose: `<ds>/images/<class>/x.jpg`,
    `<ds>/train/<class>/x.jpg` and `<ds>/<class>/x.jpg` all work, because the
    thing being looked for is "a folder of images" rather than a fixed depth.
    Kaggle dataset layouts vary far too much to hard-code one.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if any(Path(f).suffix.lower() in IMAGE_SUFFIXES for f in filenames):
            found.append(Path(dirpath))
    return found


def images_in(directory: Path) -> list[Path]:
    return sorted(
        child for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def collect(roots: list[Path], *, loose: bool) -> tuple[dict[str, list[Path]], list[tuple[str, int, str]]]:
    """Group every discovered image under our class names."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    misses: dict[str, tuple[int, str]] = {}

    for root in roots:
        for directory in find_class_dirs(root):
            name = directory.name
            if normalise(name) in STRUCTURAL_DIRS:
                # A folder of loose images with no class name — a flat dataset
                # whose labels live in a CSV. Nothing to map it to.
                name = directory.parent.name if directory.parent != root else name
            files = images_in(directory)
            if not files:
                continue
            label, reason = resolve_label(name, loose=loose)
            if label is None:
                previous = misses.get(normalise(name), (0, reason))
                misses[normalise(name)] = (previous[0] + len(files), reason)
                continue
            buckets[label].extend(files)

    unmapped = sorted(
        ((name, count, reason) for name, (count, reason) in misses.items()),
        key=lambda row: -row[1],
    )
    return buckets, unmapped


def balance(
    buckets: dict[str, list[Path]], *, minimum: int, maximum: int, seed: int
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Drop classes with too little data, cap the ones with too much.

    The cap matters more than it looks: Food-101 brings 1,000 images for
    `chicken_curry` while an Indian set brings 60 for `bhindi_masala`. Left
    alone the model would spend 94% of its gradient on one class. `train()` does
    apply inverse-frequency loss weights, but weighting a 16:1 imbalance is a
    worse instrument than not creating it.
    """
    kept: dict[str, list[Path]] = {}
    dropped: dict[str, int] = {}
    rng = random.Random(seed)
    for label, files in sorted(buckets.items()):
        if len(files) < minimum:
            dropped[label] = len(files)
            continue
        chosen = sorted(files)
        if len(chosen) > maximum:
            chosen = sorted(rng.sample(chosen, maximum))
        kept[label] = chosen
    return kept, dropped


def build_tree(kept: dict[str, list[Path]], out: Path) -> int:
    """Materialise `<out>/<our_class>/<n>.jpg`, linking rather than copying.

    Symlinks keep this near-instant and use no disk, which matters when the
    source is 5 GB of Food-101 and the writable layer is 20 GB. Hardlink then
    copy are fallbacks for filesystems that refuse symlinks.
    """
    if out.exists():
        shutil.rmtree(out)
    linked = 0
    for label, files in kept.items():
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files):
            target = folder / f"{index:05d}{source.suffix.lower()}"
            try:
                # `source` may be relative when the CLI is run from the project
                # root. A relative link is resolved from `target.parent`, which
                # silently produced `processed/data/raw/...` and zero readable
                # training samples. Resolve it before linking so both local and
                # Kaggle invocations point at the actual source file.
                os.symlink(source.resolve(), target)
            except OSError:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            linked += 1
    return linked


# --------------------------------------------------------------------- Report


def report(
    kept: dict[str, list[Path]],
    dropped: dict[str, int],
    unmapped: list[tuple[str, int, str]],
    *,
    minimum: int,
) -> dict:
    covered = sorted(kept)
    # "No data" means none was found at all. A class held back for having too few
    # images was already reported a line above, with its count — repeating it here
    # would contradict that.
    absent = [label for label in CLASS_LIST if label not in kept and label not in dropped]
    total = sum(len(files) for files in kept.values())

    print()
    print(f"Trainable classes: {len(covered)} of {len(CLASS_LIST)}   images: {total}")
    print("-" * 66)
    for label in covered:
        count = len(kept[label])
        bar = "#" * max(1, round(count / max(1, max(len(v) for v in kept.values())) * 28))
        print(f"  {label:<24} {count:>5}  {bar}")

    if dropped:
        print()
        print(f"Below --min-per-class ({minimum}), excluded from the head:")
        for label, count in sorted(dropped.items(), key=lambda row: -row[1]):
            print(f"  {label:<24} {count:>5}")

    if absent:
        print()
        print("No data at all — the signature prior keeps handling these:")
        print("  " + ", ".join(absent))

    if unmapped:
        print()
        print("Source folders that went nowhere (top 25):")
        for name, count, reason in unmapped[:25]:
            print(f"  {name:<32} {count:>6}  {reason}")

    return {
        "trainable_classes": covered,
        "images_per_class": {label: len(files) for label, files in kept.items()},
        "images_total": total,
        "dropped_too_few": dropped,
        "untrained_classes": absent,
        "unmapped_sources": [
            {"folder": name, "images": count, "reason": reason} for name, count, reason in unmapped
        ],
    }


# ----------------------------------------------------------------------- Main


def require_accelerator(args) -> None:
    """Refuse to start a CPU run unless it was asked for.

    `classify._device()` degrades to CPU when CUDA is missing, which is the right
    behaviour for inference and the wrong behaviour here: the most likely reason
    CUDA is missing in a Kaggle notebook is that the accelerator dropdown was
    never switched on, and the failure mode is not an error but an hour of quota
    spent going 20x too slow. Better to stop in two seconds and say so.
    """
    import torch

    forced = (args.device or "").lower()
    if forced.startswith("cuda") or forced == "mps" or args.allow_cpu:
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({memory:.0f} GB) — torch {torch.__version__}")
        return

    print(
        "\n".join(
            [
                "",
                "No CUDA device found, so this would train on the CPU.",
                "",
                "On Kaggle: Notebook  ->  Settings  ->  Accelerator  ->  GPU T4 x2 (or P100),",
                "then Session  ->  Restart, and run this again.",
                "",
                "If a CPU run really is what you want, pass --allow-cpu.",
                "",
            ]
        ),
        file=sys.stderr,
    )
    raise SystemExit(3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", type=Path, nargs="*", default=[Path("/kaggle/input")],
        help="Dataset roots to scan (default: /kaggle/input, i.e. everything attached).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/kaggle/working/data/processed"),
        help="Where to build the ImageFolder tree.",
    )
    parser.add_argument("--min-per-class", type=int, default=40)
    parser.add_argument(
        "--max-per-class", type=int, default=300,
        help="Cap per class. 300 x ~25 classes is roughly 100 s/epoch on a T4.",
    )
    parser.add_argument("--loose", action="store_true", help="Include approximate label matches.")
    parser.add_argument("--dry-run", action="store_true", help="Report coverage and stop.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Force a device, e.g. cuda or cpu.")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Train without a GPU. Refused by default: EfficientNet-B3 at 300px is "
             "roughly 20x slower on CPU, which turns a 40-minute run into most of a day.",
    )
    parser.add_argument("--version", default=None, help="Checkpoint tag, e.g. v1.")
    parser.add_argument("--seed", type=int, default=7)
    # Imported rather than restated, so the two entry points cannot drift apart.
    classify_module.add_recipe_arguments(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    if not args.dry_run:
        require_accelerator(args)

    roots = [root for root in args.input if root.is_dir()]
    if not roots:
        parser.error(
            f"None of {[str(p) for p in args.input]} is a directory. "
            "On Kaggle, attach datasets with 'Add Input' first."
        )
    print(f"Scanning: {', '.join(str(r) for r in roots)}")

    buckets, unmapped = collect(roots, loose=args.loose)
    if not buckets:
        print()
        print("Nothing mapped. Folders seen:")
        for name, count, reason in unmapped[:40]:
            print(f"  {name:<32} {count:>6}  {reason}")
        return 2

    kept, dropped = balance(
        buckets, minimum=args.min_per_class, maximum=args.max_per_class, seed=args.seed
    )
    if len(kept) < 2:
        print("\nFewer than two trainable classes — nothing to learn. Attach more datasets.")
        return 2

    coverage = report(kept, dropped, unmapped, minimum=args.min_per_class)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    coverage_path = args.out.parent / "coverage.json"

    if args.dry_run:
        coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        print(f"\nDry run — wrote {coverage_path}. Re-run without --dry-run to train.")
        return 0

    linked = build_tree(kept, args.out)
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"\nLinked {linked} images into {args.out}")

    summary = classify_module.train(
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        version=args.version,
        device=args.device,
        workers=args.workers,
        **classify_module.recipe_from_args(args),
    )
    summary["coverage"] = coverage

    result_path = Path(os.getenv("MODEL_DIR") or (BACKEND_DIR / "models")) / "last_run.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 66)
    print(f"  checkpoint   {summary['checkpoint']}")
    print(f"  device       {summary['device']}")
    print(f"  classes      {summary['classes']}   images {summary['images']}")
    print(f"  val top-1    {summary['val_top1']}")
    print(f"  test top-1   {summary['test_top1']}")
    print("=" * 66)
    print("\nDownload the checkpoint, then either:")
    print("  drop it at backend/models/efficientnet_v1.pt, or")
    print("  upload it to a model_api/ deployment and set CLASSIFIER_URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
