#!/usr/bin/env python3
"""Inspect a third-party food-classifier checkpoint and repack it for this API.

Why this file exists
--------------------
`tools/train_kaggle.py` is the translation layer for *datasets*: it maps public
folder names onto `classify.CLASS_LIST`. This script is its sibling for *weights*.
Somebody else already trained a good food classifier and published a `.pt`; the
weights are usable, but nothing else about the file matches what
`classify._load_checkpoint` expects:

  * it is often a bare `state_dict`, a pickled `nn.Module`, a TorchScript archive,
    or a Lightning checkpoint with a `model.` prefix on every key,
  * its label space is its own (`chana_masala`, `masala_dosa`, `jalebi`), and our
    nutrition tables are keyed on ours (`chole_masala`, `dosa`, and no jalebi),
  * its class names usually live in a separate JSON, because a `.pt` holds only
    tensors.

Run with no `--out` to get a report and nothing else. That report — especially the
coverage table — is the thing to read before adopting anything.

    python backend/tools/adopt_checkpoint.py ~/Downloads/food_b3.pt \
        --labels ~/Downloads/class_indices.json

    python backend/tools/adopt_checkpoint.py ~/Downloads/food_b3.pt \
        --labels ~/Downloads/class_indices.json \
        --out backend/models/efficientnet_v2.pt --version v2-adopted

A label this project has no nutrition row for is never renamed into a dish we do
have. It becomes `mixed_dish`, a `COARSE_FALLBACK` group whose numbers the API
reports as "category-average (estimated)", or it is removed from the head
entirely with `--drop-unmapped` so the model cannot predict it at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import train_kaggle  # noqa: E402  (same directory; provides resolve_label)

log = logging.getLogger("nutriai.adopt_checkpoint")

# Where the head lives, by convention, in the checkpoints people actually publish.
# Ordered most- to least-specific so `classifier.1.weight` is not shadowed by a
# prefix match on `classifier`.
HEAD_KEYS: tuple[str, ...] = (
    "classifier.weight",
    "classifier.1.weight",
    "classifier.fc.weight",
    "head.fc.weight",
    "head.weight",
    "last_linear.weight",
    "_fc.weight",
    "fc.weight",
)

# A fingerprint key that only one architecture family produces.
FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ("conv_stem.weight", "efficientnet (timm)"),
    ("features.0.0.weight", "efficientnet / mobilenet (torchvision)"),
    ("patch_embed.proj.weight", "vision transformer"),
    ("layer1.0.conv1.weight", "resnet"),
    ("stages.0.blocks.0.conv_dw.weight", "convnext (timm)"),
)

# `in_features` of the head, which is what actually decides whether the weights
# can be poured into the backbone `classify._build_backbone` constructs.
B3_IN_FEATURES = 1536
EFFICIENTNET_WIDTHS: dict[int, str] = {
    1280: "efficientnet_b0/b1",
    1408: "efficientnet_b2",
    1536: "efficientnet_b3",
    1792: "efficientnet_b4",
    2048: "efficientnet_b5",
}

# Keys torch.save'd wrappers prepend to every parameter.
WRAPPER_PREFIXES: tuple[str, ...] = ("module.", "model.", "net.", "backbone.")

# Where a state_dict hides inside a training checkpoint.
STATE_DICT_KEYS: tuple[str, ...] = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "weights",
    "params",
)

# Class names, by the key the publisher happened to use.
LABEL_KEYS: tuple[str, ...] = ("classes", "class_names", "labels", "idx_to_class", "class_to_idx")

# The honest destination for a dish we have no nutrition row for: a coarse group,
# not a neighbouring dish. `nutrition.resolve_per_100g` reports it as an estimate.
UNMAPPED_LABEL = "mixed_dish"


class Unusable(SystemExit):
    """The checkpoint cannot be adopted, with the reason already explained."""

    def __init__(self, message: str) -> None:
        super().__init__(f"error: {message}")


# --------------------------------------------------------------------------
# Reading whatever the publisher happened to save
# --------------------------------------------------------------------------
def load_payload(path: Path) -> Any:
    import torch

    # torch.load transparently dispatches a TorchScript archive to torch.jit.load,
    # and a traced module's state_dict still carries the original parameter names —
    # so scripted checkpoints work here too, and need no special case.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Unusable(f"{path.name} is not a checkpoint torch can read ({exc})") from exc


def _is_tensor_map(value: Any) -> bool:
    import torch

    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(item, torch.Tensor) for item in value.values())


def extract_state_dict(payload: Any) -> tuple[dict[str, Any], str]:
    """Return `(state_dict, how_it_was_found)`."""
    import torch

    if isinstance(payload, torch.nn.Module):
        return dict(payload.state_dict()), "pickled nn.Module (.state_dict() taken)"
    if not isinstance(payload, dict):
        raise Unusable(f"unsupported checkpoint type {type(payload).__name__}")
    if _is_tensor_map(payload):
        return dict(payload), "bare state_dict at the top level"
    for key in STATE_DICT_KEYS:
        inner = payload.get(key)
        if isinstance(inner, torch.nn.Module):
            return dict(inner.state_dict()), f"nn.Module under {key!r}"
        if _is_tensor_map(inner):
            return dict(inner), f"state_dict under {key!r}"
    raise Unusable(
        "no tensors found. Top-level keys were: "
        + ", ".join(sorted(map(str, payload.keys()))[:12])
    )


def strip_wrapper_prefix(state_dict: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    for prefix in WRAPPER_PREFIXES:
        if all(key.startswith(prefix) for key in state_dict):
            stripped = {key[len(prefix) :]: value for key, value in state_dict.items()}
            deeper, _ = strip_wrapper_prefix(stripped)
            return deeper, prefix
    return state_dict, None


def find_head(state_dict: dict[str, Any]) -> tuple[str, int, int]:
    """Return `(weight_key, num_classes, in_features)` for the classifier head."""
    for key in HEAD_KEYS:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return key, int(tensor.shape[0]), int(tensor.shape[1])
    trailing = [k for k, v in state_dict.items() if v.ndim == 2 and k.endswith(".weight")]
    if trailing:
        key = trailing[-1]
        tensor = state_dict[key]
        log.warning("Head key %r is a guess — none of the conventional names matched", key)
        return key, int(tensor.shape[0]), int(tensor.shape[1])
    raise Unusable("no 2-D weight tensor, so there is no linear classifier head to read")


def identify_family(state_dict: dict[str, Any]) -> str:
    for marker, family in FAMILY_MARKERS:
        if marker in state_dict:
            return family
    return "unrecognised architecture"


# --------------------------------------------------------------------------
# Class names
# --------------------------------------------------------------------------
def _names_from_mapping(mapping: dict[Any, Any], count: int | None) -> list[str] | None:
    """Accept either `{index: name}` or `{name: index}` and return index order."""
    keys_are_indices = all(str(key).lstrip("-").isdigit() for key in mapping)
    values_are_indices = all(isinstance(value, int) for value in mapping.values())
    if keys_are_indices:
        pairs = [(int(key), str(value)) for key, value in mapping.items()]
    elif values_are_indices:
        pairs = [(int(value), str(key)) for key, value in mapping.items()]
    else:
        return None
    pairs.sort()
    if [index for index, _ in pairs] != list(range(len(pairs))):
        raise Unusable(f"label indices are not 0..{len(pairs) - 1}: {[i for i, _ in pairs][:8]}…")
    names = [name for _, name in pairs]
    if count is not None and len(names) != count:
        raise Unusable(f"label file lists {len(names)} names but the head has {count} outputs")
    return names


def read_labels(path: Path, count: int) -> list[str]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise Unusable(f"could not read {path}: {exc}") from exc
    if isinstance(data, list):
        if len(data) != count:
            raise Unusable(f"{path.name} lists {len(data)} names but the head has {count} outputs")
        return [str(item) for item in data]
    if isinstance(data, dict):
        for key in LABEL_KEYS:
            if key in data and isinstance(data[key], (list, dict)):
                nested = data[key]
                if isinstance(nested, list):
                    return [str(item) for item in nested]
                names = _names_from_mapping(nested, count)
                if names:
                    return names
        names = _names_from_mapping(data, count)
        if names:
            return names
    raise Unusable(
        f"{path.name} is not a label mapping. Expected a JSON list of names, "
        "or an object of index->name / name->index."
    )


def labels_from_payload(payload: Any, count: int) -> tuple[list[str], str] | None:
    if not isinstance(payload, dict):
        return None
    for key in LABEL_KEYS:
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) == count:
            return [str(item) for item in value], f"checkpoint key {key!r}"
        if isinstance(value, dict):
            names = _names_from_mapping(value, count)
            if names:
                return names, f"checkpoint key {key!r}"
    return None


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def resolve_all(names: list[str], *, loose: bool) -> list[tuple[str, str | None, str]]:
    return [(name, *train_kaggle.resolve_label(name, loose=loose)) for name in names]


def print_report(resolved: list[tuple[str, str | None, str]]) -> tuple[int, int]:
    mapped = [row for row in resolved if row[1] is not None]
    missed = [row for row in resolved if row[1] is None]

    by_target: dict[str, list[str]] = defaultdict(list)
    for source, target, _ in mapped:
        by_target[str(target)].append(source)

    print(f"\nmapped {len(mapped)}/{len(resolved)} source labels onto {len(by_target)} of our classes")
    for target in sorted(by_target):
        sources = by_target[target]
        marker = "  <- MERGED" if len(sources) > 1 else ""
        print(f"  {target:<24} {', '.join(sorted(sources))}{marker}")

    if missed:
        print(f"\nno nutrition row for {len(missed)} source labels:")
        for source, _, reason in sorted(missed):
            print(f"  {source:<24} {reason}")

    uncovered = sorted(set(train_kaggle.CLASS_LIST) - set(by_target))
    if uncovered:
        print(f"\n{len(uncovered)} of our classes this model cannot predict:")
        print("  " + ", ".join(uncovered))

    duplicates = sum(len(sources) - 1 for sources in by_target.values() if len(sources) > 1)
    if duplicates:
        print(
            f"\nnote: {duplicates} extra head rows share a label with another row. That is "
            "safe — the label is still correct — but softmax splits one dish's "
            "probability across them, so its confidence reads lower than it is."
        )
    return len(mapped), len(missed)


# --------------------------------------------------------------------------
# Repack
# --------------------------------------------------------------------------
def repack(
    state_dict: dict[str, Any],
    resolved: list[tuple[str, str | None, str]],
    head_key: str,
    *,
    drop_unmapped: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return `(state_dict, classes)` matching `classify._load_checkpoint`."""
    import torch

    bias_key = head_key.replace(".weight", ".bias")
    if not drop_unmapped:
        return state_dict, tuple(target or UNMAPPED_LABEL for _, target, _ in resolved)

    keep = [index for index, (_, target, _) in enumerate(resolved) if target is not None]
    if not keep:
        raise Unusable("every source label was unmapped, so --drop-unmapped leaves no head")
    picker = torch.tensor(keep, dtype=torch.long)
    trimmed = dict(state_dict)
    trimmed[head_key] = state_dict[head_key].index_select(0, picker).clone()
    if bias_key in state_dict:
        trimmed[bias_key] = state_dict[bias_key].index_select(0, picker).clone()
    return trimmed, tuple(str(resolved[index][1]) for index in keep)


def verify(state_dict: dict[str, Any], classes: tuple[str, ...]) -> None:
    """Run the exact load path `classify` uses, then one forward pass."""
    import numpy as np
    import torch
    from PIL import Image

    import classify

    model = classify._build_backbone(len(classes), pretrained=False)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        first = [line.strip() for line in str(exc).splitlines() if "size mismatch" in line][:3]
        raise Unusable(
            "the weights are a different architecture from the efficientnet_b3 this API "
            "builds — same parameter names, different shapes:\n  "
            + "\n  ".join(first or [str(exc)[:200]])
        ) from exc
    if missing or unexpected:
        raise Unusable(
            "the weights do not fit the efficientnet_b3 this API builds.\n"
            f"  {len(missing)} missing parameters, first few: {list(missing)[:4]}\n"
            f"  {len(unexpected)} unexpected parameters, first few: {list(unexpected)[:4]}"
        )
    model.eval()
    noise = Image.fromarray(
        np.random.randint(0, 255, (classify.INPUT_RESOLUTION, classify.INPUT_RESOLUTION, 3), dtype=np.uint8)
    )
    with torch.no_grad():
        logits = model(classify._preprocess(noise).unsqueeze(0))
    if tuple(logits.shape) != (1, len(classes)):
        raise Unusable(f"forward pass produced {tuple(logits.shape)}, expected (1, {len(classes)})")
    print(f"\nverified: loads as efficientnet_b3 and returns {len(classes)} logits at "
          f"{classify.INPUT_RESOLUTION}x{classify.INPUT_RESOLUTION}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="the downloaded .pt / .pth file")
    parser.add_argument("--labels", type=Path, help="JSON label mapping, if not inside the checkpoint")
    parser.add_argument("--out", type=Path, help="write a repacked checkpoint here (omit for a report only)")
    parser.add_argument("--version", default=None, help="version string stored in the checkpoint")
    parser.add_argument("--loose", action="store_true", help="accept train_kaggle's approximate aliases")
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help=f"remove unmapped head rows instead of labelling them {UNMAPPED_LABEL!r}",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.checkpoint.is_file():
        raise Unusable(f"{args.checkpoint} is not a file")
    if args.out and args.out.exists() and not args.force:
        raise Unusable(f"{args.out} already exists — pass --force to overwrite it")

    payload = load_payload(args.checkpoint)
    state_dict, origin = extract_state_dict(payload)
    state_dict, prefix = strip_wrapper_prefix(state_dict)
    head_key, num_classes, in_features = find_head(state_dict)
    family = identify_family(state_dict)

    size_mb = args.checkpoint.stat().st_size / 1e6
    print(f"{args.checkpoint.name}  ({size_mb:.1f} MB)")
    print(f"  weights found as   {origin}" + (f", after stripping {prefix!r}" if prefix else ""))
    print(f"  tensors            {len(state_dict)}")
    print(f"  architecture       {family}")
    print(f"  head               {head_key}  {num_classes} classes x {in_features} features")
    if in_features != B3_IN_FEATURES:
        looks_like = (
            EFFICIENTNET_WIDTHS.get(in_features, "a width this API does not build")
            if "efficientnet" in family
            else family
        )
        print(
            f"  WARNING            a {in_features}-feature head means {looks_like}, but "
            f"classify._build_backbone makes efficientnet_b3 ({B3_IN_FEATURES} features). "
            "These weights cannot be repacked without changing the backbone."
        )
    if isinstance(payload, dict) and not _is_tensor_map(payload):
        skip = set(STATE_DICT_KEYS) | set(LABEL_KEYS)
        extras = sorted(str(key) for key in payload if str(key) not in skip)
        if extras:
            print(f"  other metadata     {', '.join(extras)}")

    if args.labels:
        names = read_labels(args.labels, num_classes)
        source = str(args.labels)
    else:
        found = labels_from_payload(payload, num_classes)
        if not found:
            raise Unusable(
                f"the head has {num_classes} outputs but the checkpoint carries no class "
                "names. Pass --labels with the publisher's mapping (Keras/PyTorch "
                "notebooks usually save it as class_indices.json), because index order "
                "cannot be guessed."
            )
        names, source = found
    print(f"  labels             {len(names)} names from {source}")

    resolved = resolve_all(names, loose=args.loose)
    mapped_count, missed_count = print_report(resolved)
    if not mapped_count:
        raise Unusable(
            "not one source label maps onto our catalog. This model is trained on a "
            "different cuisine; adopting it would mean writing new nutrition rows, "
            "not remapping labels."
        )

    if not args.out:
        print("\nreport only — pass --out to write a repacked checkpoint")
        return 0

    if in_features != B3_IN_FEATURES:
        raise Unusable(
            f"refusing to repack: the head expects {in_features} features and "
            f"efficientnet_b3 produces {B3_IN_FEATURES}. Find a checkpoint trained on "
            "efficientnet_b3, or serve this one behind CLASSIFIER_URL where its own "
            "architecture stays intact."
        )

    packed_state, classes = repack(state_dict, resolved, head_key, drop_unmapped=args.drop_unmapped)
    verify(packed_state, classes)

    import torch

    checkpoint = {
        "state_dict": packed_state,
        "classes": classes,
        "version": args.version or f"adopted-{args.checkpoint.stem}",
        "input_resolution": None,
        "adopted_from": args.checkpoint.name,
        "source_labels": tuple(names),
    }
    import classify

    checkpoint["input_resolution"] = classify.INPUT_RESOLUTION
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)

    counts = Counter(classes)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(classes)} head rows over {len(counts)} distinct labels")
    if not args.drop_unmapped and missed_count:
        print(
            f"  {counts[UNMAPPED_LABEL]} rows labelled {UNMAPPED_LABEL!r} — the API will report "
            "these as \"Mixed Dish\" with category-average numbers, never as a named dish"
        )
    print(f"\nto use it:  CLASSIFIER_CHECKPOINT={args.out} uvicorn main:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
