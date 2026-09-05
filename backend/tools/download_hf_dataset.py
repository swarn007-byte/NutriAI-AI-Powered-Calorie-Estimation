#!/usr/bin/env python3
"""Download public Hugging Face food datasets into Nutri-AI's ImageFolder tree.

The training entry point intentionally only scans local folders. This script is
the reproducible internet step before it: it downloads parquet shards, decodes
the embedded images, applies the same explicit label map as ``train_kaggle.py``,
and records source/license/coverage metadata beside the images.

Examples::

    python backend/tools/download_hf_dataset.py --clean
    python backend/tools/train_kaggle.py \
        --input data/raw/huggingface --out data/processed --dry-run

The default sources are the CC BY 4.0 Enhanced Indian Food Classification set
and the CC0 Indian Foods set. Add another public dataset with ``--dataset``.
The script never silently maps a dish into a different nutrition class: loose
aliases stay disabled unless ``--loose`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402

import train_kaggle  # noqa: E402

log = logging.getLogger("nutriai.download_hf_dataset")

DEFAULT_DATASETS = (
    "SohlHealth/enhanced-indian-food-classification",
    "bharat-raghunathan/indian-foods-dataset",
)

# Some dataset repos do not publish dataset_infos.json even though their parquet
# labels are ClassLabel integers. These names are also visible in each dataset's
# public card and make the download robust when the datasets-server is busy.
KNOWN_LABEL_NAMES: dict[str, list[str]] = {
    "bharat-raghunathan/indian-foods-dataset": [
        "biryani", "cholebhature", "dabeli", "dal", "dhokla", "dosa",
        "jalebi", "kathiroll", "kofta", "naan", "pakora", "paneer",
        "panipuri", "pavbhaji", "vadapav",
    ],
}


def _request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "NutriAI-dataset-builder/1.0"})
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _repo_metadata(repo: str) -> dict:
    return _request_json(f"https://huggingface.co/api/datasets/{quote(repo, safe='/')}")


def _label_names(repo: str) -> list[str] | None:
    """Read ClassLabel names from datasets-server, with a small local fallback."""
    query = urlencode({"dataset": repo, "config": "default", "split": "train"})
    try:
        body = _request_json(f"https://datasets-server.huggingface.co/first-rows?{query}")
        for feature in body.get("features") or []:
            if feature.get("name") != "label":
                continue
            kind = feature.get("type") or {}
            names = kind.get("names")
            if isinstance(names, list):
                return [str(name) for name in names]
            if isinstance(names, dict):
                return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
            return None
    except Exception as exc:
        log.warning("Could not read label metadata for %s (%s)", repo, exc)
    return KNOWN_LABEL_NAMES.get(repo)


def _parquet_files(metadata: dict, *, splits: set[str]) -> list[tuple[str, str]]:
    files = []
    for sibling in metadata.get("siblings") or []:
        filename = str(sibling.get("rfilename") if isinstance(sibling, dict) else sibling)
        if not filename.endswith(".parquet") or not filename.startswith("data/"):
            continue
        basename = Path(filename).name.lower()
        split = "test" if "test" in basename else "validation" if "valid" in basename else "train"
        if split in splits:
            files.append((filename, split))
    return sorted(files)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "NutriAI-dataset-builder/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    log.info("Downloading %s", url)
    with urlopen(request, timeout=120) as response:
        # Some mirrors ignore Range and return the whole file. Append only when
        # the server confirms a partial response; otherwise restart safely.
        mode = "ab" if offset and getattr(response, "status", 200) == 206 else "wb"
        with temporary.open(mode) as handle:
            if mode == "wb":
                offset = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temporary.replace(destination)


def _image_bytes(image: object, repo: str) -> bytes | None:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if not isinstance(image, dict):
        return None
    payload = image.get("bytes")
    if payload:
        return bytes(payload)
    path = image.get("path")
    if not path:
        return None
    url = f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/main/{quote(str(path), safe='/')}"
    try:
        request = Request(url, headers={"User-Agent": "NutriAI-dataset-builder/1.0"})
        with urlopen(request, timeout=45) as response:
            return response.read()
    except Exception as exc:
        log.warning("Could not fetch image %s/%s (%s)", repo, path, exc)
        return None


def _raw_label(label: object, names: list[str] | None) -> str | None:
    if isinstance(label, dict):
        label = label.get("label", label.get("id"))
    if isinstance(label, int):
        return names[label] if names is not None and 0 <= label < len(names) else None
    if label is None:
        return None
    return str(label)


def _canonical_image(payload: bytes) -> bytes | None:
    from io import BytesIO

    try:
        with Image.open(BytesIO(payload)) as image:
            image = image.convert("RGB")
            image.load()
            normalized = BytesIO()
            image.save(normalized, "JPEG", quality=95, optimize=True)
            return normalized.getvalue()
    except Exception:
        return None


def _extract_shard(
    parquet: Path,
    *,
    repo: str,
    split: str,
    names: list[str] | None,
    output: Path,
    seen_hashes: set[str],
    counts: Counter,
    skipped: Counter,
    loose: bool,
) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Dataset extraction needs pyarrow. Install it with: pip install pyarrow") from exc

    written = 0
    parquet_file = pq.ParquetFile(parquet)
    for batch_index, batch in enumerate(parquet_file.iter_batches(columns=["image", "label"], batch_size=64)):
        for row_index, row in enumerate(batch.to_pylist()):
            source_label = _raw_label(row.get("label"), names)
            if not source_label:
                skipped["unknown_label"] += 1
                continue
            target_label, reason = train_kaggle.resolve_label(source_label, loose=loose)
            if target_label is None:
                skipped[f"unmapped:{reason}"] += 1
                continue
            target = output / target_label / f"{repo.replace('/', '__')}__{split}__{batch_index:06d}_{row_index:02d}.jpg"
            # Same source/split/row means the output is already materialized;
            # avoid decoding the image again on a cached audit.
            if target.exists():
                continue
            payload = _image_bytes(row.get("image"), repo)
            if not payload:
                skipped["missing_image"] += 1
                continue
            canonical = _canonical_image(payload)
            if canonical is None:
                skipped["corrupt_image"] += 1
                continue
            # Hash the bytes written to ImageFolder, not the source container.
            # PNG/JPEG re-encodings of the same photo then deduplicate too.
            digest = hashlib.sha256(canonical).hexdigest()
            if digest in seen_hashes:
                skipped["duplicate"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(canonical)
            seen_hashes.add(digest)
            counts[target_label] += 1
            written += 1
    return written


def download_dataset(
    repos: list[str],
    *,
    output: Path,
    cache: Path,
    splits: set[str],
    loose: bool,
    clean: bool,
) -> dict:
    manifest_path = output.parent / "huggingface_manifest.json"
    previous: dict = {}
    if not clean and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Ignoring unreadable previous manifest at %s", manifest_path)
    previous_mode = previous.get("mapping_mode")
    requested_mode = "loose" if loose else "strict"
    if previous_mode and previous_mode != requested_mode:
        raise SystemExit(
            f"Existing tree uses {previous_mode} mappings; pass --clean before switching to {requested_mode}."
        )
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # A strict pass is the default. The explicit flag is carried into the
    # manifest so a future training run can audit how labels were produced.
    seen_hashes: set[str] = set()
    if not clean:
        # Existing output may have been produced by an earlier invocation with
        # another source. Hash it once so cross-source duplicates are skipped.
        for image in output.rglob("*"):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                try:
                    seen_hashes.add(hashlib.sha256(image.read_bytes()).hexdigest())
                except OSError:
                    pass
    counts: Counter = Counter()
    for folder in output.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        counts[folder.name] = sum(
            1 for image in folder.rglob("*")
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
    # Recompute rejections for this invocation. Keeping old rejection counters
    # would double them every time a cached shard is audited again.
    skipped: Counter = Counter()
    sources: dict[str, dict] = dict(previous.get("sources") or {})

    for repo in repos:
        metadata = _repo_metadata(repo)
        names = _label_names(repo)
        files = _parquet_files(metadata, splits=splits)
        if not files:
            log.warning("No parquet shards found for %s", repo)
            continue
        source = sources.setdefault(repo, {
            "license": (metadata.get("cardData") or {}).get("license") or next(
                (tag.split(":", 1)[1] for tag in metadata.get("tags") or [] if tag.startswith("license:")),
                "unknown",
            ),
            "labels": names or [],
            "files": [],
        })
        for filename, split in files:
            cached = cache / repo.replace("/", "__") / Path(filename).name
            url = f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/main/{quote(filename, safe='/')}?download=true"
            _download(url, cached)
            file_record = {"name": filename, "split": split, "bytes": cached.stat().st_size}
            if file_record not in source["files"]:
                source["files"].append(file_record)
            _extract_shard(
                cached,
                repo=repo,
                split=split,
                names=names,
                output=output,
                seen_hashes=seen_hashes,
                counts=counts,
                skipped=skipped,
                loose=loose,
            )

    manifest = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "sources": sources,
        "mapping_mode": "loose" if loose else "strict",
        "images_total": sum(counts.values()),
        "images_per_class": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "deduplicated_images": len(seen_hashes),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", dest="datasets", help="Hugging Face dataset repo (repeatable).")
    parser.add_argument("--output", type=Path, default=Path("data/raw/huggingface"))
    parser.add_argument("--cache", type=Path, default=Path("data/.hf-cache"))
    parser.add_argument("--split", action="append", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--loose", action="store_true", help="Enable approximate aliases from train_kaggle.py.")
    parser.add_argument("--clean", action="store_true", help="Delete the output ImageFolder before extraction.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    manifest = download_dataset(
        args.datasets or list(DEFAULT_DATASETS),
        output=args.output,
        cache=args.cache,
        # Training defaults to the publisher's train split. Pull validation or
        # test explicitly so the reported score remains a real held-out number.
        splits=set(args.split or ("train",)),
        loose=args.loose,
        clean=args.clean,
    )
    print(json.dumps({"images_total": manifest["images_total"], "images_per_class": manifest["images_per_class"], "skipped": manifest["skipped"]}, indent=2))
    print(f"\nWrote {args.output} and {args.output.parent / 'huggingface_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
