# Nutri-AI

**AI-based calorie prediction and nutrient estimation from multi-item food photos.**

Take a photo of your plate. Nutri-AI finds every item on it, works out how much of each there is, identifies the dish, and returns a full calorie and nutrient breakdown — no manual logging.

> **Status: idea stage, pre-funding, built solo.** The classifier is trained but below target (78.46% test top-1 vs. an 85% goal), and its head covers 24 of the catalog's 42 dishes. Read [Model Status](#9-model-status--the-honest-numbers) before quoting any accuracy number. `design.md` is the design of record; its §22 records every place the code departs from the spec and why.

---

## Table of Contents

**Getting oriented**
1. [What problem this solves](#1-what-problem-this-solves)
2. [Quick start — running in 5 minutes](#2-quick-start--running-in-5-minutes)
3. [Architecture: the big picture](#3-architecture-the-big-picture)
4. [The five stages, explained](#4-the-five-stages-explained)
5. [How a photo becomes grams](#5-how-a-photo-becomes-grams)
6. [Degradation: why it never crashes](#6-degradation-why-it-never-crashes)

**Reference**

7. [Every file in the repo](#7-every-file-in-the-repo)
8. [The 42 dishes](#8-the-42-dishes)
9. [Model status — the honest numbers](#9-model-status--the-honest-numbers)
10. [Configuration reference](#10-configuration-reference)
11. [API reference](#11-api-reference)
12. [Database schema](#12-database-schema)
13. [The mobile app](#13-the-mobile-app)
14. [Training the classifier](#14-training-the-classifier)
15. [Tests](#15-tests)
16. [Deployment](#16-deployment)

**Context**

17. [Troubleshooting](#17-troubleshooting)
18. [Known limitations](#18-known-limitations)
19. [Roadmap](#19-roadmap)
20. [A note on health-linked features](#20-a-note-on-health-linked-features)
21. [References](#21-references)

---

## 1. What problem this solves

Over 2.5 billion adults worldwide are overweight (WHO, 2024), and diet-related disease accounts for a large share of premature deaths in India (ICMR). People generally know that diet matters. What they won't do is type every meal into an app — and when they do, self-reported intake is typically underestimated by 25–30%.

Nutri-AI removes the typing. One photo, one answer.

**What makes it different from the obvious "food photo app":**

- **Multi-item.** A thali has six things on it. Most published systems classify one dish per image.
- **Portion-aware.** It estimates *how much*, not just *what*. A photo of rice is worth nothing without grams.
- **Indian-food-first.** The nutrition tables are IFCT 2017 (the Indian Food Composition Tables), not a US database with an Indian dish bolted on.
- **Honest about uncertainty.** Every number carries a provenance label. When it isn't sure, it says so instead of guessing prettily.

### Scope

**Built and working:** photo analysis, multi-item detection, portion estimation, 42-dish classification, calories + macros + 13 micronutrients with %DV, a human review step, meal history, daily summaries, per-item correction, a React Native mobile app, and a web app.

**Deliberately not built (vision-stage only):** doctor-linked diet monitoring, medication-aligned guidance, disease-specific alerts, multi-service backend, Kubernetes, CI/CD. See [§20](#20-a-note-on-health-linked-features) — the health-linked items are gated on legal review, not on effort.

---

## 2. Quick start — running in 5 minutes

### Prerequisites

**Python 3.10 or newer. That is the entire list.**

No Node (the web frontend has no build step). No PostgreSQL (SQLite is the default). No GPU. No API keys. No `.env` file. No model weights.

### Install and run

```bash
git clone <your-repo-url>
cd NutriGuide/backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Open <http://localhost:8000>. The web app is served from `/`, interactive API docs are at `/docs`.

That install is roughly 230 MB — FastAPI, uvicorn, pydantic, SQLAlchemy, psycopg, PyJWT, pillow, numpy, scipy, scikit-learn and httpx. Notably **not** torch, which would have added ~2.5 GB. The app runs the full pipeline without it; see [§6](#6-degradation-why-it-never-crashes).

### Try it without uploading anything

The home page shows a strip of four sample plates. Click one and it runs a complete analysis.

### Try it from the terminal

```bash
# 1. Get a token. Guest sessions need no signup.
TOKEN=$(curl -sX POST http://localhost:8000/api/auth/guest \
  | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# 2. One-shot analysis.
curl -X POST http://localhost:8000/api/meals/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@plate.jpg"
```

### Optional: turn on the neural networks

```bash
pip install torch torchvision ultralytics timm
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"
```

Restart. `GET /api/health` will now report `backend: yolov8` and `backend: midas` instead of the fallbacks. This is a ~2.5 GB install and **nothing requires it** — it improves accuracy, it does not enable functionality.

---

## 3. Architecture: the big picture

### One service, five stages, two phases

Nutri-AI is a **single FastAPI process**. Not microservices. One person maintains it, and the whole thing fits in one head. The only component that may optionally live elsewhere is the classifier ([`model_api/`](#model_api--the-classifier-as-a-service)), because it is the one piece that needs torch.

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENTS                                     │
│  mobile/ (React Native + Expo)  ·  frontend/ (vanilla ES modules)│
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP + JWT
┌────────────────────────────▼────────────────────────────────────┐
│              backend/main.py — FastAPI, all routes               │
│  auth.py · schemas.py · db.py · drafts.py                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              backend/pipeline.py — the orchestrator              │
│                                                                  │
│  imaging.py → detection.py → depth.py → classify.py →nutrition.py│
│   stage 1       stage 2       stage 3     stage 4      stage 5   │
└─────────────────────────────────────────────────────────────────┘
```

### The two-phase flow — the most important design decision

The obvious design is one call: photo in, meal out. **It is built as two calls with a human in the middle**, and understanding why explains most of the codebase.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ PHASE 1 · SCAN            POST /api/meals/scan               │
   │                                                              │
   │ stage 1  input      decode, EXIF-orient, downscale, fit plate│
   │ stage 2  detection  YOLOv8            → one region per item  │
   │ stage 4  classify   EfficientNet-B3   → the exact dish       │
   │                                                              │
   │ No weights. No calories. No plate size asked for yet.        │
   │ A region with no measurable area is DROPPED here, not priced.│
   └───────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   HUMAN REVIEW       │
                    │  rename · remove     │
                    │  add · count pieces  │
                    │  → then plate size   │
                    └──────────┬───────────┘
                               │
   ┌───────────────────────────▼──────────────────────────────────┐
   │ PHASE 2 · DEEP PASS   POST /api/meals/{draft_id}/analyze     │
   │                                                              │
   │ stage 3  depth      MiDaS v3 + plate geometry → ml → grams   │
   │          ...or  count × grams-per-piece, for countable food  │
   │ stage 5  nutrition  IFCT 2017 + USDA → kcal, macros, micros  │
   └───────────────────────────┬──────────────────────────────────┘
                               │
                          Results + history
```

**Why split it?** One photo — four samosas and a lemon wedge — exposed three problems at once. It came back as 943 kcal across three items: `Samosa 260 g`, `Curry or Gravy 45 g` (24% confident) and `Fish Curry 45 g` (35% confident).

1. **Phantom items were priced before anyone could deny them.** Both 45 g figures were `WEIGHT_BOUNDS["curry"][0]` — the floor for a region with no measurable area. The lemon wedge and the basket weave got billed. The fix is a rule, not a threshold: *a detection must earn its area during the scan or it never becomes an item.* `pipeline.scan_image` drops anything where `area_px == 0 or area_cm2 <= MIN_ITEM_AREA_CM2`.
2. **Countable food was back-solved from geometry.** Four samosas were one 260 g blob. The user could not say "four" — only override the total weight, which throws the measurement away. Countable dishes now carry grams-per-piece in `nutrition.PIECE_WEIGHTS`.
3. **The plate diameter was being assumed silently.** It is the single largest error term in the whole system, and it was a default. Now the app asks — but only after the item list is agreed, so the user is answering one question at a time.

Full reasoning in `design.md` §22.13.

**Note on stage numbering:** stages are numbered by the pipeline definition, not by execution order. Stage 4 (classification) runs *before* stage 3 (depth) because naming a dish needs no plate scale, and MiDaS is the expensive part — no point paying for it on items the user is about to delete.

**Single-shot is still available.** `POST /api/meals/analyze` runs both phases in one call, for callers with no human to ask.

---

## 4. The five stages, explained

### Stage 1 — Input (`imaging.py`, 143 lines)

Turns raw upload bytes into something the rest of the pipeline can trust.

- `read_image(payload)` — decode, validating the format.
- `autocorrect(image)` — apply EXIF orientation. Phone photos are frequently stored rotated with a flag; skip this and every downstream mask is sideways.
- `resize_max(image, max_dim)` — downscale to `MAX_IMAGE_DIMENSION` (default 1600 px).
- `prepare(image, work_dim=640)` — produces the `PreparedImage` the pipeline passes around: the working-size array plus its Lab colour conversion.
- `rgb_to_lab()` / `chroma()` / `hue_degrees()` — colour-space conversion. **Lab, not RGB**, because the classical engine reasons about colour perceptually: in Lab, "how different are these two colours" is a plain Euclidean distance.

### Stage 2 — Detection (`detection.py`, 639 lines)

Finds the regions. Outputs a list of `Detection` objects (bbox, optional pixel mask, coarse label).

The plate is found first — `estimate_plate()` fits an ellipse — because it does double duty: it bounds the search, and later it sets the physical scale.

- **Primary: YOLOv8.** Pretrained on COCO, filtered to food-like classes. Gives boxes, not masks.
- **Fallback: a plate-aware segmenter** built on numpy/scipy. `plate_deviation()` measures how far each pixel departs from the plate's own background statistics; `food_mask()` thresholds that at `FOOD_DEVIATION_Z = 6.5` standard deviations; `_cluster_food()` splits the result into items. Adjacent similar regions are merged (`_merge_adjacent_similar`) and overlapping ones collapsed at IoU 0.55.

The fallback yields *pixel masks*; YOLO yields *boxes*, so `depth.mask_for()` derives a mask by intersecting each box with the global food mask.

### Stage 3 — Depth & volume (`depth.py`, 350 lines)

Turns a region into millilitres, then grams. Fully explained in [§5](#5-how-a-photo-becomes-grams).

- **Primary: MiDaS v3**, a monocular depth network — it estimates relative depth from a single ordinary photo.
- **Fallback: a distance-transform elevation proxy.** Assumes food is mounded — thickest in the middle, thinnest at the edges — and uses distance-from-edge as a stand-in for height.

**Keyed by category, not by dish.** `SERVING_DEPTH_CM` has 16 entries (15 real categories plus `unknown`), not 42. This is deliberate and load-bearing: adding a new dish to the catalog requires no geometry work at all, as long as it fits an existing category.

### Stage 4 — Classification (`classify.py`, 1593 lines)

The one custom-trained component. Takes a crop, returns a `Prediction` (label, confidence, alternatives).

Three engines, most capable first, **as strict alternatives — never blended**:

1. **`RemoteClassifier`** — a hosted `model_api/` deployment at `CLASSIFIER_URL`. All crops from one plate go in a single batched request, pre-resized to 300×300 (~20 KB each instead of ~500 KB). Three consecutive failures open a 120-second circuit breaker, so a dead host costs one timeout per photo, not six.
2. **`DishClassifier`** — a local checkpoint, when this deploy has torch. One stacked forward pass per plate.
3. **`SignatureClassifier`** — always available. A hand-built colour/texture prior over all 42 classes, using Lab anchors. Caps its own confidence at `CONFIDENCE_CAP = 0.88` because a heuristic should never claim near-certainty.

This file also contains the entire **training** implementation — `train()`, augmentation, mixup/cutmix, EMA, layer-decay parameter groups, warm-starting, stratified splitting and the training-log appender. Inference and training live together because they must agree on preprocessing exactly.

### Stage 5 — Nutrition (`nutrition.py`, 733 lines)

Label + grams → calories, macros and 13 micronutrients.

`resolve_per_100g(session, label)` falls through, in order:

| Order | Source | Result |
|---|---|---|
| 1 | `NutritionCache` table | whatever was cached |
| 2 | `COMPOSITION` — 42 IFCT 2017 rows | `ifct` |
| 3 | `COARSE_FALLBACK` — 8 broad groups | `category-average (estimated)` |
| 4 | USDA FoodData Central (needs `USDA_API_KEY`) | `usda` |
| 5 | `CATEGORY_FALLBACK["unknown"]` | `category-average (estimated)` |

**The bundled tables come first, deliberately.** They are offline, they cover all 42 labels, and for Indian dishes they are simply the better source. USDA is the extension for labels the tables don't carry.

> ⚠️ **The dangerous property of this chain: it never errors.** An unrecognised label produces *plausible generic numbers*, not a failure. The UI looks perfectly correct and the calories are fiction. This is why the 42 label names are treated as a join key and never renamed — see [§8](#8-the-42-dishes).

---

## 5. How a photo becomes grams

This is the part most people find surprising, so here it is end to end. No computer vision background assumed.

### Step 1 — the plate gives you a ruler

A photo has no scale. A bowl 20 cm away and a swimming pool 200 m away can occupy identical pixels. You need one object of known real-world size.

That object is the plate. The user says "this plate is 26 cm across." `estimate_plate()` has already measured its width in pixels — say 800.

```
cm_per_pixel = 26 cm / 800 px = 0.0325 cm/px
px_area_cm²  = 0.0325² = 0.00105625 cm² per pixel
```

Now every pixel has a real area, so any region has a real footprint:

```
area_cm² = area_px × px_area_cm²
```

### Step 2 — depth gives you the shape of the mound

The mask says *where* the food is; it says nothing about *how thick*. That's the depth map's job — either MiDaS, or the distance-transform proxy.

Crucially, `normalized_elevation()` keeps only the **shape** of the mound, not its absolute height. Monocular depth is good at "the middle is higher than the edge" and bad at "the middle is 3.1 cm high."

### Step 3 — a category prior gives you the absolute height

Absolute depth comes from a lookup, not from the network:

```python
mean_depth_cm = SERVING_DEPTH_CM[category] × clip((area_cm² / 60)^0.22, 0.65, 1.6)
# then clipped to [MIN_DEPTH_CM=0.2, MAX_DEPTH_CM=6.0]
```

`SERVING_DEPTH_CM` is the typical *mean* served depth per category — dal 3.3 cm, curry 3.1, rice 3.1, steamed 1.7, bread 0.6. **Mean** rather than peak is deliberate: mean depth is what portion tables actually pin down, and it makes the result independent of how peaked the mound happens to be.

`NOMINAL_AREA_CM2 = 60.0` is the footprint at which that number is exactly right. The exponent `0.22` encodes "a bigger helping is both wider *and* slightly deeper" — a weak effect on purpose. A square-root law would let a large footprint inflate weight almost quadratically.

### Step 4 — integrate

`integrate_volume()` is a Riemann sum. The elevation field is rescaled to have unit mean, then multiplied by the target mean depth and the per-pixel area:

```python
heights = elevation × (mean_cm × area_px / elevation.sum())
volume_ml = heights.sum() × px_area_cm²      # 1 cm³ == 1 ml
```

The normalisation is what makes this physically sensible: a dome and a flat slab with the same footprint and the same mean depth hold the same volume.

### Step 5 — density, then a sanity clamp

```python
weight_g = volume_ml × density_g_per_ml        # from nutrition.density_for(label)
weight_g, was_clamped = clamp_weight(weight_g, category)
```

`WEIGHT_BOUNDS` is a plausible served-weight envelope per category — curry 45–340 g, bread 18–220 g, condiment 8–120 g. Anything outside is clamped and flagged `weight_estimated=True` so the UI can say the number was corrected.

### The shortcut: countable food

For things you'd naturally enumerate — samosas, idlis, rotis — none of the above runs. Weight resolves in this order:

| Order | Condition | Weight from | `geometry.method` |
|---|---|---|---|
| 1 | countable label **with a count** | `count × grams-per-piece` | `piece-count` |
| 2 | has a region | `estimate_volume()` | `midas+geometry` / `shape-prior+geometry` |
| 3 | user-added, no region | `nutrition.nominal_portion_g()` | `nominal-portion` |

There is a fourth branch, `fallback-portion`, which returns `WEIGHT_BOUNDS[category][0]`. **It is unreachable in practice** — the scan drops area-less regions before they can reach it. It stays in the code because a threshold that can only be crossed by a bug should still not divide by zero.

### Why the plate diameter matters so much

Footprint scales with diameter **squared**, and weight a little faster still — measured at roughly d^2.3 end to end. A 20% error on the plate size is a ~50% error on calories. This is why the app asks rather than assumes, and why the answer stays correctable per meal.

---

## 6. Degradation: why it never crashes

Every stage has a primary implementation and a working fallback. The pipeline never returns an error where it could return a labelled estimate.

| Stage | Primary | Fallback | Reports |
|---|---|---|---|
| 2 Detection | YOLOv8 | Plate-aware segmenter (numpy/scipy) | `heuristic` |
| 3 Depth | MiDaS v3 | Distance-transform elevation proxy | `heuristic` |
| 4 Classification | EfficientNet-B3, hosted or local | 42-class colour/texture prior | `signature` |
| 5 Nutrition | Bundled IFCT 2017 | USDA, then flagged category average | `ifct` |

The top-level `engine` field summarises the run: **`full`** (trained classifier *and* both pretrained networks), **`partial`** (one of the two), **`heuristic`** (neither).

**This is not a debug mode.** A photo analysed with no weights present is a complete, honest answer — it is simply labelled as an estimate rather than a model prediction. Every stage reports which implementation produced its result, and that label reaches the UI.

Here is what honest degradation looks like in a real response (classical engine, one item of five):

```json
{
  "detected_label": "rice_or_bread",
  "classified_label": "rice_or_bread",
  "display_name": "Rice or Bread",
  "confidence": 0.18,
  "low_confidence": true,
  "estimated_weight_g": 91.6,
  "estimated_volume_ml": 117.4,
  "calories": 160.3,
  "nutrition_source": "category-average (estimated)",
  "alternatives": [
    { "label": "boiled_egg", "confidence": 0.1127 },
    { "label": "kheer",      "confidence": 0.1062 },
    { "label": "raita",      "confidence": 0.0992 }
  ],
  "geometry": {
    "area_cm2": 41.15, "mean_height_cm": 2.85, "peak_height_cm": 15.0,
    "density_g_per_ml": 0.78, "method": "shape-prior+geometry"
  }
}
```

`confidence: 0.18` is below the `0.28` unrecognized threshold, so the pipeline **declines to name a dish** and returns the coarse group `rice_or_bread` — a description it can stand behind — with `low_confidence: true`, an estimated-nutrition flag, and every alternative it considered. The geometry is still real arithmetic on a real mask.

The threshold is applied **per item**, not per response. On that same plate the other four items came back `Upma`, `Lentil or Yellow Dish`, `Curry or Gravy`, `Green Salad` — two named outright, three fell back. The response says which is which.

### Checking what's running

```bash
curl -s http://localhost:8000/api/health | python -m json.tool
```

```json
{
  "status": "ok",
  "version": "0.9.0-mvp",
  "engine": "heuristic",
  "database": "sqlite",
  "models": {
    "detection":      { "backend": "segmenter",   "version": "plate-segmenter-v1",    "ready": true },
    "depth":          { "backend": "shape-prior", "version": "distance-transform-v1", "ready": true },
    "classification": { "backend": "signature",   "version": "signature-v1",          "ready": true,
                        "trained_model": false, "classes": 42, "fallback": [] },
    "nutrition":      { "backend": "ifct",        "version": "ifct-2017/usda-fdc",    "ready": true }
  },
  "limits": {
    "max_upload_bytes": 10485760,
    "max_image_dimension": 1600,
    "max_items_per_plate": 6,
    "low_confidence_threshold": 0.55,
    "unrecognized_threshold": 0.28,
    "default_plate_diameter_cm": 26.0,
    "allowed_mime": ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
  }
}
```

The `limits` block exists so a client never has to hardcode a threshold it could ask for.

**All models load once at startup, never per request** (`pipeline.warm_models()`). `design.md` §12.1 calls this out as the single most common mistake that makes a working pipeline feel broken in a demo.

---

## 7. Every file in the repo

```
NutriGuide/
├── backend/          the single FastAPI service
├── mobile/           React Native + Expo app  ← the primary client
├── frontend/         vanilla-JS web app, no build step
├── model_api/        the classifier as a standalone service (optional)
├── data/             datasets + coverage manifests (gitignored)
├── design.md         design of record; §22 = as-built deviations
├── TRAINING_LOG.md   training runbook + a log of every run
├── readme.md         this file
└── .env.example      every setting, all optional
```

### `backend/` — the service

| File | Lines | What it does |
|---|---|---|
| **`main.py`** | 1059 | The FastAPI app and every route. Also: the startup `lifespan` hook that warms models, `_media_token`/`_verify_media_token` (HMAC image protection), `_item_out`/`_meal_out`/`_user_out` (ORM → response mapping), `_persist_items`, `_recalculate_meal`, `_apply_edits` (applies the review step's edits to a scan), and `_sweep_stale_drafts`. |
| **`pipeline.py`** | 806 | The orchestrator — the only file that knows all five stages exist. Defines the core dataclasses `AnalyzedItem`, `AnalysisResult`, `ScannedItem`, `ScanResult`. Entry points: `scan_image()` (phase 1), `analyze_scanned()` (phase 2), `analyze_image()` (both), `recompute_item()`, `remeasure_for_plate()`. Plus `warm_models()`, `model_status()`, `engine_name()`, `guess_piece_count()`. |
| **`imaging.py`** | 143 | Stage 1. Decode, EXIF-orient, downscale, Lab conversion, crop, JPEG encode. Defines `PreparedImage`. |
| **`detection.py`** | 639 | Stage 2. `FoodDetector` class, `estimate_plate()`, `food_mask()`, `plate_deviation()`, region clustering and merging. Defines `Detection` and `PlateEstimate`. |
| **`depth.py`** | 350 | Stage 3. `DepthEstimator` class, `estimate_volume()`, `integrate_volume()`, `mean_depth_cm()`, `weight_from_volume()`, `clamp_weight()`, `mask_for()`. Holds `SERVING_DEPTH_CM` and `WEIGHT_BOUNDS`. Defines `VolumeEstimate`. |
| **`classify.py`** | 1593 | Stage 4 **and** all training. `SignatureClassifier`, `RemoteClassifier`, `DishClassifier`, `Prediction`. Training: `train()`, `_augment()`, `_mix_batch()` (mixup/cutmix), `_EMA`, `_warm_start()`, `_param_groups()`, `_stratified_split()`, `_evaluate()`, `_append_training_log()`. Holds `SIGNATURES` → `CLASS_LIST` (the 42 labels) and `INPUT_RESOLUTION = 300`. |
| **`nutrition.py`** | 733 | Stage 5. `resolve_per_100g()`, `scale_nutrients()`, `sum_nutrients()`, `daily_value_percent()`, `fetch_usda()`, `catalog()`, `density_for()`, `category_of()`, `piece_weight_g()`, `nominal_portion_g()`. Holds `COMPOSITION` (42 rows), `COARSE_FALLBACK` (8), `CATEGORY_FALLBACK` (16), `PIECE_WEIGHTS`, `DETECTOR_ALIASES`. |
| **`drafts.py`** | 236 | The scan→review→analyze handoff. A `ScannedItem` carries a numpy mask, which cannot cross a JSON boundary — so `encode_regions()` packs all masks into a single PNG label map and `decode_regions()`/`restore_scan()` rebuild them on the second call. |
| **`db.py`** | 255 | SQLAlchemy 2.0 models: `User`, `Meal`, `MealItem`, `MealDraft`, `NutritionCache`. Also `FlexibleJSON`, a column type that is `JSONB` on Postgres and `JSON` on SQLite so one schema serves both. |
| **`schemas.py`** | 293 | Pydantic request/response contracts. The wire format lives here, not in `main.py`. |
| **`auth.py`** | 127 | JWT issuing/decoding, `pbkdf2_sha256` password hashing (stdlib `hashlib`, no bcrypt dependency), guest-user creation, and the `current_user` / `current_user_or_guest` FastAPI dependencies. |
| **`config.py`** | 138 | Every setting, all env-overridable, in one flat `Settings` class. Includes a minimal `.env` reader so `python-dotenv` isn't a dependency. |
| **`requirements.txt`** | — | Core deps only. Torch/ultralytics/timm are listed as commented-out optionals. |
| **`models/`** | — | Checkpoints land here (gitignored). `last_run.json` records the most recent training run. |
| **`uploads/`** | — | User meal photos (gitignored — never committed). |

### `backend/tools/` — developer scripts

| File | Lines | Purpose | Invocation |
|---|---|---|---|
| **`train_kaggle.py`** | 621 | Dataset assembly + training entry point. Maps third-party folder names onto our 42 labels via `resolve_label()`. | `python tools/train_kaggle.py --dry-run` |
| **`adopt_checkpoint.py`** | 478 | Inspect a **third-party** `.pt`/`.pth` and repack it for this API. Reports which of its labels map onto our catalog and refuses to fake the rest. | `python tools/adopt_checkpoint.py model.pth --labels classes.json --out out.pt` |
| **`download_hf_dataset.py`** | 357 | Downloads verified public Indian-food datasets from Hugging Face, SHA-256 deduplicates, and writes a license/coverage manifest. | `python tools/download_hf_dataset.py --clean` |
| **`make_samples.py`** | 209 | Generates the four demo plates in `frontend/samples/`. | `python tools/make_samples.py` |
| **`check_segmentation.py`** | 58 | Eyeball the detector on a single image. | `python tools/check_segmentation.py photo.jpg` |

**`train_kaggle.py` flags:** `--input --out --version --epochs --batch-size --learning-rate/--lr --patience --seed --workers --device --allow-cpu --dry-run --loose --min-per-class --max-per-class`, plus the shared recipe flags `--init-from --freeze-fraction --layer-decay --warmup-epochs --weight-decay --label-smoothing --mixup-alpha --cutmix-alpha --mix-prob --ema-decay`.

**`adopt_checkpoint.py` flags:** `--labels --out --version --loose --drop-unmapped --force`.

**`download_hf_dataset.py` flags:** `--dataset --split --output --cache --clean --loose`.

### `backend/tests/` — 270 tests, stdlib `unittest`

```bash
cd backend
python -m unittest discover -s tests -t . -q
```

| File | Tests | Covers |
|---|---|---|
| `test_api.py` | 92 | Every endpoint, auth, upload validation, both analysis phases, draft ownership, corrections, history |
| `test_training.py` | 60 | Label mapping, dataset assembly, loader determinism, checkpoint round-trip, CLI parity |
| `test_geometry.py` | 49 | px→cm scaling, volume integration, density, weight clamping, area-less regions dropped, plate re-measurement |
| `test_nutrition_math.py` | 37 | Per-100 g scaling, totals, %DV, USDA fallback, per-piece weights |
| `test_model_api.py` | 32 | Preprocessing parity between local and hosted, wire format, limits, circuit breaker, fallback chain |

The bias is deliberately toward **the math and the wire contracts**, because those break silently. A wrong `cm/px` scale doesn't raise an exception — it just reports 300 kcal as 900.

### `mobile/` — the React Native app

| File | Lines | What it does |
|---|---|---|
| **`App.js`** | 2286 | The entire app in one file. See [§13](#13-the-mobile-app) for the section map. |
| `package.json` | — | Expo 57, React 19.2, React Native 0.86, `expo-camera`, `expo-image-picker`, `expo-secure-store`, `react-native-svg`. |
| `app.json` | — | Expo project config. |

### `frontend/` — the web app

Vanilla ES modules, **no build step, no `node_modules`**. Served by FastAPI when `SERVE_FRONTEND=true` (the default). `design.md` §22.1 explains the departure from React.

| Path | Purpose |
|---|---|
| `index.html` | The single page. |
| `src/main.js` | Route table, lazy page imports. |
| `src/router.js` | Hash router. |
| `src/store.js` | Client state. |
| `src/api.js` | Typed wrapper over every endpoint; holds the bearer token. |
| `src/charts.js` | Hand-rolled SVG charts. |
| `src/dom.js` | Tiny element-builder helper (the "no framework" substitute). |
| `src/pending.js` · `toast.js` · `tooltip.js` | In-flight request tracking, notifications, tooltips. |
| `src/components/shell.js` · `meal.js` | App chrome; the meal-card component. |
| `src/pages/` | `home`, `analyzing`, `review`, `results`, `today`, `history`, `method`, `settings`, `auth`, `notfound`. |
| `src/styles/` | `tokens`, `base`, `layout`, `components`, `pages`, `anim`. |
| `samples/` | Four generated demo plates (`design.md` §22.10 — illustrations, not photographs). |

**Is it legacy?** No. It calls both `/meals/scan` and `/meals/analyze`, so it implements the current two-phase flow and has a `review` page. The mobile app is the primary client now, but the web app is current and functional.

### `model_api/` — the classifier as a service

Optional. Stage 4 is the only component that needs torch, so hosting it separately keeps the main API deployable on a 512 MB box.

| File | Lines | What it does |
|---|---|---|
| `app.py` | 337 | A minimal FastAPI service. `GET /` (metadata), `GET /health` (class list + readiness), `POST /classify` (batched crops → predictions, bearer-token protected). `resolve_checkpoint()` pulls weights from the Hugging Face Hub at startup via `HF_REPO_ID`, so the container image carries no 50 MB binary. |
| `requirements.txt` | — | torch, torchvision, timm, fastapi. |
| `Dockerfile` | — | Container build for the Space (SDK `docker`). |
| `README.md` | — | Hugging Face Space card. Its YAML frontmatter sets `app_port: 7860`, which is what makes the Space route traffic to the service. |

Point the backend at it with `CLASSIFIER_URL`. `test_model_api.py` holds the hosted and local paths to **bit-identical preprocessing and identical labels** — which machine served a photo cannot change what the user is told.

### Root files

| File | What it is |
|---|---|
| `design.md` | The design of record, 22 sections. §1–21 are the specification as written; **§22 records every place the code deliberately departs from it**, with reasoning. Read §22 before assuming a mismatch is a bug. |
| `TRAINING_LOG.md` | The training runbook (which datasets to attach, how to read the coverage table, how to publish weights, how to deploy `model_api/`) **and** the append-only log of every run. `classify._append_training_log()` writes to it automatically, including failed runs. |
| `.env.example` | Every setting with commentary. All optional. |
| `.gitignore` | Excludes weights (`*.pt`, `*.pth`), `data/`, `backend/uploads/`, `*.db`, `.env`, `node_modules/`. |

---

## 8. The 42 dishes

`classify.CLASS_LIST` and `nutrition.COMPOSITION` share exactly these 42 keys, grouped by the 15 geometry categories:

| Category | Dishes |
|---|---|
| **curry** (10) | `paneer_butter_masala` `palak_paneer` `chole_masala` `rajma_masala` `mixed_veg_curry` `butter_chicken` `chicken_curry` `fish_curry` `egg_curry` `pav_bhaji` |
| **bread** (5) | `roti_chapati` `naan` `paratha` `poori` `dosa` |
| **rice** (4) | `plain_rice` `jeera_rice` `veg_biryani` `chicken_biryani` |
| **fried** (4) | `medu_vada` `samosa` `papad` `french_fries` |
| **grain** (3) | `upma` `poha` `pasta_red_sauce` |
| **dal** (2) | `dal_tadka` `sambhar` |
| **dairy** (2) | `curd_yogurt` `raita` |
| **dessert** (2) | `gulab_jamun` `kheer` |
| **dry_sabzi** (2) | `aloo_gobi` `bhindi_masala` |
| **fruit** (2) | `banana` `apple` |
| **protein** (2) | `boiled_egg` `grilled_chicken` |
| **condiment** (1) | `coconut_chutney` |
| **fast_food** (1) | `pizza_slice` |
| **salad** (1) | `green_salad` |
| **steamed** (1) | `idli` |

### Why these names are a join key, not labels

These 42 snake_case strings are the **foreign key between three independently maintained tables**: `classify.SIGNATURES`/`CLASS_LIST`, `nutrition.COMPOSITION`, and `nutrition.PIECE_WEIGHTS` (which has an import-time guard against drift).

Because `resolve_per_100g()` never errors ([§4](#stage-5--nutrition-nutritionpy-733-lines)), an unmapped label produces **plausible generic numbers rather than a failure**. That is the dangerous mode: the UI looks correct and the calories are invented.

**So the rule is: never map a food label onto a *different* dish just to make it resolve.** If a model predicts something the catalog has no row for, it must degrade to a coarse group (reported as `category-average (estimated)`) or be dropped from the model's head — never silently become a neighbouring dish.

The code already enforces this. `tools/train_kaggle.py` keeps approximate matches in `LOOSE_ALIASES` behind an explicit `--loose` flag, and excludes with written reasons:

- `biryani` — can't tell veg from chicken
- `cholebhature` — two dishes, one label
- `fried_rice` — is not `jeera_rice`

When adopting any third-party checkpoint, route its labels through `train_kaggle.resolve_label(name, loose=False)` and **report the misses instead of forcing them**. `tools/adopt_checkpoint.py` is the checkpoint-side implementation of exactly this rule.

---

## 9. Model status — the honest numbers

### The production checkpoint: `efficientnet_v1.pt`

| | |
|---|---|
| Trained | 2026-09-03 |
| Hardware | Google Colab, Tesla T4 (Kaggle would not grant the accelerator without identity verification) |
| Dataset | 2,500 images across **24 classes** |
| Epochs | 22, early-stopped |
| **Val top-1** | **77.81%** |
| **Test top-1** | **78.46%** |
| Target | ≥85% (`design.md` §17) |

**v1 is below target and is explicitly for integration testing — not a final accuracy claim.**

Weakest per-class F1: `grilled_chicken` 0.333, `mixed_veg_curry` 0.560, `chole_masala` 0.571, `fish_curry` 0.571, `dal_tadka` 0.588.
Strongest: `french_fries` 1.000, `gulab_jamun` 0.933, `poori` 0.933.

The failures cluster in the **brown-gravy group**, where four labels differ by ingredient rather than by appearance. That means the gap to 85% is unlikely to close with more epochs — it wants more images for the gravy labels, or an acceptance that some of them are one class.

### The coverage gap — separate from, and bigger than, the accuracy gap

**v1's head has 24 outputs. The nutrition catalog has 42 classes.** It physically cannot predict the other 18:

```
plain_rice     jeera_rice      veg_biryani     chicken_biryani
paratha        sambhar         medu_vada       upma
rajma_masala   egg_curry       boiled_egg      curd_yogurt
raita          coconut_chutney green_salad     papad
banana         apple
```

This is not a confidence-threshold issue — there is no output neuron for these dishes. A plate of rice and dal can only ever come back as dal. **"78% accurate" therefore understates the real-world miss rate.**

The 24 v1 *can* predict:

```
aloo_gobi  bhindi_masala  butter_chicken  chicken_curry  chole_masala
dal_tadka  dosa  fish_curry  french_fries  grilled_chicken  gulab_jamun
idli  kheer  mixed_veg_curry  naan  palak_paneer  paneer_butter_masala
pasta_red_sauce  pav_bhaji  pizza_slice  poha  poori  roti_chapati  samosa
```

### The unpromoted candidate: `efficientnet_mps_smoke.pt`

A 1-epoch run on Apple MPS (2026-09-04), warm-started from v1, 5,225 images across 20 classes. It scored **val 88.89% / test 85.86%** — which clears the target on held-out data.

**It was not promoted.** `last_run.json` records `status: candidate_not_promoted`, because real plated-sample smoke checks still overpredicted `idli`. A single epoch that beats a 22-epoch run on held-out images, while failing on real plates, is a signal about the split — not a better model. It needs a longer run and crop-level evaluation before it replaces v1.

Do not quote 85.86% as this project's accuracy. **The number that serves users is 78.46%.**

### Before promoting any checkpoint

- [ ] Test top-1 ≥ 85%, with val and test within a few points (a large gap means the split leaked)
- [ ] No trainable class with F1 below ~0.5
- [ ] No `no mapping` folder with hundreds of images in the coverage table
- [ ] `GET /health` on the Space returns the expected class list
- [ ] A real photo through the API returns plausible labels, not just a good test number

---

## 10. Configuration reference

Everything is optional. `cp .env.example backend/.env` and uncomment what you need — `config._load_dotenv()` checks `backend/.env` first, then the repo root `.env`, so either location works. Settings live in `backend/config.py`.

### Core

| Env var | Default | Controls |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/nutriai.db` | Set a `postgresql+psycopg://` URL to switch. Nothing else changes. |
| `JWT_SECRET` | `dev-only-insecure-secret-change-me` | **Set this in production** — otherwise anyone can mint a token. |
| `JWT_TTL_SECONDS` | `2592000` (30 days) | Token lifetime. |
| `SERVE_FRONTEND` | `true` | Serve `frontend/` from the API process. |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8000` | Comma-separated allowlist. |
| `UPLOAD_DIR` | `backend/uploads` | Point at a persistent volume in production. |
| `MODEL_DIR` | `backend/models` | Overridable because training doesn't run where the service does. |

### Upload limits

| Env var | Default | Controls |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Rejected above this. |
| `MAX_IMAGE_DIMENSION` | `1600` | Downscale target. |
| — | `image/jpeg, png, webp, heic, heif` | Allowed MIME types (not env-configurable). |

### Pipeline tuning

| Env var | Default | Controls |
|---|---|---|
| `DEFAULT_PLATE_DIAMETER_CM` | `26.0` | Assumed plate size. User-correctable per meal. |
| `LOW_CONFIDENCE_THRESHOLD` | `0.55` | Below this an item is flagged `low_confidence`. |
| `UNRECOGNIZED_THRESHOLD` | `0.28` | Below this the dish is reported as a coarse group instead of a wrong guess. |
| `MAX_ITEMS_PER_PLATE` | `6` | Cap on detections. |

### Models

| Env var | Default | Controls |
|---|---|---|
| `ENABLE_TORCH_MODELS` | `true` | Set `false` to skip torch entirely for a faster boot. |
| `YOLO_WEIGHTS` | `yolov8n.pt` | Detection weights. |
| `MIDAS_MODEL` | `MiDaS_small` | Depth model variant. |
| `CLASSIFIER_CHECKPOINT` | `MODEL_DIR/efficientnet_v1.pt` | Local stage-4 checkpoint. |

### Hosted classifier

| Env var | Default | Controls |
|---|---|---|
| `CLASSIFIER_URL` | *(empty)* | `/classify` URL of a `model_api/` deployment. Never required. |
| `CLASSIFIER_TOKEN` | *(empty)* | Bearer token for that service. |
| `CLASSIFIER_TIMEOUT_S` | `25.0` | Generous, because a free HF Space cold-starts. |
| `CLASSIFIER_TTA_PASSES` | `2` | Horizontal-flip test-time augmentation. Set `1` when latency matters more than accuracy. |

### Nutrition

| Env var | Default | Controls |
|---|---|---|
| `USDA_API_KEY` | *(empty)* | **Unset by default.** The bundled IFCT tables answer everything for the 42 labels; the key adds coverage for non-Indian foods. |
| `USDA_BASE_URL` | `https://api.nal.usda.gov/fdc/v1` | |
| `USDA_TIMEOUT_S` | `6.0` | |

---

## 11. API reference

All application routes are namespaced under `/api` (`design.md` §22.9). Full interactive reference at `/docs`; exact contracts in `backend/schemas.py`.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | no | Engine, version and readiness per stage, plus every limit |
| `POST` | `/api/auth/guest` | no | Session with no signup |
| `POST` | `/api/auth/register` | no | Email + password |
| `POST` | `/api/auth/login` | no | Email + password |
| `GET` | `/api/auth/me` | yes | Current user |
| `PATCH` | `/api/users/me/preferences` | yes | Calorie goal, default plate diameter |
| `POST` | `/api/meals/scan` | yes | **Phase 1.** `multipart/form-data`, field `image`. Names items, costs nothing |
| `POST` | `/api/meals/{draft_id}/analyze` | yes | **Phase 2.** Reviewed list + plate diameter → finished meal |
| `POST` | `/api/meals/analyze` | yes | Both phases in one call |
| `GET` | `/api/meals/{meal_id}` | yes | One meal |
| `DELETE` | `/api/meals/{meal_id}` | yes | Delete a meal and its image |
| `PATCH` | `/api/meals/{meal_id}/items/{item_id}` | yes | Correct a label or weight; nutrition recomputes |
| `PATCH` | `/api/meals/{meal_id}/plate` | yes | Correct plate diameter; measured weights re-measure (counted and hand-added ones don't) |
| `GET` | `/api/users/{user_id}/history` | yes | Paginated history |
| `GET` | `/api/users/{user_id}/summary` | yes | Daily/weekly totals and trend |
| `GET` | `/api/nutrition/catalog` | no | All 42 labels — what the correction UI offers |
| `GET` | `/api/nutrition/lookup?food=` | no | Per-100 g composition for one label |
| `GET` | `/media/{meal_id}?t=&size=` | token | The stored image; `size=thumb` for the thumbnail |

### Auth

`pbkdf2_sha256` password hashing via stdlib `hashlib` (no bcrypt dependency), JWTs signed HS256 via PyJWT.

Guest sessions exist because of `design.md` §13.1 — **no login wall before the first try**. `POST /api/auth/guest` mints a user row and a token in one call. Every authenticated response carries `token` and `user` so a guest session can be picked up mid-flight.

Send the token as `Authorization: Bearer <token>`.

### Media protection

`/media/{meal_id}` requires a `?t=` token — an HMAC over the meal id, handed out in the meal response. Meal ids alone are not enough to read someone's photo.

### The two-phase call, worked example

```bash
DRAFT=$(curl -sX POST http://localhost:8000/api/meals/scan \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@plate.jpg" \
  | python -c 'import sys,json;print(json.load(sys.stdin)["draft_id"])')

# Between these two calls is where a human edits the list.
# Here: keep item 0 but call it four samosas, and drop item 1 (a lemon wedge).
curl -X POST "http://localhost:8000/api/meals/$DRAFT/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"plate_diameter_cm": 26.0,
       "items": [{"index": 0, "label": "samosa", "piece_count": 4},
                 {"index": 1, "removed": true}]}'
```

---

## 12. Database schema

SQLAlchemy 2.0. **SQLite by default, PostgreSQL by configuration** — one schema serves both. The `FlexibleJSON` column type in `db.py` resolves to `JSONB` on Postgres and `JSON` on SQLite, which is the only place the two differ (`design.md` §22.2).

```
users ──1:N──> meals ──1:N──> meal_items
  │
  └──1:N──> meal_drafts

nutrition_cache  (standalone)
```

### `users`
`id` (str36, PK) · `email` (str320, unique, nullable) · `name` (str120, nullable) · `password_hash` (text, nullable) · `is_guest` (bool) · `preferences` (JSON) · `created_at` (datetime)

### `meals`
`id` (str36, PK) · `user_id` (FK → users, CASCADE, indexed) · `image_url` (text) · `image_width` / `image_height` (int) · `captured_at` (datetime, indexed) · `total_calories` / `total_protein_g` / `total_carbs_g` / `total_fat_g` (numeric 10,2) · `engine` (str40) · `model_versions` (JSON) · `plate_diameter_cm` (numeric 6,2) · `notes` (text) · `micros` (JSON)

### `meal_items`
`id` (str36, PK) · `meal_id` (FK → meals, CASCADE, indexed) · `position` (int) · `detected_label` / `classified_label` (str120) · `confidence` (numeric 5,4) · `estimated_weight_g` / `estimated_volume_ml` (numeric 10,2) · `calories` / `protein_g` / `carbs_g` / `fat_g` (numeric 10,2) · `user_corrected` / `low_confidence` / `weight_estimated` (bool) · `bbox` (JSON) · `nutrients` (JSON) · `alternatives` (JSON) · `nutrition_source` (str60)

### `meal_drafts`
Holds a scan between phase 1 and phase 2.
`id` (str36, PK) · `user_id` (FK → users, CASCADE, indexed) · `created_at` (datetime, indexed) · `image_width` / `image_height` (int) · `plate_diameter_cm` (numeric 6,2) · `engine` (str40) · `notes` (text) · `payload` (JSON)

Stale drafts are swept by `main._sweep_stale_drafts()`.

### `nutrition_cache`
`food_label` (str160, PK) · `source` (str60) · `per_100g_json` (JSON) · `cached_at` (datetime)

---

## 13. The mobile app

React Native via Expo. The whole app is `mobile/App.js` — 2,286 lines, one file, no navigation library.

```bash
cd mobile
npm install
npx expo start
```

Point it at your backend:

```bash
EXPO_PUBLIC_API_URL=http://192.168.1.50:8000 npx expo start
```

Defaults to `http://127.0.0.1:8000`, which works in the simulator but **not** on a physical device — use your machine's LAN IP.

### Section map of `App.js`

| Lines | Section |
|---|---|
| 28–31 | `API_URL`, and the SecureStore keys `nutriai.mobile.token` / `.user` / `.onboarded` |
| 33–72 | **`C`** — the colour token object (light mode), `SHADOW`, `SHADOW_SOFT`, platform insets |
| 74–124 | Formatting helpers: `formatKcal`, `formatGrams`, `percent`, `titleCase`, `greetingForHour`, date utilities |
| 125–183 | Networking: `request()`, `uploadMealImage()`, and session persistence (`persistSession`, `clearSession`, `readStoredSession`) |
| 184–511 | **`App()`** — root component; owns auth state, routing, data fetching |
| 512–644 | `Splash`, `BrandMark`, `Segments`, `Chip`, and the three hero illustrations |
| 645–730 | `SLIDES` + **`Onboarding`** — the "Get Started" flow |
| 731–931 | `Field` + **`AuthScreen`** — real login/register with guest upgrade |
| 932–1116 | Shared UI: `Banner`, `ProgressRing`, `TopBar`, `StatCard`, `CalendarStrip`, `ActionCard`, `MealRow`, `EmptyState` |
| 1117–1253 | **`HomeScreen`** |
| 1254–1457 | `BarChart`, `MetricCard`, **`StatisticScreen`** |
| 1458–1517 | **`MealsScreen`** |
| 1518–1715 | `DetectedPhoto`, `MacroPill`, `ItemCard`, **`ResultsScreen`** |
| 1716–1808 | `SettingRow`, `GOAL_STEPS`, **`ProfileScreen`** |
| 1809–1870 | **`CameraCapture`** — the camera modal |
| 1871–1925 | **`CorrectionModal`** — per-item label/weight correction |
| 1926–1958 | `TABS`, `BottomTabs`, `TabButton` |
| 1959–2286 | `StyleSheet.create` |

### Flow

```
Splash → Onboarding (first launch only) → AuthScreen → HomeScreen
                                                          │
              ┌───────────────┬─────────────┬─────────────┤
              ▼               ▼             ▼             ▼
           Home          Statistics       Meals        Profile
              │
              └── scan → CameraCapture → ResultsScreen → CorrectionModal
```

The token lives in **`expo-secure-store`** (Keychain on iOS, Keystore on Android), not `AsyncStorage`.

---

## 14. Training the classifier

**Training needs a GPU.** EfficientNet-B3 at 300 px is roughly 20× slower on CPU, and `train_kaggle.py` refuses to start without an accelerator rather than quietly burning an hour of quota going 20× too slow. (`--allow-cpu` exists to override this; you almost never want it.)

**`TRAINING_LOG.md` is the full runbook.** It covers which datasets to attach, how to read the coverage table, how to publish weights, how to deploy `model_api/`, and how to point the backend at it. It is also the log itself — `classify._append_training_log()` writes an entry for every run, including the bad ones, because a log that only records successes is not a log.

### On a hosted GPU notebook

```python
!pip install -q timm
!cp -r /kaggle/input/nutriai-backend/backend /kaggle/working/backend
%env MODEL_DIR=/kaggle/working/models

# ALWAYS dry-run first — prints the coverage table, touches no GPU
!python /kaggle/working/backend/tools/train_kaggle.py --dry-run

!python /kaggle/working/backend/tools/train_kaggle.py --epochs 24 --version v2
```

### Reproducible source build

```bash
python backend/tools/download_hf_dataset.py --clean
python backend/tools/train_kaggle.py --input data/raw/huggingface --dry-run
```

The last audit (2026-09-05) merged the publisher train splits of `SohlHealth/enhanced-indian-food-classification` (CC BY 4.0) and `bharat-raghunathan/indian-foods-dataset` (CC0) into **6,981 SHA-256-deduplicated images across 20 app labels**. Rejections are itemised in `data/raw/huggingface_manifest.json`.

### Warm-starting from an existing checkpoint

```bash
python tools/train_kaggle.py --init-from backend/models/efficientnet_v1.pt --epochs 20 --version v2
```

`_warm_start()` reports how many tensors loaded, how many were skipped, and — usefully — which head rows were reused versus newly initialised for classes the previous checkpoint didn't have.

### Adopting someone else's checkpoint

```bash
python tools/adopt_checkpoint.py third_party.pth --labels classes.json
```

Prints a report: architecture family, head width, how many of its labels map onto our 42, and which don't. Add `--out repacked.pt` to actually repack. It **refuses** to repack when the head width doesn't match EfficientNet-B3's 1536 features, and it will not invent a mapping for an unmatched dish — see [§8](#8-the-42-dishes).

---

## 15. Tests

```bash
cd backend
python -m unittest discover -s tests -t . -q      # 270 tests
```

Stdlib `unittest`, not pytest — no runner to install (`design.md` §22.7 records this departure from §14.3, which specified pytest).

---

## 16. Deployment

### The API — one container

```bash
pip install -r backend/requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `JWT_SECRET` and `DATABASE_URL`; point `UPLOAD_DIR` at a persistent volume.

It **fits a 512 MB instance** — measured at 321 MB peak RSS through boot plus one full five-item analysis, on macOS with no torch. Comfortable but not roomy, and that headroom is exactly why stage 4 is worth hosting elsewhere: importing torch in this process would roughly triple the floor.

### The classifier — optionally its own container

`model_api/` ships `app.py`, `requirements.txt` and a Hugging Face Space README. It pulls weights from the Hub at startup (`HF_REPO_ID`), so the image carries no 50 MB binary. Full procedure in `TRAINING_LOG.md` §5.

Then set `CLASSIFIER_URL` on the backend. Full → partial → heuristic degradation means a dead Space never takes the app down.

### The web frontend — no deployment step

`SERVE_FRONTEND=true` (the default) serves it from the API process. There is nothing to build.

---

## 17. Troubleshooting

**`/api/health` says `engine: heuristic` — is it broken?**
No. That's the classical engine, and it produces complete answers. Install torch to move to `full`. See [§6](#6-degradation-why-it-never-crashes).

**The mobile app can't reach the backend.**
`EXPO_PUBLIC_API_URL` defaults to `127.0.0.1`, which on a physical device means the phone itself. Use your machine's LAN IP and make sure both are on the same network.

**Calories look wildly wrong.**
Check the plate diameter first. Weight scales at roughly d^2.3, so a 20% error there is a ~50% calorie error. Correct it with `PATCH /api/meals/{id}/plate`.

**A dish came back as "Rice or Bread" / "Curry or Gravy".**
That's the unrecognized threshold (0.28) working as designed — the classifier wasn't confident enough to name a dish, so it returned a coarse group it can stand behind. Check `nutrition_source`; it will say `category-average (estimated)`.

**A dish it should know isn't being predicted.**
Check the [18 uncovered classes](#the-coverage-gap--separate-from-and-bigger-than-the-accuracy-gap). v1's head has no output for them.

**Four samosas came back as one item.**
Expected. The detector finds *regions*, not instances. Set `piece_count` in the review step.

**`ModuleNotFoundError: torch`.**
Intended — torch is optional. Either install it or ignore the log line; the pipeline already fell back.

**Training exits immediately with status 3.**
No accelerator detected. Run on a GPU notebook. `--allow-cpu` overrides but will be ~20× slower.

---

## 18. Known limitations

- **Plate diameter is the largest single error term.** It cannot be recovered from the photo. The review step asks directly rather than assuming, and it stays correctable — but a wrong answer there dominates everything downstream.
- **The detector finds regions, not instances.** Four samosas are one blob. Piece counting is the mitigation, not a fix; true instance separation is a detector change.
- **The classifier covers 24 of 42 catalog classes** and scores 78.46% on those. See [§9](#9-model-status--the-honest-numbers).
- **Training data carries approximate labels by construction.** Public datasets mapped onto our labels; `--loose` mappings are off by default and each is named in `train_kaggle.py`.
- **Nutrition reflects standard composition**, not home-preparation variance (added oil, ghee, sugar).
- **`frontend/samples/` are procedurally generated illustrations**, not photographs (`design.md` §22.10).
- **Built by one person, no code review** — mitigated with 270 tests concentrated on the math and the wire contracts.

---

## 19. Roadmap

- [x] Lean, solo-buildable architecture
- [x] Pretrained-vs-trained decision per stage
- [x] End-to-end vertical slice (detect → volume → classify → nutrition)
- [x] Multi-item platters
- [x] Web frontend + FastAPI backend
- [x] Tests around the math-heavy modules
- [x] GPU training path with dataset assembly
- [x] Hosted classifier service + backend fallback chain
- [x] Two-phase analysis with a human review step
- [x] **Train the classifier and record the numbers** — v1, 78.46%
- [x] React Native mobile app with onboarding and real login
- [x] Third-party checkpoint adoption tool
- [ ] **Close the coverage gap** — get the other 18 classes into the head
- [ ] **Close the accuracy gap** — more images for the brown-gravy cluster
- [ ] Deploy the Space, point a deployed backend at it
- [ ] Replace generated sample plates with real photographs
- [ ] Get a working demo in front of real users
- [ ] Evaluate, write up findings
- [ ] Post-validation: consider fundraising, revisit `design.md` §16

---

## 20. A note on health-linked features

Doctor-linked diet monitoring, medication-aligned guidance and disease-specific alerts are **not** being built in this phase.

They involve health-adjacent personal data and can brush against medical-device and health-data regulation (e.g. India's DPDP Act). This is not a scope decision to revisit casually — it needs a real legal and compliance review before any version of these features exists, regardless of team size. Full reasoning in `design.md` §15.

Nutri-AI is a nutrition-estimation tool. It is not a medical device and gives no medical advice.

---

## 21. References

- Google Research, *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food*
- *DepthCalorieCam: A Mobile Application for Volume-Based Food Calorie Estimation using Depth Cameras*, MADiMa 2019
- *DPF-Nutrition* — cross-modal RGB + depth fusion for volume estimation
- *Food Portion Estimation via 3D Object Scaling*, arXiv 2024
- *Computer vision-based food calorie estimation: dataset, method, and experiment* (ECUSTFD)
- Indian Food Composition Tables 2017, National Institute of Nutrition (ICMR)
- World Health Organization, 2024 obesity statistics
- Indian Council of Medical Research, 2023 report

---

<div align="center">

Built solo with Claude Code as the primary development partner.
`design.md` is the design of record · `TRAINING_LOG.md` is the runbook.

</div>
