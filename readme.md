# Nutri-AI

**AI-Based Calorie Prediction and Nutrient Estimation from Multi-Item Food Images**

Upload a photo of your meal. Nutri-AI detects every item on the plate, estimates how much of it there is, identifies exactly what dish it is, and returns a full calorie and nutrient breakdown — automatically, no manual logging.

> Idea stage, pre-funding. Built solo with Claude Code as the primary development partner. `design.md` is the design of record; §22 of it records where this code departs from §1–21 and why.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [What's In the MVP (and What's Not)](#whats-in-the-mvp-and-whats-not)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
8. [Engines and Degradation](#engines-and-degradation)
9. [Dataset & Models](#dataset--models)
10. [Training the Classifier](#training-the-classifier)
11. [Tests](#tests)
12. [Deployment](#deployment)
13. [Evaluation](#evaluation)
14. [Comparison with Existing Systems](#comparison-with-existing-systems)
15. [Roadmap](#roadmap)
16. [Known Limitations](#known-limitations)
17. [A Note on Health-Linked Features](#a-note-on-health-linked-features)
18. [References](#references)

---

## Problem Statement

Over 2.5 billion adults worldwide are overweight (WHO, 2024), and diet-related diseases account for over 60% of premature deaths in India (ICMR). People generally know healthy eating matters, but manual calorie tracking is tedious, and self-reported intake is typically underestimated by 25–30%. Nutri-AI removes that friction with a single photo.

## What's In the MVP (and What's Not)

**In scope now — and built:**
- 📷 One-click meal photo analysis (web app)
- 🍽️ Multi-item food detection on a single plate
- 📏 Portion/weight estimation via monocular depth + a documented plate-size assumption
- 🍛 Fine-grained dish classification across 42 labels — the one custom-trained model here
- 🔢 Automatic calorie + macro/micronutrient breakdown (13 micronutrients, with %DV)
- ✅ A review step between detection and measurement — confirm the item list, count pieces, and set the plate size before anything is weighed
- 📊 Meal history, a daily summary, and per-item correction

**Explicitly not in the MVP** (vision-stage only):
- Doctor-linked diet monitoring
- Medication-aligned diet guidance
- Disease-specific real-time alerts
- Native mobile app
- Multi-service backend / multiple databases / orchestration / CI-CD

See [A Note on Health-Linked Features](#a-note-on-health-linked-features) for why the health-linked items aren't just "later" but specifically gated on legal review.

## How It Works

The five stages run in two passes, with the user between them. The scan says
*what is on the plate*; the deep pass says *how much of it there is*. Nothing is
weighed or costed until the person who can see the photo has confirmed the list.

```
Photo of meal (web upload)
      │
      ▼
┌─ PHASE 1 · scan ─────────────────── POST /api/meals/scan ──────────────┐
│ 1. Input     — decode, EXIF-orient, downscale, plate ellipse fit       │
│ 2. Detection — YOLOv8 (pretrained)          → one region per item     │
│ 4. Classify  — EfficientNet-B3 (custom)     → the exact dish          │
│                                                                        │
│    A region with no measurable area is dropped here, not priced.       │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
   REVIEW — the user renames, removes, adds, counts pieces,
            and *then* gives the plate diameter
      │
      ▼
┌─ PHASE 2 · deep pass ───── POST /api/meals/{draft_id}/analyze ─────────┐
│ 3. Depth     — MiDaS v3 (pretrained)                                   │
│    + geometry — plate diameter → cm/px scale → ml → grams              │
│      ...or `count × grams per piece`, for food you'd enumerate         │
│ 5. Nutrition — IFCT 2017 tables + USDA FoodData → kcal, macros, micros │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
Breakdown shown in the web app, saved to history
```

Stages are numbered by the pipeline, not by running order — stage 4 moves ahead
of stage 3 because naming a dish needs no plate scale and MiDaS is the expensive
part. What stage 2 hands over depends on which engine ran: the classical
segmenter yields per-item pixel masks, plain YOLOv8 yields boxes and the masks
are derived by intersecting them with the plate's food mask (`design.md` §22.3).
Within the pipeline `plate_diameter_cm` reaches exactly one property, which is
what makes the seam this clean; `design.md` §22.13 has the full reasoning,
including the phantom-item bug that prompted the split and the 20% disagreement
that showed up in the one place the plate scale is applied outside that property.

Only stage 4 involves training. Stages 1, 2, 3 and 5 use pretrained models, geometry or lookups — full reasoning in `design.md` §7.

Every stage reports which implementation produced its answer, and that label reaches the UI. A photo analysed without weights present is still a complete answer; it is just labelled as an estimate rather than a model prediction. See [Engines and Degradation](#engines-and-degradation).

Single-shot analysis is still available for callers with no user to ask:
`POST /api/meals/analyze` takes a photo and a plate diameter and returns a
finished meal, exactly as before.

## Tech Stack

| Layer | Technology | Note |
|---|---|---|
| Frontend | Vanilla ES modules + hand-written CSS | No build step, no `node_modules`. `design.md` §22.1 explains the departure from React |
| Backend | Python + FastAPI (single service) | Serves the API *and* the frontend |
| Detection | YOLOv8 (Ultralytics), pretrained | Falls back to a numpy/scipy plate segmenter |
| Depth/Volume | MiDaS v3, pretrained + custom geometry | Falls back to a shape-from-mask elevation proxy |
| Classification | EfficientNet-B3, transfer learning | The one real training effort. Optionally hosted separately — see `model_api/` |
| Nutrition Data | IFCT 2017 tables (bundled) + USDA FoodData Central | Works offline; the API key is optional |
| Database | SQLite by default, PostgreSQL by config | One schema, both backends (`design.md` §22.2) |
| Tests | stdlib `unittest` | 246 tests, no runner to install |
| Deployment | One container for the API, optionally one for the model | Render / Railway / Fly / a VM / a HF Space |

Deliberately not used: NestJS, MongoDB, React Native, Kubernetes, CI/CD, MLflow, opencv. Good tools, wrong stage — `design.md` §16 says when to add them back.

## Project Structure

```
NutriGuide/
├── backend/                   # the single FastAPI service
│   ├── main.py                  # app + all routes                   (1059 lines)
│   ├── pipeline.py              # the 5 stages, split into 2 phases   (805)
│   ├── imaging.py               # stage 1 — decode, orient, plate fit (143)
│   ├── detection.py             # stage 2 — YOLOv8 | plate segmenter  (639)
│   ├── depth.py                 # stage 3 — MiDaS | shape prior + geometry (350)
│   ├── classify.py              # stage 4 — train + 3 inference engines (1048)
│   ├── nutrition.py             # stage 5 — IFCT/USDA + 42-dish table (733)
│   ├── drafts.py                # scan → review handoff: masks as a PNG (236)
│   ├── db.py                    # SQLAlchemy models (Postgres + SQLite)
│   ├── auth.py                  # JWT, guest sessions, pbkdf2 hashing
│   ├── schemas.py               # pydantic request/response contracts
│   ├── config.py                # every knob, all env-overridable
│   ├── models/                  # trained checkpoints land here (gitignored)
│   ├── tools/
│   │   ├── train_kaggle.py        # dataset assembly + training entry point
│   │   ├── download_hf_dataset.py # verified public dataset download + manifest
│   │   ├── make_samples.py        # generates frontend/samples/
│   │   └── check_segmentation.py  # eyeball the detector on one image
│   └── tests/                   # 246 tests, stdlib unittest
│
├── frontend/                  # no build step; served by FastAPI
│   ├── index.html
│   ├── samples/                 # 4 generated demo plates (design.md §22.10)
│   └── src/
│       ├── main.js              # route table, lazy page imports
│       ├── router.js  dom.js  store.js  api.js  charts.js
│       ├── pending.js  toast.js  tooltip.js
│       ├── components/          # shell.js, meal.js
│       ├── pages/               # home, analyzing, review, results, today,
│       │                        # history, method, settings, auth, notfound
│       ├── styles/              # tokens, base, layout, components, pages, anim
│       └── assets/
│
├── model_api/                 # stage 4 as a standalone service (optional)
│   ├── app.py  requirements.txt  Dockerfile  README.md
│
├── data/                      # raw/ and processed/ datasets (gitignored)
├── design.md                  # design of record; §22 = as-built deviations
├── readme.md                  # this file
├── TRAINING_LOG.md            # how to train + a log of every run
└── .env.example               # every setting, all optional
```

## Getting Started

### Prerequisites
- **Python 3.10+.** That is the whole list.
- No Node, no npm — the frontend has no build step.
- No PostgreSQL — SQLite is the default.
- No GPU. One is only needed to *train* the classifier, and that runs on Kaggle.

### 1. Clone and install

```bash
git clone <repo-url>
cd NutriGuide/backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That is fastapi, uvicorn, python-multipart, pydantic, SQLAlchemy, psycopg, PyJWT, pillow, numpy, scipy, scikit-learn and httpx — a little over 230 MB installed, of which scipy and scikit-learn are two thirds. Notably **not** torch, which is where another ~2.5 GB would have gone. See [Engines and Degradation](#engines-and-degradation).

The Postgres driver installs unconditionally and then sits unused on SQLite. At this size that is not worth an extras group.

### 2. Run it

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000>. No `.env`, no database setup, no weights: the app boots, the frontend is served from `/`, and analysis works on the classical engine. Interactive API docs at `/docs`.

The home page shows a strip of four sample plates — click one to run an analysis without uploading anything.

### 3. Configure only what you need

Everything is optional. `cp .env.example backend/.env` and uncomment what applies:

| Setting | Effect when unset |
|---|---|
| `JWT_SECRET` | A public dev constant is used. **Set this in production** — otherwise anyone can mint a token. |
| `DATABASE_URL` | SQLite at `backend/nutriai.db`. Set a `postgresql+psycopg://` URL to switch; nothing else changes. |
| `USDA_API_KEY` | The bundled IFCT 2017 tables answer everything. The key adds USDA coverage for non-Indian foods. |
| `CLASSIFIER_URL` | Stage 4 uses a local checkpoint if present, else the signature prior. Set it to use a hosted model. |
| `ENABLE_TORCH_MODELS` | `true`. Set `false` to skip torch entirely for a faster boot. |

### 4. Optional: enable the pretrained stages

```bash
pip install torch torchvision ultralytics timm
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"
```

Stages 2 and 3 pick the real networks up automatically on next start. `GET /api/health` will show `backend: yolov8` and `backend: midas` instead of the fallbacks. This is a ~2.5 GB install and is not required for anything to work.

## Usage

**Authenticate** (guest sessions need no signup):

```bash
TOKEN=$(curl -sX POST http://localhost:8000/api/auth/guest | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
```

**Analyse a plate.** Two calls, the way the app does it — scan, then cost the
reviewed list:

```bash
DRAFT=$(curl -sX POST http://localhost:8000/api/meals/scan \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@plate.jpg" \
  | python -c 'import sys,json;print(json.load(sys.stdin)["draft_id"])')

# Between these two calls is where a human edits the list. Here: keep item 0 but
# call it four samosas, and drop item 1 because it was a lemon wedge.
curl -X POST "http://localhost:8000/api/meals/$DRAFT/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"plate_diameter_cm": 26.0,
       "items": [{"index": 0, "label": "samosa", "piece_count": 4},
                 {"index": 1, "removed": true}]}'
```

Or in one call, when there is nobody to ask:

```bash
curl -X POST http://localhost:8000/api/meals/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@plate.jpg"
```

**Real response** from `frontend/samples/thali.jpg` on the classical engine. Abridged in four ways, all of them for length: the plate produced five items and one is shown in full; each item's `nutrients` block — the same 13 micronutrient keys, scaled to that item — is left out; `timings_ms` also carries `input`; and the body also carries `token` and `user`, which every authenticated response does so a guest session can be picked up. Everything present is verbatim.

```json
{
  "meal_id": "95dd573e-0360-4cd9-a976-5339945e16ee",
  "engine": "heuristic",
  "plate_diameter_cm": 26.0,
  "items": [
    {
      "id": "444c3c2f-d6da-470d-9316-2cc19c0ac47f",
      "position": 0,
      "detected_label": "rice_or_bread",
      "classified_label": "rice_or_bread",
      "display_name": "Rice or Bread",
      "confidence": 0.18,
      "low_confidence": true,
      "user_corrected": false,
      "weight_estimated": false,
      "estimated_weight_g": 91.6,
      "estimated_volume_ml": 117.4,
      "calories": 160.3,
      "protein_g": 4.0,
      "carbs_g": 31.1,
      "fat_g": 2.9,
      "nutrition_source": "category-average (estimated)",
      "alternatives": [
        { "label": "boiled_egg", "display_name": "Boiled Egg", "confidence": 0.1127 },
        { "label": "kheer", "display_name": "Kheer", "confidence": 0.1062 },
        { "label": "raita", "display_name": "Raita", "confidence": 0.0992 }
      ],
      "bbox": { "x": 0.26875, "y": 0.27656, "w": 0.25, "h": 0.44688 },
      "geometry": {
        "area_cm2": 41.15,
        "mean_height_cm": 2.85,
        "peak_height_cm": 15.0,
        "density_g_per_ml": 0.78,
        "method": "shape-prior+geometry",
        "position": 0,
        "coarse_confidence": 0.5779
      }
    }
  ],
  "totals": { "calories": 437.9, "protein_g": 13.6, "carbs_g": 63.9, "fat_g": 14.7 },
  "micronutrients": {
    "fiber_g": 6.5, "sugar_g": 1.3, "sodium_mg": 816.1, "calcium_mg": 18.8, "iron_mg": 0.7,
    "potassium_mg": 119.4, "magnesium_mg": 14.8, "zinc_mg": 0.4, "vitamin_a_mcg": 82.0,
    "vitamin_c_mg": 7.0, "vitamin_d_mcg": 0.0, "vitamin_b12_mcg": 0.0, "folate_mcg": 18.5
  },
  "daily_values": {
    "fiber_g": 21.7, "sugar_g": 2.6, "sodium_mg": 40.8, "calcium_mg": 1.9, "iron_mg": 3.9,
    "potassium_mg": 3.4, "magnesium_mg": 3.7, "zinc_mg": 3.3, "vitamin_a_mcg": 9.1,
    "vitamin_c_mg": 8.8, "vitamin_d_mcg": 0.0, "vitamin_b12_mcg": 0.0, "folate_mcg": 6.2
  },
  "low_confidence": true,
  "timings_ms": { "detection": 605.0, "depth": 316.1, "classification": 11.8, "volume": 127.8, "nutrition": 4.0, "total": 1140.4 },
  "model_versions": {
    "detection": "segmenter:plate-segmenter-v1",
    "depth": "shape-prior:distance-transform-v1",
    "classification": "signature:signature-v1",
    "nutrition": "ifct"
  }
}
```

Worth reading closely, because it is what honest degradation looks like: `confidence: 0.18` is below the 0.28 unrecognized threshold, so the pipeline declines to name a dish and returns the coarse group `rice_or_bread` — a description it can stand behind — with `low_confidence: true`, `nutrition_source: "category-average (estimated)"`, and every alternative it considered. The geometry is still real arithmetic on a real mask.

The other four items on this plate came back `Upma`, `Lentil or Yellow Dish`, `Curry or Gravy`, `Green Salad` — so the colour/texture prior named two of five outright and fell back to a coarse group for three. That mix is the point. The threshold is not a switch that turns the whole response into an estimate; it is applied per item, and the response says which is which.

**Every endpoint:**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Engine, version and readiness per stage, plus every limit |
| `POST` | `/api/auth/guest` | Session with no signup |
| `POST` | `/api/auth/register` · `/api/auth/login` | Email + password |
| `GET` | `/api/auth/me` | Current user |
| `PATCH` | `/api/users/me/preferences` | Calorie goal, default plate diameter |
| `POST` | `/api/meals/scan` | Phase 1. `multipart/form-data`, field `image`. Names the items, costs nothing |
| `POST` | `/api/meals/{draft_id}/analyze` | Phase 2. The reviewed list plus a plate diameter → a finished meal |
| `POST` | `/api/meals/analyze` | Both phases in one call, for a caller with no user to ask |
| `GET` | `/api/meals/{id}` | One meal |
| `DELETE` | `/api/meals/{id}` | Delete a meal and its image |
| `PATCH` | `/api/meals/{id}/items/{item_id}` | Correct a label or weight; nutrition recomputes |
| `PATCH` | `/api/meals/{id}/plate` | Correct the plate diameter; measured weights are re-measured against it (counted and hand-added ones aren't) |
| `GET` | `/api/users/{id}/history` | Paginated history |
| `GET` | `/api/users/{id}/summary` | Daily/weekly totals and trend |
| `GET` | `/api/nutrition/catalog` | All 42 labels — what the correction UI offers |
| `GET` | `/api/nutrition/lookup?food=` | Per-100 g composition for one label |
| `GET` | `/media/{meal_id}` | The stored image, `?size=thumb` for the thumbnail. Needs the `?t=` token the meal response hands out — an HMAC over the id, so ids alone are not readable |

Full reference at `/docs`, contracts in `backend/schemas.py`, design rationale in `design.md` §10.

## Engines and Degradation

Every stage has a primary implementation and a working fallback, so the pipeline never returns an error where it could return a labelled estimate (`design.md` §12.2, §22.3).

| Stage | Primary | Fallback | Fallback reports |
|---|---|---|---|
| 2 Detection | YOLOv8 | Plate-aware segmenter (numpy/scipy) | `heuristic` |
| 3 Depth | MiDaS v3 | Distance-transform elevation proxy | `heuristic` |
| 4 Classification | EfficientNet-B3, hosted or local | 42-class colour/texture signature prior | `signature` |
| 5 Nutrition | Bundled IFCT 2017 tables | USDA FoodData Central, then a flagged category average | `ifct` |

Stage 5 is the one that reads backwards from expectation, and deliberately: the bundled tables come *first*. They are offline, they cover all 42 labels, and for Indian dishes they are the better source. USDA is the extension for labels the tables do not carry, and a conservative category average — always flagged `estimated` — is the last resort. Full order in `backend/nutrition.py`.

The top-level `engine` field summarises the whole run: `full` (trained classifier **and** both pretrained networks), `partial` (one of the two), `heuristic` (neither).

Stage 4 specifically resolves in three steps, most capable first:

1. **Hosted** — `CLASSIFIER_URL` pointing at a `model_api/` deployment. All crops from one plate go in a single batched request, pre-resized to 300×300 (~20 KB each instead of ~500 KB). Three consecutive failures open a 120 s circuit breaker, so a dead host costs one timeout per photo rather than six.
2. **Local** — a checkpoint at `CLASSIFIER_CHECKPOINT`, when this deploy has torch. One stacked forward pass for the plate.
3. **Signature prior** — always available.

The hosted and local paths are held to bit-identical preprocessing and identical labels by `backend/tests/test_model_api.py`. Which machine served a photo cannot change what the user is told.

`GET /api/health` reports which engine is answering and what is underneath it:

```json
{
  "status": "ok",
  "version": "0.9.0-mvp",
  "engine": "heuristic",
  "database": "sqlite",
  "models": {
    "detection":      { "backend": "segmenter",   "version": "plate-segmenter-v1",    "ready": true },
    "depth":          { "backend": "shape-prior", "version": "distance-transform-v1", "ready": true },
    "classification": { "backend": "signature",   "version": "signature-v1", "ready": true,
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

The `limits` block is there so a client never has to hardcode a threshold it can ask for. The `0.28` above is the same number that produced the `rice_or_bread` answer earlier in this file.

All models load once at startup, never per request — `design.md` §12.1 calls this out as the single most common mistake that makes a working pipeline feel broken in a demo.

## Dataset & Models

| Component | Source | Trained here? |
|---|---|---|
| Detection | YOLOv8 pretrained | No |
| Depth | MiDaS v3 pretrained | No |
| Classification | Public Kaggle food datasets mapped onto our 42 labels, fine-tuned on EfficientNet-B3 | **Yes — the core training component** |
| Nutrition data | Indian Food Composition Table 2017 (bundled, 42 dishes) + USDA FoodData Central | No |

The class list has 42 labels; a given checkpoint typically trains on the 20–30 of them the attached datasets actually cover, and the signature prior answers for the rest. That was a deliberate choice over padding thin classes — `design.md` §22.5. `backend/tools/download_hf_dataset.py` can download the verified public Indian-food sources and writes a license/coverage manifest before training.

## Training the Classifier

Training runs on **Kaggle's GPU**, not locally. EfficientNet-B3 at 300 px is roughly 20× slower on CPU, and `tools/train_kaggle.py` refuses to start without an accelerator rather than quietly spending an hour of quota going 20× too slow.

The short version:

```python
!pip install -q timm
!cp -r /kaggle/input/nutriai-backend/backend /kaggle/working/backend
%env MODEL_DIR=/kaggle/working/models

# always dry-run first — prints the coverage table, touches no GPU
!python /kaggle/working/backend/tools/train_kaggle.py --dry-run
!python /kaggle/working/backend/tools/train_kaggle.py --epochs 24 --version v1
```

For a reproducible source build outside Kaggle, download only publisher train
splits first, then attach or copy the resulting ImageFolder tree:

```bash
python backend/tools/download_hf_dataset.py --clean
python backend/tools/train_kaggle.py --input data/raw/huggingface --dry-run
```

The classifier averages the original crop and a horizontal flip at inference;
set `CLASSIFIER_TTA_PASSES=1` when latency matters more than the small accuracy
gain.

**`TRAINING_LOG.md` is the full runbook** — which datasets to attach, how to read the coverage table, how to publish the weights, how to deploy `model_api/`, and how to point the backend at it. It is also the log itself: `classify.train()` appends an entry for every run, including the bad ones.

Target is test top-1 ≥ 85% (`design.md` §17). Below that, more data beats more epochs nearly every time.

## Tests

```bash
cd backend
python -m unittest discover -s tests -t . -q      # 246 tests, ~133 s
```

| File | Tests | Covers |
|---|---|---|
| `test_api.py` | 92 | Every endpoint, auth, upload validation, both analysis phases, draft ownership, corrections, history |
| `test_geometry.py` | 49 | px→cm scaling, volume integration, density, weight, area-less regions dropped, plate re-measurement |
| `test_nutrition_math.py` | 37 | Per-100 g scaling, totals, %DV, USDA fallback, per-piece weights |
| `test_training.py` | 36 | Label mapping, dataset assembly, loader determinism, checkpoint round-trip, CLI parity |
| `test_model_api.py` | 32 | Preprocessing parity, wire format, limits, circuit breaker, fallback chain |

The bias is toward the math and the contracts, per `design.md` §14.3 — those are what break silently. A wrong `cm/px` scale does not raise; it just reports 300 kcal as 900. (§14.3 says `pytest`; the runner departure is recorded in §22.7.)

## Deployment

**The API** is one container. `pip install -r backend/requirements.txt`, then
`uvicorn main:app --host 0.0.0.0 --port $PORT`. Set `JWT_SECRET` and `DATABASE_URL`; point `UPLOAD_DIR` at a persistent volume.

It fits a 512 MB instance as long as stage 4 is hosted or falls back — measured at 321 MB peak RSS through boot plus one full five-item analysis, on macOS with no torch. That is comfortable but not roomy, and the headroom is why stage 4 is worth hosting elsewhere: importing torch in this process would roughly triple the floor.

**The classifier** is optionally its own container — `model_api/` ships an `app.py`, a `Dockerfile` and a Hugging Face Space README. It pulls weights from the Hub at startup, so the image carries no 50 MB binary. Full procedure in `TRAINING_LOG.md` §5.

**The frontend** needs no deployment step. `SERVE_FRONTEND=true` (the default) serves it from the API process; there is nothing to build.

## Evaluation

- **Detection:** mAP@0.5 on held-out images.
- **Volume/portion:** mean % error vs. ground truth, against published ranges (Nutrition5k, ECUSTFD).
- **Classification:** top-1 accuracy and per-class F1 — both recorded per run in `TRAINING_LOG.md`, with the five weakest classes named.
- **End-to-end calorie estimate:** MAE (%).

## Comparison with Existing Systems

| Feature | Im2Calories (Google) | NutriNet | FoodAI (NUS) | DPF-Nutrition | **Nutri-AI** |
|---|---|---|---|---|---|
| Multi-item detection | ✗ | ✓ | ✓ | ✓ | ✓ |
| Volume/portion estimation | ✗ | ✗ | ✗ | ✓ | ✓ |
| Fine-grained classification | ✗ | ✓ | ✗ | ✗ | ✓ |
| Macro/micronutrient estimation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Cultural adaptation (Indian dataset) | ✗ | ✗ | ✗ | ✗ | ✓ |

Full table in `design.md` §19.

## Roadmap

- [x] Finalize lean, solo-buildable architecture
- [x] Confirm pretrained-vs-trained decisions for each pipeline stage
- [x] Build the end-to-end vertical slice (detect → volume → classify → nutrition)
- [x] Extend to multi-item platters
- [x] Build the web frontend + FastAPI backend, wire together
- [x] Tests around the math-heavy modules
- [x] GPU-capable training path with dataset assembly from public Kaggle sets
- [x] Hosted classifier service + backend fallback chain
- [ ] Attach the datasets, run the training, record the numbers in `TRAINING_LOG.md`
- [ ] Deploy the Space, point a deployed backend at it
- [ ] Replace the generated sample plates with real photographs
- [ ] Get a working demo in front of real users
- [ ] Evaluate results, write up findings
- [ ] Post-validation: consider fundraising, then revisit the Scale-Up Path in `design.md` §16

## Known Limitations

- Volume estimation needs a plate diameter and cannot recover one from the photo. The review step asks for it directly rather than assuming silently, and it stays correctable per meal — but a wrong answer there is the largest single error term: footprint scales with the diameter squared, and weight a little faster still, near d^2.3 measured end to end.
- The detector finds *regions*, not instances: four samosas come back as one blob, not four detections. Piece counting is the mitigation, not a fix — the count is seeded from area and the user corrects it. True instance separation is a detector change (`design.md` §22.13).
- The classifier trains on public datasets mapped onto our labels, so some classes carry approximate labels by construction. `--loose` mappings are off by default and each one is named in `train_kaggle.py`.
- Nutrition values reflect standard ingredient composition and may not capture home-preparation variance (added oil, ghee, sugar).
- No mobile app; web only. The UI is responsive down to 320 px.
- `frontend/samples/` are procedurally generated illustrations, not photographs (`design.md` §22.10).
- Built by one person, no code review — mitigated with 246 tests concentrated on the math and the wire contracts.

## A Note on Health-Linked Features

Doctor-linked diet monitoring, medication-aligned guidance, and disease-specific alerts are not being built in this phase. They involve handling health-adjacent personal data and can brush against medical-device and health-data regulation (e.g. India's DPDP Act). This isn't a scope decision to revisit casually later — it needs a real legal/compliance review before any version of these features is built, regardless of team size. Full reasoning in `design.md` §15.

## References
- Google Research, *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food*.
- *DepthCalorieCam: A Mobile Application for Volume-Based Food Calorie Estimation using Depth Cameras*, MADiMa 2019.
- *DPF-Nutrition* — cross-modal RGB + depth fusion for volume estimation.
- *Food Portion Estimation via 3D Object Scaling*, arXiv 2024.
- *Computer vision-based food calorie estimation: dataset, method, and experiment* (ECUSTFD).
- Indian Food Composition Tables 2017, National Institute of Nutrition (ICMR).
- World Health Organization (WHO), 2024 obesity statistics.
- Indian Council of Medical Research (ICMR), 2023 report.
