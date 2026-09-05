# Nutri-AI — System Design Document

**Project:** AI-Based Calorie Prediction and Nutrient Estimation from Multi-Item Food Images
**Stage:** Idea stage, pre-funding
**Team:** Solo founder/builder, using Claude Code (Max) as the primary build partner
**Document Type:** Technical Design Document (TDD)
**Version:** 3.0 — adds operational, UX, and completion-criteria detail
**Status:** Draft — architecture finalized for MVP phase

---

## 1. Executive Summary

Nutri-AI lets a user photograph a plate of food and automatically receive the identified food items, their estimated portion sizes, and a full calorie/nutrient breakdown — no manual logging.

This document reflects the real constraints of this phase: one person, building with an AI coding agent, at idea stage, with no fixed deadline and no funding yet. The goal is not to design for a future 21-person engineering org — it's to design something a single builder can finish, debug, demo, and use to validate the idea.

---

## 2. Why This Version Looks Different From a "Startup-Scale" Design

The right question at idea stage is not "what would a funded startup use?" — it's "what is the smallest system that proves the core idea works, that one person plus an AI agent can build, run, and confidently demo?"

**Concretely:**
- One backend service (FastAPI does both API and ML orchestration), not two.
- One database (PostgreSQL), not two.
- Web app first, not mobile.
- No Docker/CI/CD/container orchestration required to build the MVP.
- No MLflow — a plain training log is sufficient for one model.
- Doctor-linking, medication guidance, and disease alerting are **removed from MVP scope entirely** — they carry real regulatory weight and belong in the pitch/vision only, not the codebase, until there's been a legal review.

---

## 3. Problem Statement

To build an AI-powered product where users can upload a picture of their meal, and the system automatically identifies the food items, estimates portion sizes, and provides instant calorie and nutrient information.

### 3.1 Motivating Data
- WHO (2024): over 2.5 billion adults overweight worldwide — roughly 1 in 3 people.
- Obesity rates have tripled since 1975.
- Over 650 million adults are clinically obese, contributing to Type-2 Diabetes, Hypertension, Cardiovascular disease, and NAFLD.
- India: 135 million obese individuals, 101+ million diabetics (ICMR, 2023).
- ICMR: diet-related diseases account for over 60% of premature deaths in India.
- Self-reported calorie intake is typically underestimated by 25–30%.

### 3.2 Core Insight
People generally know healthy eating matters — they don't track it accurately because manual logging is tedious. Automating detection + portion estimation + nutrient lookup removes that friction.

---

## 4. Objectives

### 4.1 Near-term (this phase — solo, pre-funding)
1. Prove the core loop works: photo in → correct-enough food identification, portion estimate, calorie/nutrient output.
2. Build something demo-able to validate the idea with real users and, eventually, investors.
3. Keep the codebase small enough that one person + an AI agent can hold the whole system in their head.

### 4.2 Long-term (post-funding / post-team — vision, not current build target)
4. Promote healthy eating at scale via engaging dashboards and insights.
5. Extend into doctor-linked and medically-aware guidance — after legal/compliance review and a real health-data architecture.
6. Reach a broad audience — students, fitness users, healthcare-linked users.

---

## 5. Scope

### 5.1 MVP Scope (build this now)
- Single-image, multi-item food detection.
- Monocular depth-based, geometry-driven volume/weight estimation.
- Fine-grained dish classification — the one real training job.
- Calorie and macro/micronutrient computation via nutrient-database lookup.
- A single web app (upload photo → see results) backed by one API service.
- Basic meal history.

### 5.2 Explicitly Out of Scope for MVP
- Doctor-linked diet monitoring, medication-aligned guidance, disease-specific alerting.
- Native mobile apps.
- Multi-service backend, multiple databases, container orchestration, CI/CD pipelines.
- Gamification, social features, IoT integration.

### 5.3 Why No Volume-Model Training
Monocular volume estimation is an open research problem; even strong published systems (Nutrition5k, DPF-Nutrition) report 15–20%+ calorie MAE. Instead:
- **MiDaS**, pretrained, used purely for inference.
- Relative depth is converted to real-world volume using a default plate-diameter assumption (~26cm), following the same geometric logic as ECUSTFD and Nutrition5k.
- This is a documented simplifying assumption, stated openly rather than hidden behind a fragile mechanism like requiring a reference object in every photo.

---

## 6. System Architecture (MVP)

### 6.1 High-Level Data Flow

```
┌─────────────────────┐
│   Web App (React)    │   photo upload / capture (browser)
└──────────┬────────────┘
           │ image
           ▼
┌───────────────────────────────────────────────────────────┐
│              Single Backend Service (FastAPI, Python)        │
│                                                                │
│  API layer: auth, upload handling, request validation         │
│                                                                │
│  ML Pipeline (models loaded once at startup — see §12.1):      │
│    Stage 1: Food Detection (YOLOv8, pretrained)                │
│    Stage 2: Depth Estimation (MiDaS, pretrained, no training)  │
│    Stage 3: Volume/Weight (geometry + plate-diameter default)  │
│    Stage 4: Fine-Grained Classification (EfficientNet-B3)      │
│              — THE ONE TRAINED MODEL                            │
│    Stage 5: Nutrition Lookup (USDA FoodData Central API)        │
│                                                                │
└──────────┬─────────────────────────────────────────────────┘
           │ structured JSON result
           ▼
┌─────────────────────┐
│   PostgreSQL          │   users, meals, meal_items (single DB)
└──────────┬────────────┘
           │
           ▼
┌─────────────────────┐
│   Web App Dashboard   │   annotated image, nutrition breakdown, history
└─────────────────────┘
```

### 6.2 Component Responsibilities

| Component | Responsibility | Trained by you? |
|---|---|---|
| Web App (React) | Photo upload, results display, basic history view | No ML |
| FastAPI Backend | Auth, request handling, orchestrates all 5 ML stages, returns JSON | No |
| YOLOv8 | Localizes each food item, coarse label | No (pretrained; light fine-tune optional) |
| MiDaS | Per-pixel relative depth map | No (pretrained, inference only) |
| Volume/Weight module | Depth + plate-diameter assumption → volume (ml) → weight (g) | No (pure code/geometry) |
| EfficientNet-B3 | Exact dish identity | **Yes — the one real training job** |
| Nutrition Lookup | (dish, weight) → kcal + macro/micronutrients | No (API lookup) |
| PostgreSQL | Persists users, meals, meal items | N/A |

---

## 7. Detailed Pipeline / Module Design

### 7.1 Stage 1 — Input Layer
- Accepts a single RGB image (browser file upload or camera capture via browser APIs).
- Preprocessing: resize, normalize, basic brightness/contrast correction.
- No reference-object requirement placed on the user — plate-diameter default assumption is used instead.

### 7.2 Stage 2 — Food Detection (YOLOv8)
- Pretrained YOLOv8 checkpoint; optional light fine-tune on Food-101/IndianFood101, not required for MVP to function.
- Output: bounding box + coarse label + confidence per item.

### 7.3 Stage 3 — Depth & Volume Estimation (MiDaS + Geometry)
- MiDaS v3 run purely for inference.
- Segment food-item pixels per detected box.
- Calibrate relative depth using the plate-diameter default (user-correctable).
- Sum per-pixel volume over food pixels → volume (ml) → weight (g) via a food-density lookup table.

### 7.4 Stage 4 — Fine-Grained Classification (EfficientNet-B3) — THE TRAINED MODEL
- Architecture: EfficientNet-B3 backbone (ImageNet-pretrained) + fine-tuned classification head.
- Strategy: transfer learning — freeze early layers, fine-tune later blocks + head.
- Input: cropped region per detected item.
- Output: exact dish label + confidence.

### 7.5 Stage 5 — Nutrition Lookup
- USDA FoodData Central API + Indian Food Composition Table fallback.
- (dish label, weight in grams) → per-item kcal/protein/carbs/fat/vitamins/minerals.

### 7.6 Output & Aggregation
- Combine per-item results into one JSON response, persist to PostgreSQL, render in the web app.

---

## 8. Data Design

### 8.1 Dataset Sources

| Dataset | Purpose |
|---|---|
| Food-101 | General pretraining/detection baseline |
| IndianFood101 | Broaden Indian dish coverage |
| Custom curated dataset (500–1,000 images across target classes) | Fine-tune the EfficientNet-B3 classifier |

### 8.2 Database Schema (PostgreSQL only)

```sql
users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE,
  name TEXT,
  created_at TIMESTAMP
)

meals (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  image_url TEXT,
  captured_at TIMESTAMP,
  total_calories NUMERIC,
  total_protein_g NUMERIC,
  total_carbs_g NUMERIC,
  total_fat_g NUMERIC
)

meal_items (
  id UUID PRIMARY KEY,
  meal_id UUID REFERENCES meals(id),
  detected_label TEXT,
  classified_label TEXT,
  confidence NUMERIC,
  estimated_weight_g NUMERIC,
  calories NUMERIC,
  protein_g NUMERIC,
  carbs_g NUMERIC,
  fat_g NUMERIC,
  user_corrected BOOLEAN DEFAULT FALSE
)

nutrition_cache (
  food_label TEXT PRIMARY KEY,
  source TEXT,
  per_100g_json JSONB,
  cached_at TIMESTAMP
)
```

`meal_items.user_corrected` is new in this revision — see §12.2 for why capturing user corrections matters even in the MVP.

A single Postgres instance with a JSONB column covers the "flexible nutrient detail" need that would otherwise justify adding MongoDB.

---

## 9. Model Training Plan (EfficientNet-B3 — the one trained component)

| Parameter | Value / Approach |
|---|---|
| Backbone | EfficientNet-B3, ImageNet-pretrained |
| Fine-tuning | Freeze ~70% of layers; fine-tune remaining blocks + new head |
| Input resolution | 300×300 |
| Classes | Start with 15–25 well-represented Indian dish classes |
| Optimizer | Adam/AdamW, small learning rate (~1e-4) with decay |
| Loss | Categorical cross-entropy, class-weighted if imbalanced |
| Batch size | 16–32 |
| Epochs | 15–30 with early stopping |
| Split | 70/15/15, stratified |
| Hardware | Free-tier Colab/Kaggle GPU |
| Experiment tracking | A simple markdown/CSV log of runs (`TRAINING_LOG.md`) |
| Versioning | Save each checkpoint with a version tag (e.g., `efficientnet_v1.pt`) and record which version produced which historical meal result — see §12.1 |

**YOLOv8:** pretrained checkpoint used as-is. **MiDaS:** no training, inference only.

---

## 10. API Design (Single FastAPI Service)

### `POST /meals/analyze`
Multipart upload of an image; runs the full pipeline; returns structured nutrition data.

```json
{
  "meal_id": "uuid",
  "items": [
    {
      "classified_label": "paneer_butter_masala",
      "confidence": 0.91,
      "estimated_weight_g": 145,
      "calories": 406,
      "protein_g": 16,
      "carbs_g": 13,
      "fat_g": 33
    }
  ],
  "totals": { "calories": 640, "protein_g": 20.9, "carbs_g": 64, "fat_g": 33.4 }
}
```

**Error responses** (see §12.2 for the reasoning behind each):
| Status | Case |
|---|---|
| 400 | Image missing, wrong format, or exceeds size limit |
| 422 | Detection found zero food items |
| 200 with `low_confidence: true` flag | Classification confidence below threshold — item still returned, flagged for user review |
| 503 | A pipeline stage failed to load/run (model not initialized, etc.) |

### `PATCH /meals/{id}/items/{item_id}`
Lets a user correct a misclassified item (sets `user_corrected = true`, updates label/weight). Small addition, but it's the seed of your future retraining feedback loop.

### `GET /meals/{id}`
Returns a previously analyzed meal.

### `GET /users/{id}/history`
Returns a user's meal history.

### `GET /nutrition/lookup?food={label}`
Direct nutrition lookup (internal use + debugging).

### Auth
JWT-based, minimal — no OAuth providers needed until there's a real user base asking for it.

---

## 11. Technology Stack (MVP)

| Layer | Technology | Why this and not more |
|---|---|---|
| Frontend | React (Vite or Next.js), Tailwind | Web-first — faster to build and demo |
| Backend | Python + FastAPI | One service, API + ML orchestration together |
| Detection | YOLOv8 (Ultralytics), pretrained | No training required |
| Depth/Volume | MiDaS v3, pretrained + custom geometry code | No training required |
| Classification | EfficientNet-B3, transfer learning | The one component worth real training effort |
| Nutrition data | USDA FoodData Central API | Free, unlimited, no expiry |
| Database | PostgreSQL (JSONB for flexible fields) | One database is enough |
| Deployment | Single server/container (Render, Railway, Fly.io, or one VM) | No orchestration needed for one service |

---

## 12. Operational Considerations for the MVP

*(New in this revision — these are the practical details that determine whether the MVP actually works smoothly, not just whether it works in principle.)*

### 12.1 Model Loading & Performance
- **Load YOLOv8, MiDaS, and the EfficientNet-B3 classifier once, at application startup, as module-level singletons** — not inside the request handler. Reloading any of these per-request would make each API call take seconds to tens of seconds longer than necessary and is a common mistake in FastAPI + ML projects.
- Keep the loaded model versions logged (see §9) so you always know which classifier checkpoint produced which historical result — matters once you retrain and want to compare before/after.
- Target response time for `/meals/analyze` on a single-plate image: a few seconds end-to-end on modest hardware (CPU-only is acceptable for a demo; GPU inference is a later optimization, not a blocker).
- Set a maximum upload size (e.g., 8–10MB) and a maximum image dimension, resizing server-side if needed, so a large photo can't stall the pipeline or exhaust memory.

### 12.2 Error Handling & Edge Cases
Concrete cases the MVP needs to handle gracefully, not just on the happy path:
- **No food detected** — return a clear message rather than a crash or an empty/confusing result.
- **Low classifier confidence** — return the best guess but flag it (`low_confidence: true`) so the UI can visibly ask the user to confirm or correct it, rather than silently presenting a possibly-wrong dish as fact.
- **Unsupported dish** (not in the trained class list) — the classifier will still output *something*; treat anything below a confidence threshold as "unrecognized" rather than a confident wrong answer.
- **Corrupted or non-image files** — validate file type/content before running the pipeline.
- **Partial pipeline failure** (e.g., depth model fails but detection succeeds) — decide in advance whether to fail the whole request or degrade gracefully (e.g., return items with an "estimated" flag on weight using a fallback average portion size).
- **User corrections** — the `PATCH /meals/{id}/items/{item_id}` endpoint (see §10) exists specifically so wrong classifications become logged data instead of silently-accepted errors, which is valuable both for user trust and for future retraining.

### 12.3 Basic Privacy & Data Handling
Even without formal compliance work (see §15), a few plain, easy-to-implement practices matter from day one:
- Store uploaded meal images with access scoped to the uploading user only.
- State plainly in the product (even a one-line notice) what happens to a photo after analysis — e.g., "stored to show your history; you can delete any meal and its image at any time."
- Implement a real delete: removing a meal record should also remove its stored image, not just hide it from the UI.
- Don't collect more than needed — no reason to request location, contacts, or other permissions the app doesn't use.

### 12.4 Rough Cost Estimate
Useful to know upfront even at idea stage:
- **Classifier training:** free (Colab/Kaggle free-tier GPU).
- **USDA FoodData Central API:** free, no cost at any realistic MVP volume.
- **Hosting (single small server for FastAPI + Postgres + static frontend):** roughly $5–20/month on a budget host (Render/Railway/Fly.io free or hobby tiers, or a small VM) — negligible until there's real user traffic.
- **Total to get a working, demoable MVP live: effectively $0–20/month.** This is worth stating plainly in any early pitch conversation — it shows capital efficiency at this stage.

---

## 13. UX Flow (MVP Screens)

*(New in this revision — the pipeline was fully specified, the actual user-facing flow wasn't.)*

1. **Home / Upload screen** — a single, obvious call to action: "Take or upload a photo of your meal." No login wall before the first try, if possible — reduces friction for a first-time demo.
2. **Processing state** — a simple loading indicator while the pipeline runs (a few seconds); avoid a blank screen, since silent multi-second waits feel broken.
3. **Results screen** — the uploaded image with bounding boxes/labels overlaid, a per-item list (dish name, estimated weight, calories, macros), and a clear "totals" summary at the top. Any low-confidence item is visually flagged with a lightweight "is this right?" correction affordance (feeds §10's `PATCH` endpoint).
4. **History screen** — a simple reverse-chronological list of past meals with thumbnails and calorie totals; tapping one reopens the results view.
5. **(Optional, if time allows) Simple daily summary** — total calories/macros for the current day, computed from stored meals — the smallest possible version of a "dashboard," useful for demo purposes without building a full analytics feature.

This is intentionally minimal — five screens is enough to demonstrate the full value proposition without turning frontend work into its own multi-week project.

---

## 14. Solo Build Workflow with Claude Code

1. **Keep the repo structure flat and legible** — clearly-named modules (`detection.py`, `depth.py`, `classify.py`, `nutrition.py`, `pipeline.py`, `main.py`) are easier for an agent to reason about across sessions than deep nesting.
2. **Build one working vertical slice first** — a single food item, default plate-diameter assumption, one classifier class, one nutrition lookup, end to end — before adding multi-item support.
3. **Write small tests as you go** (`pytest`), especially around the geometry/volume math and nutrition-scaling math — pure functions, easy to get subtly wrong, and a solo builder has no second reviewer to catch it.
4. **Log training runs** even informally (`TRAINING_LOG.md`) — useful for debugging and for any future technical due-diligence conversation.
5. **Let the agent hold context via docs, not memory** — keep `design.md` and the README current, and feed relevant sections back into Claude Code sessions rather than relying on it remembering prior sessions.

---

## 15. Regulatory & Compliance Note

Doctor-linked diet monitoring and medication-aligned guidance involve handling health-adjacent personal data and can brush against medical-device and health-data regulation (e.g., India's DPDP Act), regardless of team size. Get a real legal/compliance review before building any version of these — this document intentionally does not include a technical design for them.

---

## 16. Scale-Up Path (For When There's a Team and Funding)

Not the current build target — here so the MVP architecture has a deliberate path forward rather than needing a rewrite:
- Split the single FastAPI service into an API layer and a dedicated ML inference service once volume/specialization justifies it.
- Introduce CI/CD (GitHub Actions) once there's more than one contributor.
- Add MLflow once there's more than one model or more than one person training.
- Add a mobile app (React Native) once the web MVP has validated the core experience.
- Revisit doctor-linked/medication-aware features only after legal/compliance review, with a dedicated health-data architecture separate from the core app database.
- Consider a second data store only if a genuine schema-less or high-write-throughput need emerges.

---

## 17. Evaluation Metrics

| Component | Metric | Reference |
|---|---|---|
| Detection | mAP@0.5, recall | Standard object detection baselines |
| Volume estimation | Mean % error vs. ground truth | Nutrition5k reports ~18–21% MAE; reference-object methods report 3.6–12.3% |
| Classification | Top-1 accuracy, per-class F1 | Target ≥85% top-1 on held-out test set |
| End-to-end calorie estimate | MAE (%) | `MEᵢ = (1/nᵢ) Σ |vⱼ − Vⱼ| / Vⱼ` |

---

## 18. MVP Definition of Done

*(New in this revision — with no fixed deadline, a concrete finish line matters more, not less.)*

The MVP is done when all of the following are true:
- [ ] A user can upload a photo through the web app and receive a nutrition breakdown without any manual data entry.
- [ ] The pipeline correctly handles the edge cases in §12.2 (no food detected, low confidence, bad file) without crashing.
- [ ] The classifier hits its target accuracy (§17) on a held-out test set, and that number is written down somewhere real (`TRAINING_LOG.md`).
- [ ] A user can view their meal history and see past results.
- [ ] A user can correct a wrong classification, and that correction is stored.
- [ ] The whole thing runs end-to-end on a single deployed instance (§12.4) that you can hand someone a link to.
- [ ] You can explain, in plain language, exactly which parts are pretrained and which part you trained — this is your actual pitch, and it should be effortless to state clearly.

Anything beyond this list is a good idea for later, not a reason to delay calling the MVP finished.

---

## 19. Comparison with Existing Systems

| Feature / Capability | Dish Detection in Food Platters | DPF-Nutrition | Im2Calories (Google) | NutriNet | FoodAI (NUS) | **Nutri-AI (Proposed)** |
|---|---|---|---|---|---|---|
| Multi-item Food Detection | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Volume/Portion Estimation | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Fine-grained Classification | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| Calorie Estimation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Micro/Macro Nutrient Estimation | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Cultural Adaptation (Indian dataset) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Single-Person, AI-Assisted Buildable | — | — | — | — | — | ✓ (by design) |

---

## 20. Risks, Limitations & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Plate-diameter default is wrong for a given meal | Volume/calorie error | Let the user optionally correct plate size in the UI; state as a known limitation |
| Small custom classifier dataset | Overfitting, weak generalization | Heavy augmentation, transfer learning, start with fewer well-represented classes |
| Reloading models per request (performance mistake) | Unusably slow demo | Load all models once at startup as singletons (§12.1) |
| Silent wrong classifications | Erodes user trust | Confidence threshold + visible flag + user-correction endpoint (§12.2, §10) |
| Solo builder, no code review | Bugs go unnoticed longer | Small test suite around math-heavy modules; keep modules small and readable |
| No fixed timeline | Scope creep, indefinite build | Treat §5.1 MVP scope and §18 Definition of Done as hard boundaries |
| USDA data sparse for some Indian dishes | Missing nutrition data | Indian Food Composition Table fallback |
| Regulatory exposure if health-linked features are added prematurely | Legal/compliance risk | Explicitly out of MVP scope; requires legal review first (§15) |

---

## 21. References
- Google Research — *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food*.
- *DepthCalorieCam: A Mobile Application for Volume-Based Food Calorie Estimation using Depth Cameras*, MADiMa 2019.
- *DPF-Nutrition* — cross-modal fusion of semantic and geometric features for volume estimation.
- *Food Portion Estimation via 3D Object Scaling*, arXiv 2024.
- *Computer vision-based food calorie estimation: dataset, method, and experiment* (ECUSTFD).
- World Health Organization (WHO), 2024 obesity statistics.
- Indian Council of Medical Research (ICMR), 2023 report.

---

## 22. As-Built Notes

Sections 1–21 are the design as specified. This section records where the built
code departs from it, and why. It is appended rather than woven in so the
section numbers the codebase cites (`design.md §7.3`, `design.md §12.2`, and
two dozen others) keep pointing at the same places.

Nothing here is a change of goal. Each entry is a decision made against a
constraint the design could not have known about in advance.

### 22.1 Frontend: vanilla ES modules, not React

§11 specifies React (Vite/Next.js). Built instead as ES modules with
hand-written CSS: `frontend/index.html` plus `frontend/src/`, no build step, no
`node_modules`, no bundler.

The design's own reasoning is what led here. React was chosen as "the fastest
path to a demoable product" — but for a five-screen app the fastest path turned
out to be no toolchain at all. FastAPI serves `frontend/` as static files, so
`uvicorn main:app` is the entire dev environment; there is no second process, no
port to proxy, and no dependency tree to keep current.

What was given up: JSX, and the component ecosystem. What replaced them:

| Concern | React would provide | Built as |
| --- | --- | --- |
| Routing | React Router | `src/main.js` — path table with lazily `import()`ed pages |
| Components | JSX + reconciler | `src/dom.js` — a small hyperscript helper; `src/components/` |
| State | Context / Redux | `src/store.js` — one observable store |
| Styling | Tailwind / CSS-in-JS | `src/styles/` — six stylesheets on a design-token layer |
| Charts | Recharts | `src/charts.js` — inline SVG |

Route table as built: `/`, `/analyzing`, `/review`, `/results`, `/meal/:id`,
`/today`, `/history`, `/method`, `/settings`, `/auth`, and a 404 fallback. That
is §13's five screens plus `/method` (a plain-language explanation of the
pipeline and its assumptions, which §5.3 and §20 both argue for stating openly),
`/settings` (the plate-diameter correction from §7.3), `/auth`, and `/review`
(the confirmation step between the two analysis phases — §22.13).

Revisit if a second developer joins or the screen count roughly doubles. The
component boundaries were kept conventional so the port would be mechanical.

### 22.2 Database: SQLite by default, Postgres by configuration

§8.2 says "PostgreSQL only", and the reasoning — one database, JSONB instead of
Mongo — holds. The deviation is only about the default: `DATABASE_URL` unset
means SQLite at `backend/nutriai.db`, so a clean checkout runs without
installing a database server.

The schema is written once for both. `db.py` declares
`FlexibleJSON = JSONB().with_variant(JSON(), "sqlite")`, so Postgres gets real
JSONB and SQLite gets plain JSON from the same model definitions. Setting
`DATABASE_URL` to a Postgres URL is the whole migration; the driver is already
in `requirements.txt`.

### 22.3 Stages 1–3 have complete classical implementations, not error paths

§12.2 asks for graceful degradation. Implemented more strongly than the word
usually implies: each of the first three stages has a second, fully working
implementation that runs when torch or the weights are absent.

| Stage | Primary | Fallback | Reported as |
| --- | --- | --- | --- |
| 2 — detection | YOLOv8 (Ultralytics) | Plate-aware segmenter on numpy/scipy | `engine: heuristic` |
| 3 — depth | MiDaS v3 | Shape-from-mask elevation proxy (distance transform × shading) | `engine: heuristic` |
| 4 — classification | EfficientNet-B3, hosted or local | Colour/texture signature prior over all 42 classes | `engine: signature` |

So the pipeline produces a complete, honestly-labelled answer on a machine with
no torch installed at all. This mattered more than expected: it is what let
every stage after 1 be built and tested before any weights existed, and it is
what keeps `pip install -r requirements.txt` a little over 230 MB rather than
2.5 GB. The classical segmenter also yields per-item pixel masks, which
plain YOLO boxes do not, and stage 3 integrates volume over those masks
directly.

The engine name is always in the response and always shown in the UI. A
heuristic answer is never presented as a model answer.

### 22.4 Stage 4 can be hosted as a separate service

Not in the design. §6 specifies a single FastAPI service, and for stages 1, 2, 3
and 5 that is exactly what was built.

The classifier is the exception because it is the only component that needs
torch. Bundling it means every deploy of the API ships CUDA-capable wheels and
needs a box that can hold EfficientNet-B3 in memory — for one stage out of five.
`model_api/` is that stage as a standalone container (a Hugging Face Space, a Fly
machine, anything that runs Docker). The backend calls it over HTTP and holds a
512 MB footprint.

This is not the "multi-service backend" §5.2 rules out. There is one service and
one optional model host; there is no service mesh, no message queue, and no
second database. Set `CLASSIFIER_URL` to use it, leave it unset to use a local
checkpoint. Stage 4 resolves in order: hosted → local checkpoint → signature
prior, and `GET /api/health` reports which one answered plus what remains
underneath it.

Two properties are enforced by test rather than by comment
(`backend/tests/test_model_api.py`): the hosted service and the in-process path
produce bit-identical preprocessed tensors, and the same checkpoint produces the
same labels through either one. Otherwise which machine happened to serve a
photo could change what the user is told.

### 22.5 Class list: 42 labels, trained on what the data covers

§9 plans 15–25 classes from a curated set of 500–1,000 images. Built with 42
labels in `classify.CLASS_LIST`, each with a full nutrition row in
`nutrition.COMPOSITION` and a signature-prior entry.

The two numbers measure different things. 42 is what the system can *name* and
price. What a given checkpoint is *trained* on is whatever the attached datasets
actually cover — typically 20–30 of the 42, which is the design's range. The
remainder are answered by the signature prior rather than being unknown.

`tools/train_kaggle.py --dry-run` prints the coverage table before any GPU time
is spent, and every run's real coverage is recorded in `TRAINING_LOG.md`. A class
with 12 images does not become a good class by being included, so
`--min-per-class` drops it and says so.

### 22.6 Training runs on Kaggle, not on the dev machine

§9 allows "free-tier Colab/Kaggle". Kaggle is now the assumed path rather than
one option: `tools/train_kaggle.py` assembles an ImageFolder tree from datasets
attached to a notebook, maps their folder names onto our labels through an
explicit alias table, and hands the tree to `classify.train()`.

It refuses to start without an accelerator (exit status 3) unless `--allow-cpu`
is passed. `classify._device()` degrades to CPU, which is right for inference and
wrong here: the likeliest reason CUDA is missing in a notebook is that the
dropdown was never switched on, and the failure mode is not an error but an hour
of quota spent going 20× too slow.

Full procedure in `TRAINING_LOG.md`.

### 22.7 Tests: stdlib unittest, not pytest

§11 lists pytest. Built on `unittest` — one less dependency, and
`python -m unittest discover -s tests -t .` needs nothing installed. The tests
themselves are what §12 asks for: the math-heavy modules carry the weight.

246 tests as of this writing: 92 API, 49 geometry, 37 nutrition math, 36
training, 32 model-API parity. Nothing about the assertions depends on the
runner, so moving to pytest later is a rename of the invocation.

### 22.8 opencv is not a dependency

§11 lists `opencv-python`. Everything the pipeline needs — connected components,
distance transforms, morphology, colour-space conversion — is in numpy, scipy
and Pillow, all three of which were already required. Dropping cv2 removed
roughly 60 MB and one of the more common install failures on fresh machines.

### 22.9 Endpoints are namespaced under `/api`

§10 writes them as `POST /meals/analyze`. Built as `POST /api/meals/analyze`,
and likewise throughout, because the same process serves the frontend at `/` —
without the prefix, `/history` is ambiguous between a page and an endpoint. The
request and response shapes are as specified.

Also added, beyond §10: `POST /api/auth/guest` (try the app without registering,
which §13's flow implies but does not name), `GET /api/nutrition/catalog` (the
label list the correction UI needs), `GET /api/users/{id}/summary` (§13's
optional daily summary), `PATCH /api/meals/{id}/plate` (re-run the arithmetic
after a plate-diameter correction, per §7.3), `DELETE /api/meals/{id}` (§12.3
data handling), and `GET /media/{meal_id}`.

### 22.10 Sample images are generated, not photographed

`frontend/samples/` holds four procedurally generated plates from
`tools/make_samples.py`, not photographs. They exist so the empty state has
something to demonstrate on and so tests have deterministic input; they are
labelled as illustrations in the UI. Replace them with real photos before any
user-facing demo — a generated plate is a fine fixture and a poor advertisement.

### 22.11 Corrections to cross-references in `readme.md`

Three pointers in the original readme resolved to the wrong sections and are
fixed in the repository copy: the comparison table is §19 (not §16), the
scale-up path is §16 (not §14), and the reasoning for gating health-linked
features is §15 (not §13).

### 22.12 How to read a `§13.N` or `§14.N` citation

Every other `§N.M` in this repository names a subsection heading. §13 and §14
have no subsections — they are numbered lists — so citations into them mean the
list item at that position. They appear in the code as:

| Citation | Resolves to |
| --- | --- |
| §13.1 | Home / upload screen, including "no login wall before the first try" — cited by `auth.py` for guest sessions |
| §13.4 | History screen — cited by `main.py` for the history endpoint |
| §13.5 | Optional daily summary — cited by `main.py` for the summary endpoint |
| §14.3 | "Write small tests as you go … especially around the geometry/volume math and nutrition-scaling math" — cited by `readme.md`, `nutrition.py`, `tests/__init__.py` and `test_nutrition_math.py` |

Noted rather than renumbered: §1–21 are the specification as written, and
editing headings there to suit the code would defeat the point of keeping the
two distinguishable.

### 22.13 Analysis is two phases with a review step between them

§7 and §10 describe one call: photo and plate diameter in, finished meal out.
Built as two, with the user in the middle:

```
upload → POST /api/meals/scan → review and edit the item list → plate size
       → POST /api/meals/{draft_id}/analyze → results
```

The scan names what is on the plate and stops. It carries no weights, no
calories and no plate diameter. The user then corrects the list — renaming,
removing, adding, and setting piece counts — and only then supplies the plate
width and pays for the expensive half.

Three separate problems made the single call untenable, and one photo showed all
three at once. A plate of four samosas with a lemon wedge came back as 943 kcal
across three items: `Samosa 260 g`, `Curry or Gravy 45 g` (24% sure) and
`Fish Curry 45 g` (35% sure).

**Phantom items were priced before anyone could deny them.** Both 45 g values
are `WEIGHT_BOUNDS["curry"][0]` — the floor returned by the `fallback-portion`
branch in `depth.py` when a region resolves to no measurable area. The lemon
wedge and the basket weave had no area to integrate, so they were assigned the
minimum portion for their guessed category and added to the bill. The fix is a
rule rather than a threshold change: **a detection must earn its area during the
scan or it does not become an item at all.** `pipeline.scan_image` resolves each
mask and drops the detection when `area_px == 0 or area_cm2 <= MIN_ITEM_AREA_CM2`,
logging the drop and reporting the count as a warning.

That leaves `fallback-portion` unreachable in practice, which is the honest way
to describe it. Nothing calls `estimate_volume` without a region any more —
user-added items get `nominal-portion` instead — and the only remaining route to
a sub-0.5 cm² footprint is an absurdly small plate: the smallest region a real
photo yields is around 12 cm² at 26 cm, so it would take a plate under about
5.5 cm to shrink one that far, and the request schema floors the diameter at 12.
The branch stays because a threshold that can only be crossed by a bug should
still not divide by zero.

**Countable food was back-solved from geometry.** Four samosas were one 260 g
blob, so the app could not say "4 pieces" and the user could not correct the
count — only the total weight, via a slider that discards the measurement. Foods
a person would enumerate now carry grams-per-piece and a nominal footprint in
`nutrition.PIECE_WEIGHTS`, and weight resolves in this order:

| Order | Condition | Weight from | `geometry.method` |
| --- | --- | --- | --- |
| 1 | countable label with a count | `count × grams per piece` | `piece-count` |
| 2 | has a region | `estimate_volume`, unchanged | `midas+geometry` / `shape-prior+geometry` |
| 3 | user-added, no region | `nutrition.nominal_portion_g` | `nominal-portion` |

The count is seeded from `area_cm2 ÷ per-piece footprint`, clamped to 1–12, and
flagged as a guess until the user touches it — at which point it stops being
`weight_estimated`, because a count is a discrete thing a person can see and has
been given the chance to fix, unlike a clamped volume. Explicitly not countable:
every curry, dal, rice, grain, dairy, salad and condiment label, plus
`french_fries` and `kheer`. A heap is not a number of things.

**Plate diameter was asked for before there was anything to scale.** It used to
sit on the upload screen, where it set the scale for a list the user had not
seen. It now sits on the review screen, next to that list.

The split point is narrow, which is why this was possible without touching
stages 1–4. Inside the pipeline `plate_diameter_cm` reaches one property —
`PlateEstimate.diameter_cm`, consumed only by `cm_per_px` / `px_area_cm2` — and
the plate *mask and radius* come from the image alone. So detection and
classification are scale-independent and belong to the scan; only
`estimate_volume` needs the number. MiDaS moves entirely into the deep pass,
which is the largest single saving: a provisional list never runs the depth
model, and work is no longer spent measuring items the user is about to delete.

There is one consumer outside the pipeline, and making the plate width a
first-class input asked at review time is what exposed it as a bug.
`PATCH /api/meals/{id}/plate` lets the user correct the number afterwards, and it
used to do that by scaling the finished weights by the *cube* of the change —
reasoning that a linear scale affects both lateral dimensions and, through the
footprint-driven height prior, the vertical one. But that prior is deliberately
damped (`depth.AREA_DEPTH_EXPONENT ≈ 0.2`, so a wide helping cannot inflate its
own depth) and both it and `WEIGHT_BOUNDS` clip, so weight follows no fixed power
of the diameter at all. Measured on `thali.jpg` from 26 to 34 cm the forward
pipeline came out near d^2.3 while the correction assumed d^3.0: **the same photo
at the same stated plate size read 20% heavier if the size arrived by correction
rather than at review.**

`pipeline.remeasure_for_plate` replaces the exponent with the measurement itself.
Footprint is an area, so it scales with the square of the diameter; and because
`integrate_volume` normalises the elevation field to unit mean, volume is
*exactly* footprint × mean depth — the shape of the mound cancels out. So the
stored footprint and density are the entire input, `mean_depth_cm` and
`clamp_weight` run again on the new footprint, and MiDaS never has to see the
photo a second time. Counted and hand-added items return `None` and keep their
weight, because neither ever passed through pixel area; their *footprint* is
still rescaled, since the results page prints it beside measured items' own and
two scales in one list is how a smaller number comes to describe a larger region.
`test_api.py` now asserts the two routes agree rather than asserting the cube.

The deep pass does **not** re-run detection. Segmenter masks are numpy arrays
that cannot survive a JSON round trip, and re-deriving them would invite index
drift against the user's edits. They persist instead as one uint8 PNG label map
beside the photo (`{draft_id}_regions.png`, pixel `i+1` marking item `i`) — flat
regions compress to a few KB and the round trip is exact. Drafts live in a new
`meal_drafts` table, and anything older than six hours is deleted at startup —
a sweep in the `lifespan` hook, not a scheduler, so a long-running process keeps
stale drafts until it restarts. That is sized for the liability, which is stored
photos rather than table rows.

`POST /api/meals/analyze` is unchanged and still tested. It is now implemented as
`analyze_scanned(scan_image(...))`, and it remains the whole API for a caller
that has no user to ask — which is why `frontend/src/api.js` keeps a method no
screen calls.

