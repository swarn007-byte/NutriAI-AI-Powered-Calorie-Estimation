# Training log

Stage 4 of the pipeline — the dish classifier — is the only part of Nutri-AI with
learned weights. Everything else is arithmetic or a lookup table, and arithmetic
does not need a changelog. This file is that changelog, and the runbook for
producing the next entry in it.

Two facts shape the whole procedure:

- **Training happens on Kaggle's GPU, not on this machine.** EfficientNet-B3 at
  300 px is roughly 20× slower on CPU: a 40-minute run becomes most of a day.
  `tools/train_kaggle.py` refuses to start without an accelerator rather than
  quietly spending an hour of quota going 20× too slow.
- **Serving happens somewhere else again.** The weights are pushed to the Hugging
  Face Hub and served by `model_api/`, a small FastAPI app on a free Space. The
  backend calls it over HTTP and never imports torch. That is why the backend
  runs on a 512 MB instance.

So a full cycle is three separate places: Kaggle trains, the Hub stores,
a Space serves, and the backend points at the Space.

```
Kaggle notebook (GPU)          Hugging Face Hub          HF Space (docker)         backend
  tools/train_kaggle.py   ──▶   model repo                model_api/app.py    ◀──   CLASSIFIER_URL
  writes efficientnet_v1.pt     efficientnet_v1.pt        loads it at startup       + CLASSIFIER_TOKEN
```

---

## 1. Assemble the data

There is no bundled dataset. `tools/train_kaggle.py` scans whatever is attached
to the notebook and maps the folder names it finds onto our 42 class labels,
because public datasets do not use our names — Food-101 says
`spaghetti_bolognese` where we say `pasta_red_sauce`, the Indian sets say
`chana_masala` where we say `chole_masala`.

### Datasets worth attaching

Search these by name in Kaggle's **Add Input** panel. The script does not depend
on slugs or on any particular one being present, so substitutions are fine — what
matters is only that the folder names resolve, and step 2 tells you whether they
did.

| Dataset | Roughly what it contributes |
| --- | --- |
| **Food 101** | Strict: `samosa`, `pizza_slice` ← `pizza`, `french_fries`, `chicken_curry`, `pasta_red_sauce` ← `spaghetti_bolognese`. Under `--loose` it also reaches `green_salad` ← `caesar_salad`/`greek_salad` and `grilled_chicken` ← `chicken_wings`. Five of its 101 folders earn their keep; see the note below on the other 96. |
| **Indian Food Classification** (20 classes, `Food Classification/train|test/`) | Strict: `samosa`, `idli`, `dosa` ← `masala_dosa`, `naan` ← `butter_naan`, `roti_chapati` ← `chapati`, `pav_bhaji`, `pizza_slice`. Under `--loose`, `dal_tadka` ← `dal_makhani` and `paneer_butter_masala` ← `kadai_paneer`. |
| **Indian Food Images Dataset** (80 classes) | The best of the four, and the only one that is mostly strict: `aloo_gobi`, `bhindi_masala`, `butter_chicken`, `dal_tadka`, `palak_paneer`, `paneer_butter_masala`, `poha`, `naan`, `gulab_jamun`, `kheer` ← `chak_hao_kheer`/`phirni`, `fish_curry` ← `maach_jhol`, `roti_chapati` ← `chapati`. `--loose` adds `mixed_veg_curry`, `poori`, `curd_yogurt` ← `misti_doi`, `grilled_chicken` ← `chicken_tikka`. |
| **Fruits 360** | `banana` strict, `apple` only under `--loose` — and read the warning below first. |

Those routes are read straight out of `resolve_label()`, so the *mapping* half is
exact. Which folder names each dataset actually ships is not verified here — this
machine has no Kaggle access. Step 2's `--dry-run` is the ground truth and will
correct any row above.

### Download the verified public sources directly

For a repeatable local or Colab build, `tools/download_hf_dataset.py` pulls the
public parquet shards from Hugging Face, decodes the images, deduplicates by
SHA-256, applies the same label map, and writes `huggingface_manifest.json` with
licenses and rejected-label reasons. It defaults to the CC BY 4.0 Enhanced Indian
Food Classification set plus the CC0 Indian Foods set. Only the publisher's
`train` split is downloaded by default; request `--split validation` or
`--split test` separately when you need a held-out evaluation tree.

```bash
python backend/tools/download_hf_dataset.py --clean
python backend/tools/train_kaggle.py \
    --input data/raw/huggingface --out data/processed --dry-run
```

Use `--loose` only after reviewing the manifest. It enables approximate aliases
such as generic `dal` → `dal_tadka`; mixed plates such as `biryani` and
`cholebhature` remain excluded even in loose mode.

`Fruits 360` is per-cultivar (`Apple Braeburn`, `Banana Lady Finger`) on a white
studio background. `--loose` collapses the cultivars, but a model trained on
those images learns the background as much as the fruit. Attach it only if the
alternative is having no fruit class at all.

Food-101's remaining 96 folders are not an oversight waiting to be fixed. Two are
deliberately in `EXCLUDED` (`fried_rice`, `spaghetti_carbonara`), and the other 94
are dishes this app has no row for — `baklava`, `foie_gras`, `poutine`. Adding
aliases for them would mean inventing nutrition data, which is the one thing the
composition table exists to prevent.

### Upload the backend as a dataset

The notebook needs this repository's `backend/` directory. Kaggle has no git
checkout, so it goes in as a dataset:

1. Zip `backend/` locally — source only. Exclude `models/`, `__pycache__/`,
   `nutriai.db`, and `data/`; none of it is needed and the zip should be well
   under 10 MB.
2. Kaggle → **Datasets** → **New Dataset** → upload the zip → title it
   `nutriai-backend`.
3. In the notebook, **Add Input** → your `nutriai-backend` dataset.

Re-upload it (new version) whenever `classify.py` or `train_kaggle.py` changes.
Nothing else in the notebook needs to change.

---

## 2. Dry-run first, and read the coverage table

```python
!pip install -q timm
!cp -r /kaggle/input/nutriai-backend/backend /kaggle/working/backend
%env MODEL_DIR=/kaggle/working/models

!python /kaggle/working/backend/tools/train_kaggle.py --dry-run
```

`--dry-run` scans, maps, balances and reports — without touching the GPU. It
costs about a minute and it is the single most useful minute in this procedure,
because the coverage table is what tells you whether the run is worth starting.

It prints four groups, and each one is a decision:

| Heading in the output | What to do about it |
| --- | --- |
| `Trainable classes: N of 42` + a bar chart of counts | This is the class list the checkpoint will actually have. Fewer than ~15 means attach more data. |
| `Below --min-per-class (40), excluded from the head` | Found, but too thin to learn. Either find more data for them or accept that the signature prior handles them. |
| `No data at all — the signature prior keeps handling these` | In `CLASS_LIST`, no images anywhere. Same choice. |
| `Source folders that went nowhere (top 25)` — each with a reason | Read these. A reason of `no mapping` on a folder with 800 images means a missing alias: a one-line addition to `train_kaggle.ALIASES` worth more than another epoch. |

The bar chart is worth a glance on its own. A class with 300 images next to one
with 41 will train unevenly no matter what the loss weights do.

Some misses are deliberate and will stay misses. `biryani` is excluded because a
folder name cannot say whether it is `veg_biryani` or `chicken_biryani`;
`chole_bhature` and `daal_baati_churma` are excluded because they are two or
three dishes on one plate and a single label would teach the model both as one.
Those read as `excluded: <reason>` and are not bugs.

Expect roughly **20–30 trainable classes out of 42** strictly, a few more with
`--loose`. That was the deliberate choice: train what the data covers, record the
coverage here, and let the signature prior answer for the rest. A class with 12
images does not become a good class by being included.

The dry run also writes `coverage.json` next to the output tree. A real run writes
it too, plus `$MODEL_DIR/last_run.json` — the training summary with the coverage
dict attached, which is the full record of what a checkpoint was trained on. The
entries at the bottom of this file are the readable digest of that; `last_run.json`
is the detail. Keep it next to the checkpoint you downloaded.

---

## 3. Train

Notebook → **Settings** → **Accelerator** → **GPU T4 ×2** (or P100) → **Session →
Restart**. Then:

```python
!python /kaggle/working/backend/tools/train_kaggle.py \
    --epochs 24 --batch-size 24 --max-per-class 300 --version v2 \
    --init-from /kaggle/input/nutriai-v1-checkpoint/efficientnet_v1.pt
```

The script prints the GPU name and VRAM before it starts. If it instead prints
`No CUDA device found` and exits with status 3, the accelerator dropdown is off —
that check exists precisely so this failure costs two seconds instead of an hour.

Roughly 100 s/epoch on a T4 at 300 images × ~25 classes, so a 24-epoch run with
early stopping lands around 30–45 minutes, inside the free 30 h/week quota.

Flags worth knowing:

| Flag | Default | Why you would change it |
| --- | --- | --- |
| `--max-per-class` | 300 | Cap. Lower it to iterate faster, raise it if the GPU is idle waiting on data. |
| `--min-per-class` | 40 | Floor. Lowering it buys classes and costs accuracy on them. |
| `--loose` | off | Turns on approximate aliases. Each one trades label accuracy for coverage — `dal_makhani` mapped to `dal_tadka` will read low on calories, because cream and butter are real. Deliberate is fine; accidental is not. |
| `--patience` | 8 | Epochs without a better val top-1 before stopping. |
| `--workers` | 4 | Image-decoding threads. Raise if the GPU is starved. |
| `--version` | auto | Checkpoint tag. Auto-increments from the files in `MODEL_DIR`. |
| `--allow-cpu` | off | Escape hatch for the accelerator check. Only for a smoke test. |
| `CLASSIFIER_TTA_PASSES` | 2 | Inference-time original + horizontal flip ensemble. Set to 1 for the lowest latency. |

`backend/classify.py --train` accepts the same flags and skips the dataset
assembly, for when the ImageFolder tree already exists.

When the run finishes, `/kaggle/working/models/` holds `efficientnet_v1.pt` and
`last_run.json`. Download both. **Do this before the session expires** — Kaggle
discards `/kaggle/working` when the session ends, and an undownloaded checkpoint
is an hour of quota spent on nothing.

The target is **test top-1 ≥ 85%**. Below that, more data beats more epochs
nearly every time; the coverage table usually says where.

---

## 4. Publish the weights

To the Hub as a **model** repo, not a dataset — Spaces and `hf_hub_download`
expect a model repo, and the wrong type fails at load with a 404 that reads like
a missing file.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli repo create nutri-ai-classifier --type model
huggingface-cli upload <your-username>/nutri-ai-classifier \
    efficientnet_v1.pt efficientnet_v1.pt
```

Keep the checkpoint out of git. It is ~50 MB, it changes with every run, and
`model_api/` is designed to fetch it at startup instead.

---

## 5. Serve it

`model_api/` is a self-contained Space: `app.py`, `requirements.txt`,
`Dockerfile`, `README.md` with the Space frontmatter already in it.

1. Hugging Face → **New Space** → SDK **Docker** → name it `nutri-ai-classifier`.
2. Push the four files from `model_api/` to it (the README's frontmatter sets
   `app_port: 7860`).
3. Space → **Settings** → **Variables and secrets**:

   | Name | Kind | Value |
   | --- | --- | --- |
   | `HF_REPO_ID` | variable | `<your-username>/nutri-ai-classifier` |
   | `HF_FILENAME` | variable | `efficientnet_v1.pt` |
   | `API_TOKEN` | **secret** | a long random string you generate |
   | `HF_TOKEN` | secret | only if the weights repo is private |

`API_TOKEN` is not optional in practice. Inference costs CPU, and an open
endpoint is someone else's free compute. The service logs which mode it started
in, so the choice is never accidental.

Confirm it with `GET /health`. `ready: true` plus a `classes` array means the
weights loaded; `ready: false` with an `error` means they did not, and the error
says which of the two sources it tried.

The first request after the Space sleeps takes 30–60 s to cold-start. The backend
is built for that — see below.

---

## 6. Point the backend at it

`backend/.env`:

```ini
CLASSIFIER_URL=https://<your-username>-nutri-ai-classifier.hf.space
CLASSIFIER_TOKEN=<the same API_TOKEN>
```

Restart the backend and check `GET /api/health`. The `models.classification`
block reports the engine that is actually answering, its version and its class
count, plus a `fallback` list of what would answer if it stopped.

Stage 4 tries three engines, most capable first:

1. **`efficientnet_b3@remote`** — the Space. One HTTP request per photo, not per
   item: all crops from one plate go in a single batched request, pre-resized to
   300×300 so each is ~20 KB instead of ~500 KB. After three consecutive
   failures a 120 s circuit breaker opens, so a dead host costs one timeout per
   photo instead of six.
2. **`efficientnet_b3`** — a local checkpoint at `CLASSIFIER_CHECKPOINT`, if this
   deploy has torch and a file. Same weights, same preprocessing arithmetic:
   `backend/tests/test_model_api.py` asserts the two produce bit-identical
   tensors and identical labels, so which machine served a photo cannot change
   what the user is told.
3. **`signature`** — the colour/texture prior. Always available, never wrong in a
   way that takes the API down.

A sleeping Space is the normal case, not an error case: the startup probe fails,
the version and class list are not known yet, and the first successful request
learns them. `/api/health` reads through to whichever engine is live rather than
reporting a snapshot from process start.

To use a local checkpoint instead of the Space, leave `CLASSIFIER_URL` unset and
put the file at `backend/models/efficientnet_v1.pt` with
`ENABLE_TORCH_MODELS=true`.

---

## 7. What a good run looks like

Before calling a checkpoint done:

- [ ] Test top-1 ≥ 85%, and val and test within a few points of each other. A
      large gap means the split leaked or a class is memorised.
- [ ] No trainable class with F1 below ~0.5. The entries below list the five
      weakest precisely so this is checkable.
- [ ] The coverage table has no `no mapping` folder with hundreds of images —
      that is a free accuracy win left on the table.
- [ ] `GET /health` on the Space returns the class list you expect.
- [ ] A real photo through `POST /api/analyze` comes back with
      `engine: efficientnet_b3@remote` and plausible labels.

---

# 2026-09-05 source audit

`backend/tools/download_hf_dataset.py` was run against the publisher train splits
of `SohlHealth/enhanced-indian-food-classification` (CC BY 4.0) and
`bharat-raghunathan/indian-foods-dataset` (CC0). The merged, SHA-256-deduplicated
tree contains **6,981 images across 20 app labels**. The largest new coverage
gains are `coconut_chutney` (200), `plain_rice` (400), `rajma_masala` (200), and
`dosa` (544). Ambiguous mixed plates and approximate aliases were rejected in
strict mode; the full rejection list is in `data/raw/huggingface_manifest.json`.

No checkpoint was replaced from this audit. Retraining still requires a GPU; use
the dry-run output above, then warm-start from `efficientnet_v1.pt` and keep the
new checkpoint only if held-out accuracy and weak-class F1 improve.

# Run history

Entries below are appended automatically by `classify._append_training_log()` at
the end of every `train()` call — including runs that ended early or scored
badly, because a log that only records the good runs is not a log. Newest last.

<!-- Each entry: "## <version> — <UTC timestamp>", then checkpoint, dataset size,
     epochs run, val/test top-1, the five weakest per-class F1 scores, and a
     collapsed JSON epoch history. Do not add hand-written sections after this
     heading; the appender writes to the end of the file. -->

## v1 — 2026-09-03 21:44 UTC
- Checkpoint: `efficientnet_v1.pt` (installed at `backend/models/efficientnet_v1.pt`)
- Dataset: 2,500 images across 24 classes (Google Colab, Tesla T4)
- Epochs run: 22 (early stopping)
- **Val top-1: 77.81%**
- **Test top-1: 78.46%** (target >=85%, design.md section 17)
- Class list, per-class F1, epoch history and dataset coverage:
  `backend/models/last_run.json`
- Note: this run is below the 85% target; use it for integration testing, not as a final accuracy claim.
- Ran on Colab, not Kaggle — Kaggle would not grant the accelerator without
  identity verification. Everything else about the procedure above still held:
  same `train_kaggle.py`, same datasets, same `--dry-run` gate.
- Transcribed by hand rather than appended by `_append_training_log()`, because
  the run wrote to that container's copy of this file. Every number here is read
  back from the run's own `last_run.json`, not from the training console.
- Weakest five per-class F1: `grilled_chicken` 0.333, `mixed_veg_curry` 0.560,
  `chole_masala` 0.571, `fish_curry` 0.571, `dal_tadka` 0.588.
  Strongest three: `french_fries` 1.000, `gulab_jamun` 0.933, `poori` 0.933.
- Reading of that spread: the failures are concentrated in the brown-gravy
  cluster, where four labels differ by ingredient rather than by appearance, and
  in `grilled_chicken`, whose source folder is the least like our other data. The
  classes that separate cleanly are the ones with distinctive shape or colour.
  So the gap to 85% is unlikely to close with more epochs — it wants either more
  images for the gravy labels or an acceptance that some of them are one class.

## mps_smoke — 2026-09-04 22:55 UTC

- Checkpoint: `efficientnet_mps_smoke.pt`
- Dataset: 5225 images across 20 classes
- Epochs run: 1
- **Val top-1: 0.00%**
- **Test top-1: 0.00%** (target ≥85%, design.md §17)
- Weakest per-class F1: aloo_gobi 0.00, bhindi_masala 0.00, chicken_curry 0.00, chole_masala 0.00, coconut_chutney 0.00

<details><summary>Epoch history</summary>

```json
[
  {
    "epoch": 1,
    "train_loss": 0.0,
    "val_top1": 0.0,
    "lr": 0.0,
    "ema_val_top1": 0.0
  }
]
```
</details>

## mps_smoke — 2026-09-04 23:07 UTC

- Checkpoint: `efficientnet_mps_smoke.pt`
- Dataset: 5225 images across 20 classes
- Epochs run: 1
- **Val top-1: 88.89%**
- **Test top-1: 85.86%** (target ≥85%, design.md §17)
- Candidate only: this one-epoch MPS run is not the production checkpoint. It
  adds four previously untrained labels, but real plated-sample smoke checks
  still overpredicted `idli`; use a longer GPU run and crop-level evaluation
  before promotion. `backend/models/last_run.json` records this status.
- Weakest per-class F1: chicken_curry 0.55, roti_chapati 0.67, fish_curry 0.71, rajma_masala 0.75, chole_masala 0.79

<details><summary>Epoch history</summary>

```json
[
  {
    "epoch": 1,
    "train_loss": 1.5349,
    "val_top1": 0.8889,
    "lr": 0.0,
    "ema_val_top1": 0.6769
  }
]
```
</details>
