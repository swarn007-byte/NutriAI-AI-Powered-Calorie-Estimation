---
title: Nutri-AI Dish Classifier
emoji: 🍛
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: EfficientNet-B3 stage-4 classifier for the Nutri-AI pipeline
---

# Nutri-AI dish classifier

Stage 4 of the [Nutri-AI](../README.md) pipeline, running on its own.

Given one image crop per detected food item, this returns the dish label and a
confidence for each — nothing else. Detection, portion estimation and the
nutrition lookup stay in the Nutri-AI backend; this service only answers *what
is on the plate*.

## Why it is separate

The classifier is the only stage that needs torch. Keeping it in the backend
means every API deploy ships ~2.5 GB of wheels and needs a box that can hold
EfficientNet-B3 in memory, for one stage out of five. Split out, the backend
runs on a 512 MB instance and this service scales — or sleeps — on its own.

The backend treats it as optional. If `CLASSIFIER_URL` is unset, or this Space
is asleep, or a request times out, the backend falls back to a local checkpoint
and then to its signature prior. A meal photo always gets an answer; without
this service the answer is just coarser.

## API

### `POST /classify`

`multipart/form-data`:

| Field    | Type              | Notes                                        |
| -------- | ----------------- | -------------------------------------------- |
| `images` | file, repeatable  | One crop per detected item. Max 8, 4 MB each. |
| `top_k`  | int, optional     | Alternatives to return per crop. Default 4.  |

Send every crop from one plate in a single request. The forward pass is batched,
so six crops cost one round trip rather than six.

```bash
curl -X POST https://<user>-<space>.hf.space/classify \
  -H "Authorization: Bearer $API_TOKEN" \
  -F images=@crop0.jpg -F images=@crop1.jpg -F top_k=4
```

```json
{
  "engine": "efficientnet_b3",
  "version": "v1",
  "input_resolution": 300,
  "took_ms": 83.2,
  "results": [
    {
      "label": "dal_tadka",
      "confidence": 0.83,
      "alternatives": [{ "label": "sambhar", "confidence": 0.07 }]
    },
    {
      "label": "naan",
      "confidence": 0.91,
      "alternatives": [{ "label": "roti_chapati", "confidence": 0.05 }]
    }
  ]
}
```

`results` is positionally aligned with the uploaded files.

Errors are chosen so the caller knows whether to retry or to give up:

| Status | Meaning                                                                 |
| ------ | ----------------------------------------------------------------------- |
| `401`  | `API_TOKEN` is set and the bearer token was missing or wrong.            |
| `413`  | Too many images, or one image over the byte limit.                       |
| `422`  | No images, or a file that is not a decodable image.                      |
| `503`  | No checkpoint loaded. Retryable — fall back locally and try again later. |

### `GET /health`

```json
{
  "status": "ok",
  "ready": true,
  "engine": "efficientnet_b3",
  "version": "v1",
  "classes": ["dal_tadka", "naan", "..."],
  "input_resolution": 300,
  "uptime_s": 412.5,
  "error": null
}
```

The shape mirrors the `model` block of the backend's own `/api/health`, so the
two can be read side by side when something looks wrong. `status` is
`"degraded"` with a populated `error` when the process is up but has no weights
— which is a configuration problem, not a crash, and reports itself as such.

Interactive docs are at `/docs`.

## Configuration

All optional. With none of it set the service starts, reports `degraded`, and
tells you which variable is missing.

| Variable                  | Default              | Purpose                                                              |
| ------------------------- | -------------------- | -------------------------------------------------------------------- |
| `CHECKPOINT_PATH`         | —                    | Local weights file. Wins over the Hub when set.                      |
| `HF_REPO_ID`              | —                    | Model repo to pull weights from, e.g. `you/nutriai-classifier`.      |
| `HF_FILENAME`             | `efficientnet_v1.pt` | File within that repo.                                               |
| `HF_TOKEN`                | —                    | Only needed for a private model repo.                                |
| `API_TOKEN`               | — (open)             | Shared secret. **Set this on a public Space.**                       |
| `MAX_IMAGES_PER_REQUEST`  | `8`                  | Above the backend's `max_items_per_plate` of 6, with room to spare.  |
| `MAX_BYTES_PER_IMAGE`     | `4194304`            | Crops are small; this is a bound on abuse, not on real input.        |
| `TORCH_THREADS`           | `2`                  | Free Spaces get 2 vCPU. More threads than cores makes it slower.     |
| `CORS_ORIGINS`            | `*`                  | Only affects browsers. The backend calls this server to server.      |

### `API_TOKEN` is not optional in practice

Unset means anyone can spend your CPU quota. The startup log states which mode
the process is in, so the choice is never accidental:

```
Auth: OPEN (set API_TOKEN to restrict)
```

On a Space, add `API_TOKEN` (and `HF_TOKEN` if the model repo is private) under
**Settings → Variables and secrets** as a *secret*, not a variable.

## Deploying to a Hugging Face Space

1. **Publish the weights.** Create a *model* repo — not a dataset — and upload
   the checkpoint produced by training:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload you/nutriai-classifier \
       backend/models/efficientnet_v1.pt efficientnet_v1.pt --repo-type model
   ```

   Weights belong in a model repo rather than in this Space's git so that
   rebuilding the Space does not re-upload 50 MB, and so a new checkpoint is a
   variable change instead of a redeploy.

2. **Create the Space** with SDK **Docker**, then push this directory to it:

   ```bash
   git clone https://huggingface.co/spaces/you/nutriai-classifier space
   cp model_api/{app.py,requirements.txt,Dockerfile,README.md} space/
   cd space && git add . && git commit -m "Nutri-AI classifier" && git push
   ```

   This README has to be the one at the Space root — the YAML frontmatter above
   is what tells Hugging Face to build the Dockerfile and which port to expose.

3. **Set the variables** under Settings: `HF_REPO_ID=you/nutriai-classifier`,
   and `API_TOKEN` as a secret.

4. **Wait for the first build.** It pulls torch, so expect several minutes. When
   the Space is running, check `/health` and confirm `ready: true` and a
   `classes` list that matches what training reported.

## Pointing the backend at it

In `backend/.env`:

```bash
CLASSIFIER_URL=https://you-nutriai-classifier.hf.space/classify
CLASSIFIER_TOKEN=<the same API_TOKEN>
CLASSIFIER_TIMEOUT_S=25
# Optional: average the original crop and a horizontal flip (default 2).
CLASSIFIER_TTA_PASSES=2
```

The timeout is generous because a free Space sleeps after inactivity and the
first request afterwards pays for the container waking up. Confirm the link
with the backend's own health endpoint:

```bash
curl -s localhost:8000/api/health | python -m json.tool
```

`model.engine` should read `efficientnet_b3@remote`.

## Running locally

```bash
cd model_api
pip install -r requirements.txt
CHECKPOINT_PATH=../backend/models/efficientnet_v1.pt uvicorn app:app --port 7860
```

Or as the container that gets deployed, which is worth doing once before
pushing — a Dockerfile that only ever ran on Hugging Face's builder is a
Dockerfile you cannot debug:

```bash
docker build -t nutriai-classifier model_api
docker run --rm -p 7860:7860 \
    -e HF_REPO_ID=you/nutriai-classifier \
    -e API_TOKEN=local-test \
    nutriai-classifier
```

## Model

EfficientNet-B3 fine-tuned on Indian and Western plated dishes, 300×300 input,
ImageNet normalisation. Trained by `backend/tools/train_kaggle.py` on a Kaggle
GPU; see `TRAINING_LOG.md` in the main repo for the datasets, class coverage and
per-class F1 of each version.

The preprocessing here is a deliberate duplicate of `backend/classify._preprocess`
rather than a shared import, because this directory deploys on its own. A
divergence — a different resample filter, a different channel order — would not
raise; it would shift the input distribution off what the model was trained on
and show up as quietly worse accuracy. `backend/tests/test_model_api.py` asserts
the two are numerically identical, so the duplication is checked rather than
trusted.
