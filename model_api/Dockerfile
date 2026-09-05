# Nutri-AI dish classifier — inference image.
#
# Targets a Hugging Face Space with `sdk: docker`, which runs the container as
# UID 1000 and expects the app on the port named by `app_port` in README.md.
# Nothing here is Space-specific though: `docker run -p 7860:7860` works the
# same, which is the point of keeping this directory free of backend imports.

FROM python:3.11-slim

# Pillow's wheels are self-contained, so no build toolchain is needed. curl is
# here only so the HEALTHCHECK below can run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as a non-root user with $HOME=/home/user. Matching that locally
# keeps the huggingface_hub cache in the same place in both environments, so a
# path problem cannot show up only after deploying.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Requirements first: this layer is cached across code edits, and it is the
# expensive one — torch alone is most of the image.
COPY --chown=user requirements.txt ./
# The CPU index is what keeps the image near 1 GB instead of past 4 GB. A Space
# on the free tier has no GPU, so the CUDA wheels would be dead weight that
# still has to be pulled on every rebuild.
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=user app.py ./

USER user

# Read by app.py; also what `app_port` in the Space README must agree with.
ENV PORT=7860
EXPOSE 7860

# /health returns 200 with status "degraded" when the checkpoint is missing, so
# this probe answers "is the process serving?" and deliberately not "does it
# have weights?" — a Space that is up but unweighted should stay up long enough
# to say so, and the backend already falls back on a 503 from /classify.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# One worker on purpose: each would hold its own copy of EfficientNet-B3, and
# the forward pass is already threaded (TORCH_THREADS). Scale by adding
# replicas, not workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
