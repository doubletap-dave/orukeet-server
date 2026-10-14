# NOTE: uses nvcr.io (not docker.io) because docker.io auth (Cloudflare-fronted)
# is intermittently unreachable from the author's LAN.
FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Orukeet model version. Bump ALL FIVE together on each model release.
# The image tag follows the model version (image 0.2.0 == Orukeet model v0.2.0).
ARG ORUKEET_MODEL_ID=orukeet-v0.1.0
ARG ORUKEET_HF_REPO=oruk/orukeet
ARG ORUKEET_HF_FILE=orukeet-v0.1.0.nemo
ARG ORUKEET_REVISION=555136b50265a132d4cea0d35560c26fc4f657ab
ARG ORUKEET_SHA256=031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    ORUKEET_MODEL_ID=${ORUKEET_MODEL_ID} \
    ORUKEET_HF_REPO=${ORUKEET_HF_REPO} \
    ORUKEET_HF_FILE=${ORUKEET_HF_FILE} \
    ORUKEET_REVISION=${ORUKEET_REVISION} \
    ORUKEET_SHA256=${ORUKEET_SHA256} \
    MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned per oruk-ai/orukeet evaluation environment.
# librosa/soundfile listed explicitly (NeMo pulls them transitively, but the
# server imports them directly for input normalization).
RUN pip3 install --upgrade pip && \
    pip3 install "nemo_toolkit[asr]==3.0.0" huggingface-hub fastapi "uvicorn[standard]" python-multipart librosa soundfile

# Bundle the model: download at build time, verify SHA256, fail the build on
# mismatch. `docker run` then starts instantly with no HuggingFace dependency.
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('${ORUKEET_HF_REPO}', '${ORUKEET_HF_FILE}', revision='${ORUKEET_REVISION}', local_dir='/models')" && \
    echo "${ORUKEET_SHA256}  /models/${ORUKEET_HF_FILE}" | sha256sum -c -

COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8000
CMD ["python3", "server.py"]
