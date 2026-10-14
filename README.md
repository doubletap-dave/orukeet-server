# orukeet-server

OpenAI-compatible speech-to-text server for [Orukeet](https://huggingface.co/oruk/orukeet) — Oruk AI's NeMo-based Parakeet-TDT model covering 25 languages.

## Why this exists

Orukeet ships as a NeMo `.nemo` checkpoint. Nothing off the shelf serves NeMo models behind an OpenAI-compatible API: [speaches](https://github.com/speaches-ai/speaches) only handles Whisper-family models, Ollama doesn't do STT at all, and no prebuilt container existed. So this is a thin FastAPI wrapper around NeMo 3.0 that speaks the `/v1/audio/transcriptions` endpoint clients already know.

## Quick start

Requires an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
docker run -d \
  --name orukeet \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v orukeet-cache:/root/.cache \
  doubletap-dave/orukeet-server:latest
```

Transcribe something:

```bash
curl -s -F file=@audio.wav -F model=orukeet http://localhost:8000/v1/audio/transcriptions
# {"text":"hello world"}
```

Or with plain-text output:

```bash
curl -s -F file=@audio.wav -F model=orukeet -F response_format=text http://localhost:8000/v1/audio/transcriptions
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status":"ok","model_loaded":true}` |
| `/v1/models` | GET | OpenAI-style model list |
| `/v1/audio/transcriptions` | POST | Transcribe audio (multipart: `file`, `model`, `language`, `response_format`) |

`response_format` accepts `json` (default) or `text`. All uploads are normalized to 16 kHz mono WAV server-side before inference — send whatever format you have.

## OpenWhispr

This was built to serve [OpenWhispr](https://github.com/OpenWhispr/OpenWhispr): Settings → Transcription → Self-hosted server, URL `http://<host>:8000/v1`, model field `orukeet`.

## Requirements

- NVIDIA GPU, ~7 GB VRAM (model is 2.5 GB; NeMo runtime overhead on top)
- CUDA 12.x driver on the host
- `--gpus all` (or `--gpus '"device=0"'` to pin a card)

CPU fallback works but is painfully slow — this image is GPU-first by design.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `ORUKEET_MODEL_ID` | `orukeet-v0.1.0` | Model ID reported by `/v1/models` |
| `ORUKEET_HF_REPO` | `oruk/orukeet` | HuggingFace repo |
| `ORUKEET_HF_FILE` | `orukeet-v0.1.0.nemo` | Checkpoint filename |
| `ORUKEET_REVISION` | `555136b…` | Pinned HF commit |
| `ORUKEET_SHA256` | `031c8dda…` | Pinned SHA256 (verified at build and startup) |
| `MODEL_DIR` | `/models` | Where the checkpoint lives in the image |

The model is **bundled in the image** (downloaded and SHA256-verified at build time), so containers start instantly with no HuggingFace dependency. If `/models/` is empty at startup (e.g. an empty volume mounted over it), the server falls back to downloading.

## Versioning

Image tags follow the Orukeet **model** version:

| Tag | Meaning |
|---|---|
| `0.1.0` | Orukeet model v0.1.0 |
| `0.1.0-1`, `0.1.0-2` | Server-only rebuilds (no model change) |
| `latest` | Newest model version |
| `edge` | Latest `main` branch build (bleeding edge) |

## Building locally

```bash
# Default (v0.1.0, matches server.py defaults)
docker build -t orukeet-server .

# Override the model version at build time
docker build \
  --build-arg ORUKEET_HF_FILE=orukeet-v0.2.0.nemo \
  --build-arg ORUKEET_MODEL_ID=orukeet-v0.2.0 \
  --build-arg ORUKEET_REVISION=<pinned-commit> \
  --build-arg ORUKEET_SHA256=<pinned-sha256> \
  -t orukeet-server:0.2.0 .
```

`linux/amd64` only — CUDA + NeMo means NVIDIA GPUs, which means x86_64.

## License

Apache 2.0. Orukeet itself is Apache 2.0 (Oruk AI).
