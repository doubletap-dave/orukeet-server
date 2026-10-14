"""Orukeet speech-to-text behind an OpenAI-compatible API.

Serves oruk/orukeet (NeMo Parakeet-TDT, 25 languages) on GPU via NeMo.
Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/audio/transcriptions  (multipart: file, model, language, response_format)
"""
import hashlib
import os
import shutil
import tempfile

import librosa
import soundfile as sf

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

MODEL_ID = os.environ.get("ORUKEET_MODEL_ID", "orukeet-v0.1.0")
HF_REPO = os.environ.get("ORUKEET_HF_REPO", "oruk/orukeet")
HF_FILE = os.environ.get("ORUKEET_HF_FILE", "orukeet-v0.1.0.nemo")
REVISION = os.environ.get(
    "ORUKEET_REVISION", "555136b50265a132d4cea0d35560c26fc4f657ab"
)
EXPECTED_SHA256 = os.environ.get(
    "ORUKEET_SHA256",
    "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
CKPT = os.path.join(MODEL_DIR, HF_FILE)

app = FastAPI(title="orukeet-server")
asr = None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model() -> str:
    from huggingface_hub import hf_hub_download

    if os.path.exists(CKPT) and _sha256(CKPT) == EXPECTED_SHA256:
        print(f"model cache hit: {CKPT}", flush=True)
        return CKPT
    if os.path.exists(CKPT):
        print("cached checkpoint hash mismatch, re-downloading", flush=True)
        os.remove(CKPT)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"downloading {HF_REPO}/{HF_FILE}@{REVISION} ...", flush=True)
    path = hf_hub_download(HF_REPO, HF_FILE, revision=REVISION, local_dir=MODEL_DIR)
    assert _sha256(path) == EXPECTED_SHA256, "downloaded checkpoint hash mismatch"
    print("download verified", flush=True)
    return path


@app.on_event("startup")
def load_model():
    global asr
    from nemo.collections.asr.models import ASRModel

    ckpt = ensure_model()
    print("loading NeMo model ...", flush=True)
    asr = ASRModel.restore_from(ckpt)
    asr.eval()
    try:
        import torch

        if torch.cuda.is_available():
            asr = asr.cuda()
            print(f"model on CUDA: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("CUDA not available, running on CPU", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"cuda move failed ({e}), running on CPU", flush=True)
    print("ready", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": asr is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "oruk"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        raw_path = tmp.name
    wav_path = raw_path + ".norm.wav"
    try:
        # Normalize to 16kHz mono WAV — NeMo is picky about input shape,
        # and clients send varying formats/sample rates/channels.
        y, _ = librosa.load(raw_path, sr=16000, mono=True)
        if y.size == 0:
            return JSONResponse({"text": ""})
        sf.write(wav_path, y, 16000)
        hyps = asr.transcribe([wav_path], return_hypotheses=True)
        text = hyps[0].text if hyps else ""
    finally:
        os.unlink(raw_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
