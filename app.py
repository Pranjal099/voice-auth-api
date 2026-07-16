import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ecapa import verify_speaker
from spoof import detect_spoof

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Voice Authentication API",
    description="Fine-Tuned ECAPA + AASIST API",
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "Voice Authentication API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# -------------------------------------------------
# ECAPA
# -------------------------------------------------
@app.post("/verify-speaker")
async def verify(
    enrollment_audio: UploadFile = File(...),
    test_audio: UploadFile = File(...)
):

    enrollment_path = os.path.join(
        UPLOAD_DIR,
        "enrollment.wav"
    )

    test_path = os.path.join(
        UPLOAD_DIR,
        "test.wav"
    )

    with open(enrollment_path, "wb") as buffer:
        shutil.copyfileobj(
            enrollment_audio.file,
            buffer
        )

    with open(test_path, "wb") as buffer:
        shutil.copyfileobj(
            test_audio.file,
            buffer
        )

    try:

        result = verify_speaker(
            enrollment_path,
            test_path
        )

        return JSONResponse(result)

    finally:

        if os.path.exists(enrollment_path):
            os.remove(enrollment_path)

        if os.path.exists(test_path):
            os.remove(test_path)


# -------------------------------------------------
# AASIST
# -------------------------------------------------
@app.post("/detect-spoof")
async def spoof_detector(
    audio: UploadFile = File(...)
):

    audio_path = os.path.join(
        UPLOAD_DIR,
        "audio.wav"
    )

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(
            audio.file,
            buffer
        )

    try:

        result = detect_spoof(
            audio_path
        )

        return JSONResponse(result)

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)