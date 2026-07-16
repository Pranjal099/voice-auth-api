import os
import time
import torch
import soundfile as sf

from speechbrain.inference.speaker import SpeakerRecognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = os.path.join(
    BASE_DIR,
    "checkpoints",
    "voicepay_best_epoch_40_eer_3.60.pth"
)

# Lazy-loaded model
verification = None


def get_verification():
    global verification

    if verification is None:

        print("Loading Fine-tuned ECAPA...")

        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(BASE_DIR, "pretrained_models")
        )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu"
        )

        verification.mods.embedding_model.load_state_dict(
            checkpoint,
            strict=True
        )

        verification.eval()

        print("ECAPA Loaded Successfully")

    return verification


def verify_speaker(enrollment_path, test_path, threshold=0.75):

    verification = get_verification()

    start = time.time()

    owner_audio, _ = sf.read(enrollment_path)
    test_audio, _ = sf.read(test_path)

    owner_audio = torch.tensor(owner_audio, dtype=torch.float32)
    test_audio = torch.tensor(test_audio, dtype=torch.float32)

    if owner_audio.ndim == 2:
        owner_audio = owner_audio.mean(dim=1)

    if test_audio.ndim == 2:
        test_audio = test_audio.mean(dim=1)

    owner_audio = owner_audio.unsqueeze(0)
    test_audio = test_audio.unsqueeze(0)

    score, _ = verification.verify_batch(
        owner_audio,
        test_audio
    )

    score = float(score.squeeze())

    latency = round(time.time() - start, 3)

    return {
        "success": True,
        "verified": score >= threshold,
        "similarity": round(score, 4),
        "threshold": threshold,
        "latency": latency
    }