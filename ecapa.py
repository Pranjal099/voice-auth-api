import os
import psutil
import resource
def log_memory(stage):
    process = psutil.Process(os.getpid())
    print(
        f"[{stage}] RAM = "
        f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
    )
import os
import time
import gc
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

        log_memory("Before SpeechBrain")

        print("Loading Fine-tuned ECAPA...")

        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(BASE_DIR, "pretrained_models")
        )

        log_memory("After SpeechBrain")

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu"
        )

        log_memory("After torch.load")

        verification.mods.embedding_model.load_state_dict(
            checkpoint,
            strict=True
        )

        del checkpoint
        gc.collect()

        log_memory("After checkpoint cleanup")

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

    with torch.inference_mode():
        score, _ = verification.verify_batch(
            owner_audio,
            test_audio
        )

    score = float(score.squeeze())

    latency = round(time.time() - start, 3)

    # Free temporary tensors
    del owner_audio
    del test_audio
    gc.collect()

    return {
        "success": True,
        "verified": score >= threshold,
        "similarity": round(score, 4),
        "threshold": threshold,
        "latency": latency
    }

def log_memory(stage):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # macOS reports bytes, Linux reports KB
    mem_mb = mem / (1024 * 1024)
    print(f"[{stage}] Peak RAM: {mem_mb:.2f} MB")