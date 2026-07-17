import os
import psutil

def log_memory(stage):
    process = psutil.Process(os.getpid())
    print(
        f"[{stage}] RAM = "
        f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
    )
import os
import gc
import time
import torch
import torchaudio
import soundfile as sf

from aasist.models.AASIST import Model
from aasist.data_utils import pad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64]
    ],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

MODEL_PATH = os.path.join(
    BASE_DIR,
    "aasist",
    "models",
    "weights",
    "AASIST.pth"
)

device = torch.device("cpu")

# Lazy loaded model
model = None

# Reuse resampler
_resamplers = {}


def get_resampler(sr):
    if sr not in _resamplers:
        _resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
    return _resamplers[sr]


def get_model():
    global model

    if model is None:

        print("Loading AASIST...")

        model = Model(CONFIG)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]

        model.load_state_dict(checkpoint)

        # Free checkpoint memory
        del checkpoint
        gc.collect()

        model.eval()
        model.to(device)

        print("AASIST Loaded Successfully")
        log_memory("Before AASIST")

        model = Model(CONFIG)

        log_memory("After model")

        checkpoint = torch.load(...)

        log_memory("After checkpoint")

        model.load_state_dict(...)

        del checkpoint
        gc.collect()

        log_memory("After cleanup")
    return model


def detect_spoof(audio_path):

    model = get_model()

    start = time.time()

    waveform, sr = sf.read(audio_path)

    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32
    )

    if sr != 16000:
        waveform = get_resampler(sr)(
            waveform.unsqueeze(0)
        ).squeeze(0)

    waveform = waveform.numpy()

    waveform = pad(
        waveform,
        64600
    )

    waveform = torch.FloatTensor(
        waveform
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        _, logits = model(waveform)

    prediction = torch.argmax(
        logits,
        dim=1
    ).item()

    latency = round(time.time() - start, 3)

    # Free temporary tensors
    del waveform
    del logits
    gc.collect()

    return {
        "success": True,
        "prediction": "Real" if prediction == 0 else "Spoof",
        "spoof": prediction != 0,
        "latency": latency
    }