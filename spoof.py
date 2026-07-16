import os
import time
import torch
import torchaudio
import soundfile as sf

from aasist.models.AASIST import Model
from aasist.data_utils import pad

print("Loading AASIST...")

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

model = Model(CONFIG)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "aasist",
    "models",
    "weights",
    "AASIST.pth"
)

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
model.eval()

device = torch.device("cpu")
model.to(device)

print("AASIST Loaded Successfully")


@torch.no_grad()
def detect_spoof(audio_path):

    start = time.time()

    waveform, sr = sf.read(audio_path)

    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32
    )

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(
            sr,
            16000
        )
        waveform = resampler(
            waveform.unsqueeze(0)
        ).squeeze(0)

    waveform = waveform.numpy()

    waveform = pad(
        waveform,
        64600
    )

    waveform = torch.FloatTensor(
        waveform
    ).unsqueeze(0)

    waveform = waveform.to(device)

    _, logits = model(waveform)

    prediction = torch.argmax(
        logits,
        dim=1
    ).item()

    latency = round(time.time() - start, 3)

    return {
        "success": True,
        "prediction": "Real" if prediction == 0 else "Spoof",
        "spoof": prediction != 0,
        "latency": latency
    }