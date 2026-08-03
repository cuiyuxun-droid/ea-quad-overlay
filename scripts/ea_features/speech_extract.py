"""Speech embedding extractor using wav2vec2-base."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000


class SpeechExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, audio_path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        dim = int(self.model.config.hidden_size)
        if audio_path is None or not Path(audio_path).is_file():
            return np.zeros((dim,), dtype=np.float32), {
                "model": self.model_name,
                "status": "missing_audio",
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        waveform, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
        if waveform.size == 0:
            return np.zeros((dim,), dtype=np.float32), {
                "model": self.model_name,
                "status": "empty_audio",
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        inputs = self.processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return emb, {
            "model": self.model_name,
            "status": "ok",
            "duration_sec": round(float(len(waveform) / TARGET_SR), 4),
            "sample_rate": TARGET_SR,
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }
