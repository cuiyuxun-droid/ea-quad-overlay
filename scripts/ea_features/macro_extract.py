"""Macro expression features via OpenCV face crop + ResNet18 embeddings."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ea_features.face_utils import crop_largest_face, sample_frames


MODEL_NAME = "torchvision/resnet18_imagenet_face_embedding"
FEATURE_DIM = 512


def _shrink(frame_bgr: np.ndarray, max_width: int = 480) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    return cv2.resize(frame_bgr, (max_width, max(1, int(h * scale))))


def _resolve_resnet18_weights() -> Path | None:
    candidates = []
    env = os.environ.get("EA_RESNET18_WEIGHTS")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
            Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
        ]
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


class MacroExtractor:
    def __init__(self, max_frames: int = 4, device: str | None = None) -> None:
        from torchvision import transforms
        from torchvision.models import ResNet18_Weights, resnet18

        self.max_frames = max_frames
        self.device = device or "cpu"
        self.model_name = MODEL_NAME
        local_weights = _resolve_resnet18_weights()
        if local_weights is not None:
            self.model = resnet18(weights=None)
            state = torch.load(str(local_weights), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=False)
            self._weights_source = str(local_weights)
        else:
            # May download from the network; prefer pre-caching on the server.
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self._weights_source = "ResNet18_Weights.DEFAULT"
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def extract(self, video_path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        if video_path is None or not Path(video_path).is_file():
            return np.zeros((FEATURE_DIM,), dtype=np.float32), {
                "model": self.model_name,
                "status": "missing_video",
                "face_detect_rate": 0.0,
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        frames = sample_frames(Path(video_path), self.max_frames)
        if not frames:
            return np.zeros((FEATURE_DIM,), dtype=np.float32), {
                "model": self.model_name,
                "status": "no_frames",
                "face_detect_rate": 0.0,
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        embeddings: list[np.ndarray] = []
        face_hits = 0
        for frame in frames:
            crop = crop_largest_face(_shrink(frame))
            if crop is None:
                continue
            face_hits += 1
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            emb = self.model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            embeddings.append(emb)

        face_rate = face_hits / max(len(frames), 1)
        if not embeddings:
            return np.zeros((FEATURE_DIM,), dtype=np.float32), {
                "model": self.model_name,
                "status": "no_face",
                "face_detect_rate": round(face_rate, 4),
                "frames_sampled": len(frames),
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        vec = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        return vec, {
            "model": self.model_name,
            "status": "ok",
            "face_detect_rate": round(face_rate, 4),
            "frames_sampled": len(frames),
            "frames_with_face": face_hits,
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "weights": self._weights_source,
        }

    def close(self) -> None:
        return None
