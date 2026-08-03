"""Text embedding extractor using multilingual-e5-small."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"


class TextExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        clean = (text or "").strip()
        if not clean:
            dim = int(self.model.config.hidden_size)
            vec = np.zeros((dim,), dtype=np.float32)
            return vec, {
                "model": self.model_name,
                "status": "empty_text",
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        encoded = self.tokenizer(
            f"passage: {clean}",
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = (summed / counts).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return emb, {
            "model": self.model_name,
            "status": "ok",
            "text_len": len(clean),
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }
