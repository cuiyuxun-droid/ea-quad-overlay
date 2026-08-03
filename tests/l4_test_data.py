from __future__ import annotations

from typing import Any


def make_valid_label() -> dict[str, Any]:
    return {
        "schema_version": "m1-l4-gold-v1",
        "ea_id": "EAQ000001",
        "segment_id": "EAQ000001_seg001",
        "source_dataset": "CH-SIMS",
        "modality_va": {
            "text": {"valence": 0.2, "arousal": 0.1, "confidence": 0.9},
            "speech": {"valence": 0.1, "arousal": 0.2, "confidence": 0.8},
            "macro": {"valence": 0.15, "arousal": 0.1, "confidence": 0.8},
            "micro": {"valence": 0.0, "arousal": 0.0, "confidence": 0.0},
        },
        "inter_va": {"valence": 0.152, "arousal": 0.132, "confidence": 0.836},
        "contradiction_type": "consistent",
        "involved_modalities": [],
        "fusion_weights": {
            "text": 0.36,
            "speech": 0.32,
            "macro": 0.32,
            "micro": 0.0,
        },
        "reason": "Text, speech, and macro evidence agree; no confirmed micro cue.",
        "annotation_meta": {
            "method": "evidence_triangulation_single_pass",
            "review_status": "single_pass_pending_second_review",
            "micro_review_status": "negative",
            "evidence": [
                "source_annotation",
                "raw_audio",
                "raw_video",
                "issue_5_micro_review",
            ],
        },
    }
