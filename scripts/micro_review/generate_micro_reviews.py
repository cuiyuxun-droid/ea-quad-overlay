#!/usr/bin/env python
"""Generate M1 micro-expression review annotations (Issue 05).

Review basis:
  - Peak-window face strips and eye/mouth zoom patches rendered from the source
    videos at the L2 micro candidate peak (scripts/micro_review/*.py).
  - Dense face-ROI motion profiles (frame diff + optical flow) around the true
    video peak frame (scripts/micro_review/dump_dense_face_sequence.py).
  - L2 micro candidate scores (z-score of face-ROI frame-diff + optical-flow).
  - Source text and emotion labels from CH-SIMS label.csv / MELD sent_emo.csv.

Verdicts:
  - negative  : peak coincides with speech articulation, clip-onset motion,
                head nods, or a face-detection glitch; no isolated
                micro-expression signature (brief / localized / suppressed).
  - uncertain : elevated motion in expressive (surprise/fear/anger) contexts
                where a micro-leak cannot be ruled out from stills; requires a
                second reviewer.
  - positive  : none confirmed in this M1 seed (acted dialogue clips).

Output: annotations/micro_review/EAQ<ID>_seg001_micro_review.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotations" / "micro_review"

REVIEW_DATE = "2026-08-01"
REVIEWER = "issue-05-reviewer-1"

# ea_id -> (verdict, primary_motion, note)
# Peak timestamps are full-video frame times from the dense face-ROI motion
# analysis, which maps the L2 peak index to the true video frame via the same
# 48-frame sampling the L2 extractor uses.
REVIEWS = {
    "EAQ000001": (
        "negative",
        "speech_articulation",
        "口部动作与说话节奏同步（'我不想嫁给李茶'），峰值 0.48s 无孤立微表情特征；diff 峰值 8.5 且前后对称回落，为语音驱动的口部运动。",
    ),
    "EAQ000002": (
        "negative",
        "clip_onset",
        "峰值 frame 2 (0.08s) 为片段开头启动运动，diff 达 24.6 后单调回落；候选分 2.0 较低，无微表情形态。",
    ),
    "EAQ000003": (
        "negative",
        "clip_onset",
        "峰值 frame 0 (0.00s) 为首帧伪影，整体运动低（diff~2-3），不存在可判定的 onset 窗口。",
    ),
    "EAQ000004": (
        "uncertain",
        "expressive_speech",
        "候选分 4.05（CH-SIMS 组最高）但真实峰值 1.96s 处绝对运动低且递减（pre 2.98/peak 2.76/post 1.84）；'有那么明显吗?'自省问句存在疑似短暂皱眉/嘴部变化，无法从静帧排除微表情，需第二审核人。",
    ),
    "EAQ000005": (
        "negative",
        "speech_articulation",
        "峰值 1.08s 与长句说话口型同步，diff 高位（16-31）持续多个帧，为语音驱动大位移口部/头部运动，无孤立微表情特征。",
    ),
    "EAQ000006": (
        "negative",
        "clip_onset",
        "峰值 frame 6 (0.24s) 为片段开头启动运动（diff 高达 18.9）；整句'我刚才信号不好...'后续口型主导。",
    ),
    "EAQ000007": (
        "negative",
        "speech_articulation",
        "峰值 2.44s 处于长句说话中，diff/flow 全程平稳（~7-8），无明显瞬时峰值，为连续语音口部运动。",
    ),
    "EAQ000008": (
        "negative",
        "clip_onset",
        "峰值 frame 3 (0.12s) 为片段开头启动运动，含一次人脸检测跳变（diff 0.1）；后续说话口型主导。",
    ),
    "EAQ000009": (
        "uncertain",
        "subtle_suppressed",
        "文本'友德读懂了您对我的暗示'含隐含情绪语境，真实峰值 1.48s；diff 升至 14.2 后回落，中低强度、疑似细微抿嘴/嘴角变化，无法从静帧定论，需第二审核人。",
    ),
    "EAQ000010": (
        "negative",
        "speech_articulation",
        "峰值 3.32s 与恳求式说话（'经理经理，别忘了我那五十万'）口部同步，diff 平稳（~10-13）无瞬时峰。",
    ),
    "EAQ000011": (
        "negative",
        "head_nod",
        "峰值 1.00s 处单帧 diff 突增至 21（flow 3.07）后回落，对应'对对对'强调性点头动作，属头部姿态运动而非面部微表情。",
    ),
    "EAQ000012": (
        "negative",
        "speech_articulation",
        "峰值 5.46s 处于 'My duties?  All right.' 尾段说话中，diff 从 1.2 单调爬升至 7.7，为持续口型运动；MELD 表演型表情。",
    ),
    "EAQ000013": (
        "uncertain",
        "expressive_speech",
        "恐惧台词 'No don't I beg of you!'，真实峰值 0.42s 靠前、候选分 3.54，diff 中低（3-7.5）含 flow 3.27 初始强调；说话起始强调 vs 微表情泄漏无法定论，需第二审核人。",
    ),
    "EAQ000014": (
        "uncertain",
        "expressive_speech",
        "惊讶台词 'Really?!'，真实峰值 2.63s、flow 达 10.7 且持续 5+ 帧，更接近宏表情级别的真实惊讶（AU1+2+5+26）；无法从静帧排除微表情泄漏成分，需第二审核人。",
    ),
    "EAQ000015": (
        "uncertain",
        "expressive_speech",
        "全组最高候选分 4.16，但真实峰值 0.83s 处绝对运动极低且平坦（diff~3.5，flow~1.0），为低方差样本中的相对 z-score 峰；'But then who? The waitress...?'惊讶顿悟语境下无法确认是否存在细微微表情，需第二审核人。",
    ),
    "EAQ000016": (
        "negative",
        "detection_glitch",
        "峰值 1.38s 处为单帧人脸检测跳变伪影（diff 0.6→37.2→1.5，flow 0.1→3.2→0.2），非连续面部运动；'You know? Forget it!'为普通说话。",
    ),
    "EAQ000017": (
        "negative",
        "speech_articulation",
        "峰值 1.25s 与 'Why do all your coffee mugs...' 说话口型同步，diff 缓慢爬升（1.4→6.0）为语音驱动。",
    ),
    "EAQ000018": (
        "negative",
        "speech_articulation",
        "峰值 2.71s 处于怒气台词中，diff 高位（20-22）持续后单调衰减，为表演型愤怒说话/头部运动，无孤立微表情特征。",
    ),
    "EAQ000019": (
        "negative",
        "speech_articulation",
        "峰值 0.83s 与 'Push 'em out, harder!' 高亢说话/用力动作同步，diff 高位（21-22）持续，口部+头部主导。",
    ),
    "EAQ000020": (
        "uncertain",
        "suppressed_face",
        "怒气台词 'There is no more left, left!' 但真实峰值 0.50s 处绝对运动极低且平坦（diff~0.9，flow~0.3），高候选分 4.03 为低方差样本中的相对峰；疑似表情抑制/掩蔽的脸部，无法从静帧确认微表情，需第二审核人。",
    ),
}

# ea_id -> source context: (dataset, source_text, emotion_label)
SOURCE_CONTEXT = {
    "EAQ000001": ("CH-SIMS", "我不想嫁给李茶", "Negative"),
    "EAQ000002": ("CH-SIMS", "你这是嫁入豪门啊！", "Positive"),
    "EAQ000003": ("CH-SIMS", "我不想嫁入什么豪门，我们不就是豪门吗？", "Negative"),
    "EAQ000004": ("CH-SIMS", "有那么明显吗？", "Negative"),
    "EAQ000005": ("CH-SIMS", "我在这消费这么多钱，我低血糖吃块糖还不行了", "Negative"),
    "EAQ000006": ("CH-SIMS", "我刚才信号不好，现在可以了，怎么样，妈，公司给我安排的海景别墅楼。", "Positive"),
    "EAQ000007": ("CH-SIMS", "我刚才是一害怕我就跟王安迪撒了个谎", "Negative"),
    "EAQ000008": ("CH-SIMS", "经理经理，别忘了我那五十万和副科长。", "Positive"),
    "EAQ000009": ("CH-SIMS", "友德读懂了您对我的暗示。", "Positive"),
    "EAQ000010": ("CH-SIMS", "到干妈的公司来做CEO啊，对对对，年薪一个亿", "Positive"),
    "EAQ000011": ("CH-SIMS", "哦，对对对，她身份很特别，不想让我张扬出去。", "Positive"),
    "EAQ000012": ("MELD", "My duties? All right.", "surprise/positive"),
    "EAQ000013": ("MELD", "No don't I beg of you!", "fear/negative"),
    "EAQ000014": ("MELD", "Really?!", "surprise/positive"),
    "EAQ000015": ("MELD", "But then who? The waitress I went out with last month?", "surprise/negative"),
    "EAQ000016": ("MELD", "You know? Forget it!", "sadness/negative"),
    "EAQ000017": ("MELD", "Why do all your coffee mugs have numbers on the bottom?", "surprise/positive"),
    "EAQ000018": ("MELD", "Oh. That's so Monica can keep track. That way if one on them is missing, she can be like, 'Where's number 27?!'", "anger/negative"),
    "EAQ000019": ("MELD", "Push 'em out, push 'em out, harder, harder.", "joy/positive"),
    "EAQ000020": ("MELD", "Okay, y'know what? There is no more left, left!", "anger/negative"),
}


def load_micro_meta(micro_dir: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if micro_dir.is_dir():
        for f in sorted(micro_dir.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            meta[d["ea_id"]] = d
    return meta


def build_event(verdict: str) -> dict | None:
    if verdict != "positive":
        return None
    return {
        "onset_sec": None,
        "apex_sec": None,
        "offset_sec": None,
        "aus": [],
        "intensity": None,
        "confidence": None,
    }


def main() -> None:
    micro_dir = ROOT / ".work" / "meta_micro"
    if len(sys.argv) > 1:
        micro_dir = Path(sys.argv[1])
    micro_meta = load_micro_meta(micro_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for ea_id in sorted(REVIEWS):
        verdict, primary_motion, note = REVIEWS[ea_id]
        dataset, text, emotion = SOURCE_CONTEXT[ea_id]
        meta = micro_meta.get(ea_id, {})
        record = {
            "ea_id": ea_id,
            "segment_id": f"{ea_id}_seg001",
            "review_status": verdict,
            "has_micro_expression": {
                "positive": True,
                "negative": False,
                "uncertain": None,
            }[verdict],
            "reviewer": REVIEWER,
            "review_date": REVIEW_DATE,
            "source_context": {
                "dataset": dataset,
                "text": text,
                "emotion_label": emotion,
                "candidate_score": meta.get("candidate_score"),
                "l2_peak_frame_index": meta.get("peak_frame_index"),
                "frames_with_face": meta.get("frames_with_face"),
            },
            "evidence": {
                "primary_motion": primary_motion,
                "note": note,
            },
            "event": build_event(verdict),
        }
        path = OUT_DIR / f"{ea_id}_seg001_micro_review.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(ea_id)
    print(f"wrote {len(written)} review files to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
