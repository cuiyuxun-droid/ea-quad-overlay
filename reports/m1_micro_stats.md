# M1 Micro-Expression Review Stats

GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/5>

## Scope

- 首批 20 条样本（`source_index/m1_sample_20.csv`）：11 条 CH-SIMS，9 条 MELD。
- 逐条基于 L2 micro 候选特征 + 源视频峰值窗口人脸序列与眼部/嘴部放大图进行人工正负确认。
- 审查产物：`annotations/micro_review/EAQ<ID>_seg001_micro_review.json`（20 个文件）。

## 判定标准

| 判定 | 含义 |
| --- | --- |
| `positive` | 峰值窗口存在明确、孤立、抑制性的微表情（时长 ≤0.5s、局部化），可给出 onset/apex/offset 与 AU |
| `negative` | 峰值与说话口型、片段开头运动、点头或人脸检测跳变重合，无微表情特征 |
| `uncertain` | 高情绪语境（惊讶/恐惧/愤怒）下存在疑似微泄漏，但静帧无法定论，需第二审核人裁决 |

## 总览

| 指标 | 值 |
| --- | ---: |
| 样本总数 | 20 |
| `positive` | **0** |
| `negative` | **14** |
| `uncertain` | **6** |
| positive 占比 | 0% |

**结论：首批 20 条样本均为表演型对白/独白镜头，L2 候选峰值大多与语音口部运动或片段启动运动重合，无一确认存在典型微表情。** 6 条在高情绪语境下标记为 `uncertain`，建议第二审核人复核后决定是否晋升为候选正样本。

## 按数据集分布

| 数据集 | 样本数 | positive | negative | uncertain |
| --- | ---: | ---: | ---: | ---: |
| CH-SIMS | 11 | 0 | 9 | 2 |
| MELD | 9 | 0 | 5 | 4 |
| 合计 | 20 | 0 | 14 | 6 |

## 按判定动机分布

| primary_motion | 数量 | 说明 |
| --- | ---: | --- |
| `speech_articulation` | 8 | 峰值与说话口部动作同步 |
| `clip_onset` | 4 | 峰值在片段开头（帧 0-6），为启动运动/首帧伪影 |
| `expressive_speech` | 4 | 高情绪语境（惊讶/恐惧）下的表达性运动，判 uncertain |
| `subtle_suppressed` | 1 | 疑似细微抿嘴/嘴角变化，判 uncertain |
| `suppressed_face` | 1 | 表情抑制/掩蔽的面部，判 uncertain |
| `head_nod` | 1 | 峰值对应'对对对'强调性点头 |
| `detection_glitch` | 1 | 单帧人脸检测跳变伪影 |

## 候选分参考（L2 micro z-score）

| 判定 | n | mean | min | max |
| --- | ---: | ---: | ---: | ---: |
| `negative` | 14 | 2.750 | 1.997 | 3.973 |
| `uncertain` | 6 | 3.597 | 2.333 | 4.159 |

`uncertain` 组平均候选分明显更高，与直觉一致；但部分高分样本（如 EAQ000015、EAQ000020）在低方差片段中的绝对运动量很小，说明候选分对"相对突发"敏感，不能单独作为微表情判据。

## 不确定样本清单（需第二审核人）

| EA ID | 数据集 | 源文本 | 情绪标签 | 候选分 | 峰值时刻 |
| --- | --- | --- | --- | ---: | --- |
| `EAQ000004` | CH-SIMS | 有那么明显吗？ | Negative | 4.052 | 1.96s |
| `EAQ000009` | CH-SIMS | 友德读懂了您对我的暗示。 | Positive | 2.333 | 1.48s |
| `EAQ000013` | MELD | No don't I beg of you! | fear | 3.540 | 0.42s |
| `EAQ000014` | MELD | Really?! | surprise | 3.468 | 2.63s |
| `EAQ000015` | MELD | But then who? The waitress...? | surprise | 4.159 | 0.83s |
| `EAQ000020` | MELD | There is no more left, left! | anger | 4.029 | 0.50s |

> 注：峰值时刻为 L2 峰值索引经 48 帧等间隔采样映射回的全视频帧时间（见 `scripts/micro_review/dump_dense_face_sequence.py` 的运动分析），与 `annotations/micro_review` 中各样本 `source_context` 的 L2 峰值索引对应。

## 复现方式

```bash
# 1. 生成峰值窗口 contact sheet（需本地视频 .work/m1_videos）
python scripts/micro_review/make_micro_contact_sheets.py \
    --index source_index/m1_sample_20.csv \
    --videos-dir .work/m1_videos \
    --micro-dir .work/meta_micro \
    --out-dir .work/contact_sheets

# 2. 生成眼部/嘴部放大图
python scripts/micro_review/make_micro_zoom_patches.py

# 3. 生成标注文件（自 .work/meta_micro 注入候选分与源上下文）
python scripts/micro_review/generate_micro_reviews.py .work/meta_micro

# 4. 校验
python scripts/micro_review/validate_micro_reviews.py
```

期望校验输出：

```text
OK: validated 20 reviews (0 positive, 14 negative, 6 uncertain)
```
