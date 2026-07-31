# EA-Quad-Overlay

**L4 多模态情感分析数据集处理工作空间**

该仓库同时承担两个角色：
1. **数据处理代码** — 从 CH-SIMS、MELD、IEMOCAP、MOSEI、MUStARD 等原始数据集，经过 ingest → align → extract → annotate → review → package 的流水线，产出标准化的 EA-Quad-Overlay L4 数据集。
2. **数据工作空间** — 存放所有中间产物（特征、标注、清单、报告），按 [`docs/file_structure.md`](docs/file_structure.md) 的目录规则组织。

## 快速开始

```bash
# 安装基础依赖
pip install -e .

# 按需安装 ML 依赖（按模态或全套）
pip install -e ".[text]"          # 仅文本
pip install -e ".[audio]"         # 仅语音
pip install -e ".[video]"         # 仅视频
pip install -e ".[ml]"            # 全模态
pip install -e ".[dev]"           # 含开发工具
```

## Pipeline 概览

详见 [`docs/pipeline.md`](docs/pipeline.md)。

```
Raw Datasets  ──▶  Ingest  ──▶  Align  ──▶  Extract  ──▶  Annotate  ──▶  Review  ──▶  Package  ──▶  L4 Gold
(CH-SIMS, MELD,          (assign EA IDs,      (multi-modal     (text/speech/    (micro-event       (human review,    (train/val/test
 IEMOCAP, MOSEI,          source index)        sync)             macro/micro)      candidates)        adjudication)     manifests)
 MUStARD)
```

## 核心标识

| 层级 | 格式 | 示例 |
|------|------|------|
| Sample | `EAQ` + 6位数字 | `EAQ000001` |
| Segment | Sample ID + `_seg` + 3位数字 | `EAQ000001_seg001` |
| Micro-event | Segment ID + `_micro` + 3位数字 | `EAQ000001_seg001_micro001` |

## 目录结构

```
├── configs/          # 数据集配置（路径、参数）
├── source_index/     # 原始数据索引表
├── docs/             # 文档
├── features/         # 各模态特征
│   ├── text/         # 文本特征 / 嵌入
│   ├── speech/       # 语音 / 声学特征
│   ├── macro/        # 宏观表情 / 人脸特征
│   └── micro/        # 微表情事件特征
├── annotations/      # 标注文件
│   ├── micro_review/ # 人工 review
│   └── l4_gold/      # L4 gold 标注
├── manifests/        # 数据集清单 / split
├── reports/          # QA 报告、实验总结
└── scripts/          # Pipeline 处理脚本
```

## 配置

所有数据集路径和参数集中在 `configs/dataset_defaults.yaml` 管理。
详见 [`configs/README.md`](configs/README.md)。

## 数据溯源

每个 EA Sample 必须在 `source_index/` 中有一条索引记录，
`source_id` 字段保留原始数据集标识，可追溯到 CH-SIMS、MELD、IEMOCAP、MOSEI 或 MUStARD。
参考模板: [`source_index/source_index_template.csv`](source_index/source_index_template.csv)。
