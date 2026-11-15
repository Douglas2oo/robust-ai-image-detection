# Robust AI Generated Image Detection

MAIE5421 / ARIN5201 课程项目（Group 11，Topic 6）。

当前状态：

| 部分                                                          | 状态                                          |
| ------------------------------------------------------------- | --------------------------------------------- |
| 数据审计、去重、固定划分（`scripts/prepare_data.py`）         | 已完成，清单已冻结                            |
| 公共模块 `common/`（数据、预处理、测试扰动、指标、校准）      | 已实现                                        |
| CLIP 主模型（冻结 ViT-L/14 + 线性头，有/无增强 × 3 个种子）   | 已训练并评测，结果见 `results/clip/`          |
| 统一评测 `scripts/evaluate.py`、单图推理 `scripts/predict.py` | 已实现                                        |
| ResNet-50 基线（`resnet/`）                                   | ⏳ 骨架，待实现；接入方式见 `resnet/README.md` |

## 分工与实验组织

- `resnet/`：ResNet 小队维护模型、训练和配置。
- `clip_detector/`：CLIP 基础实验与增强实验，共用同一个模型和训练入口，通过配置区分。
- `common/`：各模型共同使用的数据读取、预处理、测试扰动、指标和校准。**所有模型必须使用这里的实现**，结果才能放进同一张表。
- `scripts/`：数据准备、统一评测、单图推理、报告图表。
- `slurm/`、`setup/`：HKUST SuperPOD 上的作业脚本和环境配置。

## 目录

```text
.
├── common/
│   ├── data.py            清单读取 + ManifestImageDataset（训练视图 / 测试扰动视图）
│   ├── preprocessing.py   224 预处理（CLIP / ImageNet 归一化，几何处理相同）
│   ├── corruptions.py     8 种测试扰动 + 训练用随机退化
│   ├── metrics.py         binary_metrics（acc、bal_acc、P/R/F1、AUROC、AP、ECE、Brier）
│   └── calibration.py     温度缩放 + 阈值选择（只用验证集）
├── clip_detector/
│   ├── model.py           冻结 CLIP 编码器 + 线性头
│   ├── augmentations.py   训练增强
│   ├── cache_features.py  抽取并缓存特征
│   ├── train.py           训练线性头 + 校准 + 阈值
│   ├── config.py          配置与路径
│   └── configs/{baseline,augmented}.yaml
├── resnet/                ResNet-50 baseline
├── scripts/
│   ├── prepare_data.py    审计、去重、生成清单
│   ├── evaluate.py        所有模型共用的评测
│   ├── predict.py         单图推理演示
│   └── make_report_assets.py   由 summary.csv 生成报告图表
├── manifests/             冻结的数据清单
├── results/clip/          CLIP 的最终汇总结果与图表
├── docs/                  数据说明、CLIP 结果详表
├── slurm/、setup/         SuperPOD 作业脚本与环境
├── tests/                 端到端冒烟测试
└── data/ cache/ checkpoints/ outputs/   运行时目录
```

## 开发环境

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

SuperPOD 上使用 `bash setup/setup_env.sh`（conda 环境 `aigi`，并预先缓存 CLIP 权重）。

## 路径

配置里的路径默认相对于项目根目录（`data/`、`manifests/`、`cache/clip/`、`checkpoints/`、`outputs/`），
每一项都可以用命令行参数覆盖：`--data_root --manifest_dir --cache_dir --checkpoint_dir --output_dir`。

在 SuperPOD 上数据集位于公共目录 `/project/mscaispod2026/group11`，其余产物放各自的个人目录；
`slurm/config.sh` 里定义了这些路径并拼成 `$PATH_ARGS`。也可以用软链接：`ln -s /project/mscaispod2026/group11 data`。

## 运行流程

```bash
# 0. 数据清单
python -m scripts.prepare_data --help

# 1. 抽取并缓存 CLIP 特征
python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml
python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml

# 2. 训练线性头（每个配置 3 个种子；只用验证集选参、校准、定阈值）
python -m clip_detector.train --config clip_detector/configs/baseline.yaml
python -m clip_detector.train --config clip_detector/configs/augmented.yaml

# 3. 统一评测：5 个测试集 × 8 种扰动
python -m scripts.evaluate --runs checkpoints/clip_noaug checkpoints/clip_aug \
    --feat_dir cache/clip/test --out_dir outputs/eval --save_preds

# 4. 报告图表、单图推理
python -m scripts.make_report_assets --summary outputs/eval/summary.csv --out results/clip
python -m scripts.predict --checkpoint checkpoints/clip_aug/seed0/head.pt some_image.jpg
```

SuperPOD 上对应的作业：`sbatch slurm/00_build_splits.sbatch`（CPU）、`bash slurm/submit_all.sh`（依次提交抽特征和训练+评测）。

## 共同接口与实验约定

1. 固定 `Real=0`、`AI=1`。读取 `manifests/` 中的划分，禁止在训练脚本内重新划分。
2. 输入为 RGB；预处理为短边 bicubic 缩放到 224、中心裁剪 224（`common/preprocessing.py`），归一化用各自预训练权重的参数。
3. 训练增强和测试扰动都在原始分辨率上、预处理之前施加；真实图和生成图条件相同。
4. 模型输出每张图的单个 logit。`P(AI) = sigmoid(logit / T)`；温度 `T` 和判定阈值只用验证集确定（`common/calibration.py`），测试前冻结。
5. 测试集（官方 val 分区）不参与任何调参。每个测试集 = 1000 张生成图 + 同一组 1000 张真实图。
6. 测试扰动固定为 `common.corruptions.EVAL_PERTURBS`：`clean, jpeg90, jpeg70, jpeg50, down0.5, down0.25, blur0.5, blur1.0`。
7. 所有模型的指标由 `scripts/evaluate.py` 统一计算。非 CLIP 模型提交逐图预测 CSV（列：`config, seed, test, perturb, label, p_ai, threshold`，详见 `resnet/README.md`）。
8. CLIP 编码器冻结；两组 CLIP 实验只差训练增强。

## 成果与 Git

运行时按实验和种子组织：`checkpoints/<experiment>/seed<k>/head.pt`、`train_log.json`。已存在的种子默认跳过，不会被覆盖。

原始数据、特征缓存、模型权重和批量输出不提交 Git；源码、配置、文档、数据清单以及人工挑选的最终汇总结果（`results/`）提交 Git。

