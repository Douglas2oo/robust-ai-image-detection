# Robust AI Generated Image Detection

MAIE5421 / ARIN5201 课程项目。已实现冻结 CLIP ViT-L/14 特征缓存与线性分类头训练；ResNet、数据准备、校准和统一评测仍在开发中。

## 分工与实验组织

- `resnet/`：ResNet 小队维护模型、训练和配置。
- `clip_detector/`：CLIP 小队统一维护基础实验与增强实验，两组使用同一个模型和训练入口，通过配置区分。
- `common/`：各模型共同使用的数据、预处理、指标、校准和测试扰动。
- `scripts/`：数据准备、统一评测和单图推理入口。

第三个 baseline 尚未确定，当前不创建相关目录、文件或占位配置。`clip_detector` 的命名避免与 OpenAI CLIP 的 Python 包名 `clip` 冲突。

## 目录

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── resnet/
│   ├── __init__.py
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   └── config.yaml
├── clip_detector/
│   ├── __init__.py
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   ├── augmentations.py
│   ├── cache_features.py
│   └── configs/
│       ├── baseline.yaml
│       └── augmented.yaml
├── common/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── calibration.py
│   └── corruptions.py
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── evaluate.py
│   └── predict.py
├── manifests/
│   └── README.md
├── data/
├── cache/
├── checkpoints/
└── outputs/
```

## 开发环境

`requirements.txt` 是初始依赖清单，尚未锁定或验证完整训练环境。由团队完成环境试跑后固定版本。OpenAI CLIP 使用其官方 Git 仓库依赖。

可在项目根目录自行建立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

骨架入口只依赖 Python 标准库，未安装训练依赖也可以查看帮助：

```bash
python -m resnet.train --help
python -m clip_detector.train --help
python -m clip_detector.cache_features --help
python -m scripts.prepare_data --help
python -m scripts.evaluate --help
python -m scripts.predict --help
```

CLIP 训练和缓存入口可执行；ResNet、数据准备、公共评测和推理入口仍是骨架。

## 计划中的训练入口

CLIP 两组实验先缓存特征，再训练分类头：

```bash
python -m resnet.train --config resnet/config.yaml
python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml
python -m clip_detector.train --config clip_detector/configs/baseline.yaml
python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml
python -m clip_detector.train --config clip_detector/configs/augmented.yaml
```

ResNet 配置中的 `null` 仍表示尚待确定；CLIP 配置已锁定 ViT-L/14、L2 特征归一化、AdamW 参数与增强缓存设置。CLIP 参数详情见 `clip_detector/README.md`。

所有配置中的相对路径以项目根目录为基准，因此以上命令在项目根目录运行。

## 共同接口与实验约定

1. 固定 `Real=0`、`AI=1`。读取 `manifests/` 中的划分，禁止在各模型训练脚本内重新划分。
2. 输入为 RGB、224 像素图像。基础几何处理口径一致，归一化采用各预训练权重的参数。
3. 模型输出每张图片的单个原始 logit；公共校准模块计算 `P(AI)`，并应用验证集确定的温度与阈值。
4. CLIP 编码器冻结，两组实验共用模型与训练代码，分别训练分类头。主消融仅改变增强相关设置。
5. 特征缓存区分原图和增强视图，并记录数据清单、编码器、预处理、特征归一化及增强配置。缓存不能仅凭实验名称判断有效；提取过程按 chunk 原子保存并支持中断续跑。
6. 仅使用验证集选参和校准，测试前冻结规则和数值。正式测试包括已见、未见、外部数据与共同扰动条件。
7. 建议逐图预测至少保存 `run_id,image_id,label,generator,condition,logit,p_ai,prediction`。汇总表必须可以从预测文件重算。

CLIP 训练每轮原子保存 `last.pt`，并在验证 loss 改善时保存 `best.pt`。`last.pt` 保存优化器、训练历史和随机数状态用于 `--resume`；`best.pt` 供公共评测和推理加载。

## 成果与 Git

运行时按实验与分类头训练种子组织，例如：

```text
checkpoints/clip_baseline/seed_42/
checkpoints/clip_augmented/seed_42/
outputs/resnet/seed_42/
outputs/clip_baseline/seed_42/
outputs/clip_augmented/seed_42/
```

每次运行保存配置快照、日志、逐图预测与指标。使用新的运行标识或显式覆盖策略，避免同一种子的重跑覆盖已有结果。两个 CLIP 实验最终各跑三个固定种子；当前配置中的 42 是首轮种子。

原始数据、特征缓存、模型权重和批量实验输出不提交 Git；配置、源码、文档和数据划分清单提交 Git。运行目录使用 `.gitkeep` 保留结构。需要共享的最终小型图表或汇总表可经人工选择后另行纳入版本管理。

## 实现顺序

先完成数据清单与读取接口，再并行开发 ResNet、CLIP 和公共评测；随后接入 CLIP 增强与缓存，最后联调校准、评测和演示。先用固定小子集验证流程，再锁定数据和配置开展正式实验。
