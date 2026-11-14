# CLIP Detector

冻结的 OpenAI CLIP ViT-L/14 图像编码器（768 维投影特征，不做 L2 归一化）+ 线性二分类头，做法同 UniversalFakeDetect

| 文件                     | 职责                                                   |
| ------------------------ | ------------------------------------------------------ |
| `model.py`               | 冻结 CLIP 编码器、线性头、检查点读取                   |
| `augmentations.py`       | 训练增强（翻转 + `common.corruptions.random_degrade`） |
| `cache_features.py`      | 抽取并缓存训练 / 验证 / 测试特征                       |
| `train.py`               | 训练线性头，验证集选参、早停、温度缩放、阈值           |
| `config.py`              | 配置读取、路径解析                                     |
| `configs/baseline.yaml`  | C0 `clip_noaug`：无退化增强                            |
| `configs/augmented.yaml` | C1 `clip_aug`：有退化增强                              |

## 训练增强

|                              | `clip_noaug` | `clip_aug`            |
| ---------------------------- | ------------ | --------------------- |
| 随机水平翻转                 | 50%          | 50%                   |
| 缩小再放大回原尺寸（双线性） | —            | 30%，比例 ∈ [0.25, 1] |
| 高斯模糊                     | —            | 30%，σ ∈ [0.1, 1.5]   |
| JPEG 重新压缩                | —            | 50%，质量 ∈ [50, 100] |
| 每张训练图缓存的视图数       | 4            | 4                     |

三种退化独立抽样、可叠加，顺序固定为 缩放 → 模糊 → JPEG；真实图和生成图使用相同的概率和范围，在原始分辨率上施加。
注意：增强范围覆盖了测试扰动的强度，所以鲁棒性测试属于训练分布内的扰动。

## 流程

```bash
python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml
python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml
python -m clip_detector.train --config clip_detector/configs/baseline.yaml
python -m clip_detector.train --config clip_detector/configs/augmented.yaml
```

**特征缓存**（`<cache_dir>/`）：

| 文件                              | 内容                              | 形状        |
| --------------------------------- | --------------------------------- | ----------- |
| `train_clean.pt` / `train_aug.pt` | `train.csv` 的 4 个视图           | N × 4 × 768 |
| `val_eval.pt`                     | `val.csv`，8 种测试扰动各一个视图 | N × 8 × 768 |
| `test/test_*.pt`                  | 每个 `test_*.csv`，同上           | N × 8 × 768 |

验证集和测试集的缓存由两组实验共用。视图的随机种子由 `features.view_seed` 和（图片序号, 视图序号）确定，所有分类头种子共用同一份缓存。已存在的缓存会跳过；文件原子写入，作业中断后重新运行同一条命令即可。

**训练**：每个 epoch 每张图随机取 4 个视图中的一个；AdamW，BCE；在 `learning_rates × weight_decays` 网格上按验证集（原图）BCE 选最优并早停（最多 400 轮，耐心 20）。然后只用验证集拟合温度 T，并取验证集平衡准确率最高的阈值。

输出 `<checkpoint_dir>/<experiment>/seed<k>/`：

- `head.pt`：线性头权重、`temperature`、`threshold`、选中的超参数、缓存元信息；
- `train_log.json`：整个网格的结果、每轮损失、验证集指标（校准前后）。

已存在的种子默认跳过；需要重训时加 `--overwrite`。测试由 `scripts/evaluate.py` 统一完成。
