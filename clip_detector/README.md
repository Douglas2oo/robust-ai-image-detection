# CLIP Detector

两组 CLIP 实验共用冻结 OpenAI CLIP ViT-L/14 编码器和线性二分类头。C0 使用原图特征；C1 对每张训练图片生成两个固定增强视图。两组验证均使用原图。

| 文件 | 职责 |
| --- | --- |
| `model.py` | 冻结 CLIP ViT-L/14＋线性二分类头，两组实验共用 |
| `train.py` | 唯一的 CLIP 训练入口，按配置决定是否启用增强 |
| `augmentations.py` | 对真假图片对称施加训练增强 |
| `cache_features.py` | 原图或增强视图特征提取与缓存 |
| `configs/baseline.yaml` | 无额外扰动增强的实验配置 |
| `configs/augmented.yaml` | 增强实验配置 |

从项目根目录运行。首次运行先生成特征缓存：

```bash
python -m clip_detector.cache_features --config clip_detector/configs/baseline.yaml
python -m clip_detector.cache_features --config clip_detector/configs/augmented.yaml
```

再分别训练三个固定种子；每次运行使用独立输出目录，已存在的种子目录不会被覆盖：

```bash
for seed in 42 43 44; do
  python -m clip_detector.train --config clip_detector/configs/baseline.yaml --seed "$seed"
  python -m clip_detector.train --config clip_detector/configs/augmented.yaml --seed "$seed"
done
```

缓存使用官方 CLIP 图像预处理并将归一化后的 768 维特征以 float32 保存。增强配置为各自独立的 p=0.5：JPEG Q=50–100、随机缩放 0.5–1.0 后恢复尺寸、Gaussian blur sigma=0.1–1.0。视图种子固定于配置，因此三个分类头种子复用相同缓存。特征提取默认每 512 张源图片保存一个 chunk；命令中断后直接重新执行同一条缓存命令，会校验并跳过已完成 chunk。

训练默认 AdamW (`lr=1e-3`, `weight_decay=1e-4`)，最多 20 轮，按验证 BCE loss 早停（patience 5）。每轮结束后原子更新 `last.pt`，验证 loss 改善时立即原子更新 `best.pt`。日志、指标和逐图验证 logits 保存在对应的 `outputs/` 子目录。配置中的未填写字段会被拒绝。

训练中断后使用相同配置和 seed 恢复；不带路径的 `--resume` 会自动使用该 seed 目录中的 `last.pt`：

```bash
python -m clip_detector.train \
  --config clip_detector/configs/baseline.yaml \
  --seed 42 \
  --resume
```

也可以显式指定 checkpoint：

```bash
python -m clip_detector.train \
  --config clip_detector/configs/baseline.yaml \
  --seed 42 \
  --resume checkpoints/clip_baseline/seed_42/last.pt
```

`last.pt` 包含当前分类头、优化器、已完成 epoch、最佳权重、早停计数、训练历史和随机数状态。恢复时会校验完整配置、seed 以及 train/val 缓存签名；配置不一致会拒绝续跑。训练最多重跑中断时尚未完成的一个 epoch，缓存最多重跑尚未完成的一个 chunk。必须保留 `cache/`、`checkpoints/` 和 `outputs/` 才能续跑。

数据清单必须提供 `manifests/README.md` 中的全部字段；训练和验证行的 `split` 必须分别为 `train` 和 `val`。缓存前会拒绝重复 ID、跨训练/验证重复图像或重复组。温度缩放、阈值和正式测试由公共评测流程负责。

缓存或完整训练流程的轻量测试：

```bash
python -m unittest discover -s tests
```
