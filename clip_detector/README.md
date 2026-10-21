# CLIP 小队

统一维护 CLIP 基础实验与数据增强实验。当前文件是开发骨架，尚未实现模型、训练和缓存。

| 文件 | 职责 |
| --- | --- |
| `model.py` | 冻结 CLIP ViT-L/14＋线性二分类头，两组实验共用 |
| `train.py` | 唯一的 CLIP 训练入口，按配置决定是否启用增强 |
| `augmentations.py` | 对真假图片对称施加训练增强 |
| `cache_features.py` | 原图或增强视图特征提取与缓存 |
| `configs/baseline.yaml` | 无额外扰动增强的实验配置 |
| `configs/augmented.yaml` | 增强实验配置 |

在项目根目录查看入口：

```bash
python -m clip_detector.train --help
python -m clip_detector.cache_features --help
```

完成实现后计划使用：

```bash
python -m clip_detector.train --config clip_detector/configs/baseline.yaml
python -m clip_detector.train --config clip_detector/configs/augmented.yaml
```

两组实验分别训练和保存分类头，但使用相同模型实现、数据划分和对应训练种子。主消融保持其他训练设置一致；若分别调参，须注明比较口径。

增强必须作用于图片后再提取特征。若采用多视图缓存，记录每张图的视图数、增强种子及原图采样规则；不要用原图特征缓存代替增强特征。缓存标识包括数据清单版本、编码器权重、基础预处理、特征归一化与增强配置。输入特征相同时，可以跨分类头训练种子复用缓存。

验证数据用于选参、校准和阈值选择；默认使用固定的基础预处理。测试扰动由公共评测流程实施。
