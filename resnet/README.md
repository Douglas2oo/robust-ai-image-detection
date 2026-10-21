# ResNet 小队

负责 ResNet-50 基线的模型、训练与验证集选参。当前文件是开发骨架，尚未实现训练。

- `model.py`：ImageNet 预训练 ResNet-50＋单 logit 分类头。
- `train.py`：训练入口；复用 `common/data.py`、公共指标与校准模块。
- `config.yaml`：独立实验配置；试跑后确定微调范围和所有 `null` 参数。

在项目根目录查看入口：

```bash
python -m resnet.train --help
```

完成实现后计划使用：

```bash
python -m resnet.train --config resnet/config.yaml
```

使用与 CLIP 相同的数据划分和评测条件，保留 ResNet 预训练权重要求的归一化。保存配置快照、种子、检查点及验证结果；由统一评测入口生成正式测试指标。
