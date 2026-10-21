"""CLIP 训练时的图像增强：待实现。

根据 augmented.yaml 对真假图像应用相同概率的 JPEG 重压缩、
缩放与轻度 Gaussian blur。baseline 配置不启用这些额外增强。
图像增强在编码器特征提取及张量归一化之前执行。
增强概率、强度、顺序和随机种子必须显式记录。
正式测试扰动由 common/corruptions.py 单独定义。
"""
