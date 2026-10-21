"""统一数据读取：待实现。

按 manifests/README.md 的固定清单读取图片；Real=0，AI=1。
relative_path 相对于配置中的 data.root，不在训练脚本内部重新划分。
返回稳定 image_id、图像、标签与来源元数据，保留预测结果可追溯性。
图像读取与具体模型预处理分离。
"""
