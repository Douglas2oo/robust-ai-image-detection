"""统一二分类指标：待实现。

AI 为正类。报告 accuracy、balanced accuracy、precision、recall、F1、
AUROC 和 average precision；阈值指标使用验证集确定的固定阈值。
AUROC/AP 使用连续得分，不使用阈值化标签。
分别报告各来源、未见生成器宏平均及独立外部数据结果。
共享真实测试图片不能在样本不确定性分析中算作新增独立样本。
"""
