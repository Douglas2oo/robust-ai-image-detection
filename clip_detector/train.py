"""两组 CLIP 实验共用的训练入口骨架；尚未实现训练。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CLIP YAML 配置路径")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"配置文件不存在：{args.config}")
    parser.exit(
        status=2,
        message="尚未实现 CLIP 训练；请先完成模型、特征提取和训练流程。\n",
    )


if __name__ == "__main__":
    main()
