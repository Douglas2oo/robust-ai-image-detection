"""CLIP 特征缓存入口骨架；尚未实现提取。"""

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
        message=(
            "尚未实现特征缓存；需按数据清单、编码器、预处理、增强配置及"
            "增强种子生成缓存标识，并区分原图与增强视图。\n"
        ),
    )


if __name__ == "__main__":
    main()
