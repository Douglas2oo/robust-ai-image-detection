"""数据审计、去重和固定划分入口骨架；尚未实现数据处理。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="本地原始图片目录")
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.parse_args()
    parser.exit(
        status=2,
        message="尚未实现数据准备；请先确定来源映射、类别集合、去重和划分规则。\n",
    )


if __name__ == "__main__":
    main()
