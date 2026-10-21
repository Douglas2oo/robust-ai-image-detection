"""单张图片推理入口骨架；尚未实现推理。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="JPEG 或 PNG 图片")
    parser.parse_args()
    parser.exit(
        status=2,
        message="尚未实现推理；需恢复模型及校准参数后输出类别和 P(AI)。\n",
    )


if __name__ == "__main__":
    main()
