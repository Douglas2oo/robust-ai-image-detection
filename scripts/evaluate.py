"""各模型共用的评测入口骨架；尚未实现评测。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.parse_args()
    parser.exit(
        status=2,
        message=(
            "尚未实现评测；需从检查点恢复模型、预处理、温度和阈值，"
            "并保存逐图预测及汇总指标。\n"
        ),
    )


if __name__ == "__main__":
    main()
