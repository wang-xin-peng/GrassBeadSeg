"""
在本地网络环境下载 RF-DETR 预训练权重。
运行此脚本需要网络连接，会将权重保存到 models/rfdetr_seg_large/ 目录下。
之后将整个项目目录传输到服务器即可。
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "rfdetr_seg_large"
OUTPUT_PATH = MODELS_DIR / "pretrained_300.pth"
CACHE_PATH = Path.home() / ".roboflow" / "models" / "rf-detr-seg-large.pt"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"预训练权重已存在: {OUTPUT_PATH}")
        return

    if CACHE_PATH.exists():
        print(f"从缓存复制: {CACHE_PATH}")
        shutil.copy2(str(CACHE_PATH), str(OUTPUT_PATH))
        print(f"预训练权重已保存到: {OUTPUT_PATH}")
        return

    print("正在下载 RF-DETR-Seg-Large 预训练权重 (首次运行将下载到缓存)...")

    try:
        from rfdetr import RFDETRSegLarge
    except ImportError:
        print("错误: 请先安装 rfdetr: pip install rfdetr")
        sys.exit(1)

    _ = RFDETRSegLarge()

    if CACHE_PATH.exists():
        shutil.copy2(str(CACHE_PATH), str(OUTPUT_PATH))
        print(f"预训练权重已保存到: {OUTPUT_PATH}")
    else:
        print(f"错误: 下载后未找到缓存文件: {CACHE_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()