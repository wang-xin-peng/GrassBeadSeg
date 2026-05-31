"""
训练 RF-DETR 分割模型。

使用方式：
  1. 确保 models/rfdetr_seg_large_pretrained.pth 已存在
  2. conda activate gbseg
  3. python scripts/train/train_rfdetr.py

训练完成后，最佳检查点将保存在 output_dir/ 下。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = PROJECT_ROOT / "models" / "rfdetr_seg_large_pretrained.pth"
DATASET_DIR = PROJECT_ROOT / "dataset" / "data_v1_augmented"
OUTPUT_DIR = PROJECT_ROOT / "models" / "trained"

EPOCHS = 200
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1e-4
RESOLUTION = 624


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"错误: 数据集路径不存在: {DATASET_DIR}")
        sys.exit(1)

    if not PRETRAINED_PATH.exists():
        print(f"错误: 预训练权重不存在: {PRETRAINED_PATH}")
        print("请先在本地有网络的环境下运行 models/download_pretrained.py 下载权重")
        sys.exit(1)

    try:
        from rfdetr import RFDETRSegLarge
    except ImportError:
        print("错误: 请先安装 rfdetr: pip install rfdetr")
        sys.exit(1)

    print("=" * 60)
    print("RF-DETR 分割模型训练")
    print("=" * 60)
    print(f"预训练权重: {PRETRAINED_PATH}")
    print(f"数据集: {DATASET_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练轮数: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRAD_ACCUM_STEPS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"分辨率: {RESOLUTION}")
    print("=" * 60)

    model = RFDETRSegLarge(pretrain_weights=str(PRETRAINED_PATH))

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        resolution=RESOLUTION,
        output_dir=str(OUTPUT_DIR),
        early_stopping=True,
        early_stopping_patience=30,
        early_stopping_min_delta=0.001,
        tensorboard=True,
    )

    print(f"\n训练完成！检查点保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()