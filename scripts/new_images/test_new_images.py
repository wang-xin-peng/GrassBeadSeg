"""
从 dataset/new_images/ 随机挑 10 张，用 v3 最佳参数推理。

使用方式（服务器）：
    cd /path/to/GlassBeadSeg
    python scripts/test_new_images.py

本地测试（单张看效果）：
    python scripts/test_new_images.py --local
"""
import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11n-seg-v3" / "weights" / "last.pt"
SOURCE_DIR = PROJECT_ROOT / "dataset" / "new_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "new_images_test"

# ── v3 最佳参数 ──
MODE = "ellipse"
OVERLAP = 0.32
CONF = 0.38
TTA = True
YOLO_IOU = 0.7
DEDUP_IOU = 0.15
DEDUP_DIST = 30


def main():
    parser = argparse.ArgumentParser(description="new_images 抽样推理")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument("--n", type=int, default=10, help="抽样数量 (default: 10)")
    parser.add_argument("--local", action="store_true", help="本机运行（不依赖服务器路径）")
    parser.add_argument("--device", type=int, default=0, help="CUDA 设备 (default: 0)")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"错误: new_images 目录不存在: {SOURCE_DIR}")
        sys.exit(1)

    # ── 随机选 N 张 ──
    images = sorted(SOURCE_DIR.glob("*.png"))
    if not images:
        images = sorted(SOURCE_DIR.glob("*.jpg"))
    if not images:
        print(f"错误: {SOURCE_DIR} 中没有图片")
        sys.exit(1)

    random.seed(args.seed)
    selected = random.sample(images, min(args.n, len(images)))

    print(f"从 {len(images)} 张中随机选了 {len(selected)} 张:")
    for img in selected:
        print(f"  {img.name}")

    # ── 准备输入目录 ──
    input_dir = OUTPUT_DIR / "images"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    for img in selected:
        shutil.copy2(str(img), str(input_dir / img.name))

    # ── 推理 ──
    pred_dir = OUTPUT_DIR / "predictions"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "inference.py"),
        "--model", str(MODEL),
        "--source", str(input_dir),
        "--output", str(pred_dir),
        "--mode", MODE,
        "--overlap", str(OVERLAP),
        "--conf", str(CONF),
        "--yolo-iou", str(YOLO_IOU),
        "--dedup-iou", str(DEDUP_IOU),
        "--dedup-dist", str(DEDUP_DIST),
        "--device", str(args.device),
    ]
    if TTA:
        cmd.append("--tta")

    print(f"\n推理命令: {' '.join(cmd)}")
    print(f"输出目录: {pred_dir}")
    print(f"可视化:   {pred_dir / 'visualizations'}")
    print("=" * 60)

    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if rc.returncode != 0:
        print(f"\n推理失败，返回码: {rc.returncode}")
        sys.exit(1)

    print(f"\n完成。")
    print(f"  labels:        {pred_dir / 'labels'}")
    print(f"  visualizations: {pred_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
