"""
YOLOv11-seg 训练脚本（our_method Phase 1 baseline + v2 增强版）。

训练 YOLOv11n-seg (2.6M) 和 YOLOv11s-seg (9.4M)，
使用 COCO 预训练权重 fine-tune。

v2 改进：
- 离线增强: 20x per image (vs 10x), 更强的 albumentations pipeline
- 在线增强: scale=0.9, degrees=15, shear=5, perspective, mixup, copy_paste

使用方式：
    python src/train.py                          # 默认 data_v3_augmented
    python src/train.py --data dataset/data_v3_aug_v2  # v2 增强数据
    python src/train.py --data dataset/data_v4   # 使用 data_v4
    python src/train.py --model n                # 只训练 nano
    python src/train.py --batch 32 --workers 16  # 服务器 A800 配置
    python src/train.py --resume                 # 从 last.pt 续训
    python src/train.py --resume --batch 32 --workers 16  # 服务器续训
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "data_v3_augmented"
OUTPUT_DIR = PROJECT_ROOT / "models" / "our_method"

# 服务器环境可能无 mlflow 数据库后端，允许文件存储
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# ── 固定参数（无需命令行调整） ──────────────────────
IMGSZ = 640


def fix_data_yaml(data_yaml_path):
    """修复 data.yaml 中的相对路径使其适配 YOLOv11。

    原始 yaml 来自 Roboflow 导出，路径如 '../train/images' 是相对于 data_v3/ 根目录。
    但 yaml 文件实际在 data_v3_augmented/ 子目录中，所以需要修正。
    """
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    base = data_yaml_path.parent

    for key in ("train", "val", "test"):
        if key not in config or not config[key]:
            continue
        val = config[key]
        candidates = [val]
        if val.startswith("../"):
            candidates.append(val.replace("../", "", 1))
            candidates.append(val.replace("../", "./", 1))
        found = None
        for cand in candidates:
            candidate_path = (base / cand).resolve()
            if candidate_path.exists():
                found = candidate_path
                break
        if found is None:
            found = (base / candidates[-1]).resolve()
        config[key] = str(found)

    fixed_path = base / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(fixed_path)


def train_model(model_name, data_yaml_path, args):
    """训练单个模型。"""
    from ultralytics import YOLO

    run_name = f"yolo11{model_name}-seg-v2"

    print(f"\n{'=' * 60}")
    print(f"训练 YOLOv11{model_name}-seg (v2 - 增强训练)")
    print(f"{'=' * 60}")

    # Device 处理
    device = args.device
    if device is not None:
        import torch
        if device == 0 and not torch.cuda.is_available():
            print("警告: CUDA 不可用，回退到 CPU 训练")
            device = "cpu"

    if args.resume:
        # 从 last.pt 续训
        last_pt = OUTPUT_DIR / run_name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"错误: resume 模式需要 {last_pt}，但文件不存在")
            sys.exit(1)
        model = YOLO(str(last_pt))
        print(f"续训模式: 从 {last_pt} 恢复训练")
        model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            resume=True,
            # 显式指定 project/name 覆盖 checkpoint 中泄露的 Windows 路径
            project=str(OUTPUT_DIR),
            name=run_name,
            exist_ok=True,
            device=device,
            batch=args.batch,
            workers=args.workers,
            amp=args.amp,
            plots=False,
            save=True,
            save_period=10,
            val=True,
            max_det=800,
        )
    else:
        model_file = str(PROJECT_ROOT / "models" / f"yolo11{model_name}-seg.pt")
        model = YOLO(model_file)
        model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=IMGSZ,
            batch=args.batch,
            lr0=args.lr,
            patience=args.patience,
            device=device,
            workers=args.workers,
            project=str(OUTPUT_DIR),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            cos_lr=True,
            warmup_epochs=3,
            # 在线数据增强 (ultralytics built-in)
            mosaic=1.0,
            close_mosaic=150,
            scale=0.9,          # 更宽的缩放范围 (默认 0.5)
            degrees=15,          # 随机旋转 ±15° (默认 0.0)
            shear=5,            # 随机剪切 ±5° (默认 0.0)
            perspective=0.0005, # 随机透视变换 (默认 0.0)
            flipud=0.1,         # 垂直翻转 10% (默认 0.0)
            mixup=0.1,          # mixup 增强 10% (默认 0.0)
            copy_paste=0.1,     # copy-paste 增强 10% (默认 0.0)
            # 保存
            save=True,
            save_period=10,
            val=True,
            max_det=800,
            amp=args.amp,
            plots=False,        # 离线环境跳过字体下载
        )

    print(f"\n训练完成！最佳权重: {OUTPUT_DIR / run_name / 'weights' / 'best.pt'}")
    return OUTPUT_DIR / run_name / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg baseline 训练")
    parser.add_argument("--model", choices=["n", "s", "both"], default="both",
                        help="训练哪个模型 (default: both)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数 (default: 200)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8 for RTX 3060, 建议服务器用 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率 (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=30,
                        help="早停 patience (default: 30)")
    parser.add_argument("--device", default=0,
                        help="设备: 0=cuda:0, cpu=cpu (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4, 建议服务器用 16)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", default=True,
                        help="禁用 AMP 混合精度（离线服务器需要，避免下载验证模型）")
    parser.add_argument("--resume", action="store_true",
                        help="从 last.pt 续训 (默认: 从头训练)")
    parser.add_argument("--name", type=str, default="v3",
                        help="训练 run 名称 (default: v3, 输出到 models/our_method/yolo11{model}-seg-{name}/)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET),
                        help="数据集目录路径 (default: dataset/data_v3_augmented)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"错误: data.yaml 不存在: {data_yaml}")
        sys.exit(1)

    fixed_yaml = fix_data_yaml(data_yaml)
    print(f"data.yaml 路径已修复: {fixed_yaml}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"训练配置: epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"device={args.device} workers={args.workers} amp={args.amp}")

    if args.model in ("n", "both"):
        train_model("n", fixed_yaml, args)

    if args.model in ("s", "both"):
        train_model("s", fixed_yaml, args)


if __name__ == "__main__":
    main()
