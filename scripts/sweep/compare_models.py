"""
模型对比脚本 — v1 vs v3，nano vs small，4 张测试图。

自动：
  1. 准备统一测试集（test + valid → outputs/test4/）
  2. 对所有可用模型推理（baseline + ellipse）
  3. 评估 vs GT
  4. 打印对比表

使用方式：
    python scripts/sweep/compare_models.py
    python scripts/sweep/compare_models.py --mode baseline
    v3 模型训练完成后运行此脚本得到最终对比结果。
"""

import os
import sys
import re
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 配置 ────────────────────────────────────────────
TEST_SETS = [
    PROJECT_ROOT / "dataset" / "data_v3_aug" / "test",
    PROJECT_ROOT / "dataset" / "data_v3_aug" / "valid",
]

MODELS = [
    ("v1-nano", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11n-seg-v1" / "weights" / "best.pt"),
    ("v1-small", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11s-seg-v1" / "weights" / "best.pt"),
    ("v3-nano", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11n-seg-v3" / "weights" / "best.pt"),
    ("v3-nano-best", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11n-seg-v3" / "weights" / "best.pt"),
    ("v3-nano-last", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11n-seg-v3" / "weights" / "last.pt"),
    ("v3-small", PROJECT_ROOT / "models" / "glass_bead_seg" / "yolo11s-seg-v3" / "weights" / "best.pt"),
]

# 去重（v3-nano 和 v3-nano-best 指向同一个文件，只保留一个）
_seen = set()
MODELS_DEDUP = []
for label, path in MODELS:
    key = str(path.resolve()) if path.exists() else label
    if key not in _seen:
        _seen.add(key)
        MODELS_DEDUP.append((label, path))
MODELS = MODELS_DEDUP


def run_cmd(cmd, desc=""):
    print(f"\n{'─' * 60}")
    print(f"  >>> {desc or cmd}")
    print(f"{'─' * 60}")
    return subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT)).returncode


def _extract(log_text, pattern):
    m = re.search(pattern, log_text)
    return f"{float(m.group(1)):.4f}" if m else "-"


def main():
    parser = argparse.ArgumentParser(description="v1 vs v3 模型对比")
    parser.add_argument("--mode", choices=["baseline", "ellipse"], default="ellipse",
                        help="推理模式 (default: ellipse)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="置信度阈值 (default: 0.3)")
    parser.add_argument("--dedup-iou", type=float, default=0.0,
                        help="去重 IoU 阈值（0 禁用，推荐 0.15-0.2 for best.pt）")
    parser.add_argument("--dedup-dist", type=float, default=30,
                        help="去重中心距离阈值，像素 (default: 30)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理，仅用已有预测做评估")
    args = parser.parse_args()

    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
    TEST4_DIR = OUTPUT_DIR / "test4"

    # ── 准备统一测试集 ──
    if not TEST4_DIR.exists() or not args.skip_inference:
        if TEST4_DIR.exists():
            shutil.rmtree(TEST4_DIR)
        (TEST4_DIR / "images").mkdir(parents=True, exist_ok=True)
        (TEST4_DIR / "labels").mkdir(parents=True, exist_ok=True)

        for ts in TEST_SETS:
            for src_dir, dst_dir in [("images", "images"), ("labels", "labels")]:
                src = ts / src_dir
                if not src.exists():
                    continue
                for f in src.iterdir():
                    if f.suffix.lower() in {".png", ".jpg", ".txt"}:
                        shutil.copy2(str(f), str(TEST4_DIR / dst_dir / f.name))

        n_img = len(list((TEST4_DIR / "images").iterdir()))
        n_lbl = len(list((TEST4_DIR / "labels").iterdir()))
        print(f"测试集准备好: {n_img} 张图, {n_lbl} 个标注")

    print("=" * 70)
    print(f"模型对比：v1 vs v3")
    print(f"模式: {args.mode}  conf={args.conf}")
    if args.dedup_iou > 0:
        print(f"去重: IoU>{args.dedup_iou}, Dist<={args.dedup_dist}px")
    print(f"测试集: {TEST4_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    # 构建去重后缀（推理和评估共用）
    dedup_suffix = f"_dedup{args.dedup_iou}".replace(".", "") if args.dedup_iou > 0 else ""

    # ── 推理 ──
    if not args.skip_inference:
        for label, weight_path in MODELS:
            if not weight_path.exists():
                print(f"\n  跳过 {label}: 权重不存在 ({weight_path})")
                continue

            infer_out = OUTPUT_DIR / f"predictions_{label.replace('-', '_')}_{args.mode}{dedup_suffix}"
            if infer_out.exists():
                shutil.rmtree(infer_out)

            cmd = (
                f'"{sys.executable}" src/inference.py'
                f' --model "{weight_path}"'
                f' --source "{TEST4_DIR / "images"}"'
                f' --output "{infer_out}"'
                f' --mode {args.mode}'
                f' --conf {args.conf}'
                f' --dedup-iou {args.dedup_iou}'
                f' --dedup-dist {args.dedup_dist}'
            )
            rc = run_cmd(cmd, f"推理 {label}")
            if rc != 0:
                print(f"  推理失败: {label}")

    # ── 评估 ──
    print(f"\n{'=' * 70}")
    print(f"  评估中...")
    print(f"{'=' * 70}")

    summary = []

    for label, weight_path in MODELS:
        if not weight_path.exists():
            continue

        infer_out = OUTPUT_DIR / f"predictions_{label.replace('-', '_')}_{args.mode}{dedup_suffix}"
        pred_labels = infer_out / "labels"
        if not pred_labels.exists():
            summary.append((label, "-", "-", "-", "-"))
            print(f"  跳过 {label}: predictions/labels 不存在")
            continue

        eval_log = OUTPUT_DIR / f"eval_{label.replace('-', '_')}_{args.mode}{dedup_suffix}.log"
        cmd = (
            f'"{sys.executable}" src/eval.py'
            f' --gt "{TEST4_DIR / "labels"}"'
            f' --pred "{pred_labels}"'
            f' --images "{TEST4_DIR / "images"}"'
            f' > "{eval_log}" 2>&1'
        )
        run_cmd(cmd, f"评估 {label}")

        if not eval_log.exists():
            summary.append((label, "-", "-", "-", "-"))
            continue

        text = eval_log.read_text()
        summary.append((
            label,
            _extract(text, r"F1@0\.5:\s+([0-9.]+)"),
            _extract(text, r"Recall@0\.5:\s+([0-9.]+)"),
            _extract(text, r"Precision@0\.5:\s+([0-9.]+)"),
            _extract(text, r"平均计数误差:\s+([0-9.]+)"),
        ))

    # ── 打印对比表 ──
    dedup_info = f", dedup={args.dedup_iou}/{args.dedup_dist}" if args.dedup_iou > 0 else ""
    print(f"\n{'=' * 70}")
    print(f"  对比结果摘要 ({args.mode}, IoU@0.5, conf={args.conf}{dedup_info})")
    print(f"{'=' * 70}")

    header = f"  {'模型':<18} {'F1':>8} {'Recall':>8} {'Precision':>10} {'计数误差':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in summary:
        print(f"  {row[0]:<18} {row[1]:>8} {row[2]:>8} {row[3]:>10} {row[4]:>10}")

    valid = [(r[0], float(r[1])) for r in summary if r[1] != "-"]
    if valid:
        best = max(valid, key=lambda x: x[1])
        print(f"\n  最佳 F1: {best[0]} = {best[1]:.4f}")

    print(f"\n详细日志: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
