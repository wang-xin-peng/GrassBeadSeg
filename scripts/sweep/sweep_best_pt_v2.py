"""
best.pt 优化第二轮 — conf=0.33 基线, 尝试 soft-NMS / dedup微调 / yolo_iou
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = PROJECT_ROOT / "models" / "grass_bead_seg" / "yolo11n-seg-v2" / "weights" / "best.pt"
TEST_IMAGES = PROJECT_ROOT / "outputs" / "model_comparison" / "test4" / "images"
TEST_LABELS = PROJECT_ROOT / "outputs" / "model_comparison" / "test4" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sweep_best_pt_v2"

CONFIGS = [
    # (label,                     yolo_iou, dedup_iou, dedup_dist, overlap, conf,  tta,  soft_nms)
    ("baseline",                  0.5,      0.10,      40,         0.15,    0.33,  True, False),
    ("softnms",                   0.5,      0.10,      40,         0.15,    0.33,  True, True),
    ("softnms+d12",               0.5,      0.12,      40,         0.15,    0.33,  True, True),
    ("softnms+yolo45",            0.45,     0.10,      40,         0.15,    0.33,  True, True),
    ("softnms+yolo45+d12",        0.45,     0.12,      40,         0.15,    0.33,  True, True),
    ("yolo45",                    0.45,     0.10,      40,         0.15,    0.33,  True, False),
    ("yolo45+d12",                0.45,     0.12,      40,         0.15,    0.33,  True, False),
    ("d12",                       0.5,      0.12,      40,         0.15,    0.33,  True, False),
]

PYTHON = sys.executable


def extract(log_text, pattern):
    m = re.search(pattern, log_text)
    return float(m.group(1)) if m else None


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("=" * 85)
    print("best.pt 优化第二轮 (conf=0.33 + TTA 基线)")
    print(f"配置数: {len(CONFIGS)}")
    print("=" * 85)

    results = []

    for label, yolo_iou, dedup_iou, dedup_dist, overlap, conf, tta, soft_nms in CONFIGS:
        pred_dir = OUTPUT_DIR / f"pred_{label}"
        pred_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f'"{PYTHON}" src/inference.py'
            f' --model "{MODEL}"'
            f' --source "{TEST_IMAGES}"'
            f' --output "{pred_dir}"'
            f' --mode ellipse'
            f' --yolo-iou {yolo_iou}'
            f' --dedup-iou {dedup_iou}'
            f' --dedup-dist {dedup_dist}'
            f' --overlap {overlap}'
            f' --conf {conf}'
        )
        if tta:
            cmd += " --tta"
        if soft_nms:
            cmd += " --soft-nms"

        print(f"\n  [{label}] ...", flush=True)
        rc = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True).returncode
        if rc != 0:
            print(f"    推理失败!", flush=True)
            results.append((label, None, None, None, None))
            continue

        eval_cmd = (
            f'"{PYTHON}" src/eval.py'
            f' --gt "{TEST_LABELS}"'
            f' --pred "{pred_dir / "labels"}"'
            f' --images "{TEST_IMAGES}"'
        )
        rc = subprocess.run(eval_cmd, shell=True, cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True)
        log_text = rc.stdout
        (OUTPUT_DIR / f"eval_{label}.log").write_text(log_text)

        f1 = extract(log_text, r"F1@0\.5:\s+([0-9.]+)")
        recall = extract(log_text, r"Recall@0\.5:\s+([0-9.]+)")
        precision = extract(log_text, r"Precision@0\.5:\s+([0-9.]+)")
        count_err = extract(log_text, r"平均计数误差:\s+([0-9.]+)")

        results.append((label, f1, recall, precision, count_err))

    print("\n" + "=" * 85)
    print("  网格搜索结果 (best.pt v2)")
    print("=" * 85)
    header = f"  {'配置':<24} {'F1':>7} {'Recall':>7} {'Prec':>7} {'CntErr':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_f1 = None
    best_cnt = None
    for r in results:
        label, f1, rec, prec, cerr = r
        f1s = f"{f1:.4f}" if f1 else "-"
        recs = f"{rec:.4f}" if rec else "-"
        precs = f"{prec:.4f}" if prec else "-"
        cerrs = f"{cerr:.4f}" if cerr else "-"
        print(f"  {label:<24} {f1s:>7} {recs:>7} {precs:>7} {cerrs:>7}")

        if f1 and (best_f1 is None or f1 > best_f1[1]):
            best_f1 = (label, f1)
        if cerr is not None and cerr >= 0 and (best_cnt is None or cerr < best_cnt[1]):
            best_cnt = (label, cerr)

    if best_f1:
        print(f"\n  最佳 F1:   {best_f1[0]} = {best_f1[1]:.4f}")
    if best_cnt:
        print(f"  最佳计数:   {best_cnt[0]} = {best_cnt[1]:.4f}")

    print(f"\n详细输出: {OUTPUT_DIR}")
    print("=" * 85)


if __name__ == "__main__":
    main()
