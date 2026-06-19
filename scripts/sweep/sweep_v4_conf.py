"""v4 best.pt 置信度 + TTA 快速扫描"""
import subprocess, sys
from pathlib import Path

MODEL = "models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt"
SOURCE = "dataset/data_v4_aug/test/images"
OUTPUT_BASE = "outputs/v4_sweep"
GT = "dataset/data_v4_aug/test/labels"
IMAGES = "dataset/data_v4_aug/test/images"

configs = [
    ("c035",      ["--conf", "0.35"]),
    ("c040",      ["--conf", "0.40"]),
    ("c045",      ["--conf", "0.45"]),
    ("c050",      ["--conf", "0.50"]),
    ("c035_tta",  ["--conf", "0.35", "--tta"]),
    ("c040_tta",  ["--conf", "0.40", "--tta"]),
    ("c045_tta",  ["--conf", "0.45", "--tta"]),
    ("c050_tta",  ["--conf", "0.50", "--tta"]),
]

for name, extra_args in configs:
    out_dir = Path(OUTPUT_BASE) / name
    if (out_dir / "labels").exists():
        print(f"[SKIP] {name}")
        continue
    cmd = [
        sys.executable, "src/inference.py",
        "--model", MODEL, "--source", SOURCE,
        "--output", str(out_dir), "--mode", "ellipse",
    ] + extra_args
    print(f"[RUN] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\n" + "=" * 80)
results = []
for name, _ in configs:
    pred_dir = Path(OUTPUT_BASE) / name / "labels"
    if not pred_dir.exists():
        continue
    proc = subprocess.run(
        [sys.executable, "src/eval.py",
         "--pred", str(pred_dir), "--gt", GT, "--images", IMAGES],
        capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ["F1@", "Recall@", "Precision@", "平均"]):
            results.append(f"{name:20s} | {line.strip()}")
    print(f"=== {name} ===")
    print(proc.stdout)

print("\n" + "=" * 80)
print("汇总:")
for r in results:
    print(r)
