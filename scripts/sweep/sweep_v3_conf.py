"""v3 last.pt 置信度 + TTA 快速扫描"""
import subprocess
import sys
from pathlib import Path

MODEL = "models/our_method/yolo11n-seg-v3/weights/last.pt"
SOURCE = "dataset/data_v3/test/images"
OUTPUT_BASE = "outputs/v3_sweep"
GT = "dataset/data_v3/test/labels"
IMAGES = "dataset/data_v3/test/images"

configs = [
    ("last_c035",      ["--conf", "0.35"]),
    ("last_c040",      ["--conf", "0.40"]),
    ("last_c045",      ["--conf", "0.45"]),
    ("last_c050",      ["--conf", "0.50"]),
    ("last_c035_tta",  ["--conf", "0.35", "--tta"]),
    ("last_c040_tta",  ["--conf", "0.40", "--tta"]),
    ("last_c045_tta",  ["--conf", "0.45", "--tta"]),
    ("last_c050_tta",  ["--conf", "0.50", "--tta"]),
]

# ── 推理（串行，GPU 不能并行） ──
for name, extra_args in configs:
    out_dir = Path(OUTPUT_BASE) / name
    if (out_dir / "labels").exists():
        print(f"[SKIP] {name} 已存在")
        continue
    cmd = [
        sys.executable, "src/inference.py",
        "--model", MODEL,
        "--source", SOURCE,
        "--output", str(out_dir),
        "--mode", "ellipse",
    ] + extra_args
    print(f"[RUN] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ── 评估 ──
print("\n" + "=" * 70)
results = []
for name, _ in configs:
    pred_dir = Path(OUTPUT_BASE) / name / "labels"
    if not pred_dir.exists():
        print(f"[SKIP] {name}: 预测目录不存在")
        continue
    proc = subprocess.run(
        [sys.executable, "src/eval.py",
         "--pred", str(pred_dir),
         "--gt", GT,
         "--images", IMAGES],
        capture_output=True, text=True,
    )
    # 提取汇总行
    for line in proc.stdout.splitlines():
        if "F1" in line or "Recall" in line or "Err" in line or "平均" in line:
            results.append(f"{name:20s} | {line.strip()}")
    print(f"=== {name} ===")
    print(proc.stdout)

print("\n" + "=" * 70)
print("汇总:")
for r in results:
    print(r)
