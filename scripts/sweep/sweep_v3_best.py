"""v3 best.pt overlap × conf 交叉网格搜索"""
import subprocess
import sys
from pathlib import Path

MODEL = "models/glass_bead_seg/yolo11n-seg-v3/weights/best.pt"
SOURCE = "dataset/data_v3/test/images"
OUTPUT_BASE = "outputs/v3_sweep_fine"
GT = "dataset/data_v3/test/labels"
IMAGES = "dataset/data_v3/test/images"

OV_RANGE = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
CONF_RANGE = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]

# ── 推理 ──
total = len(OV_RANGE) * len(CONF_RANGE)
count = 0
for ov in OV_RANGE:
    for c in CONF_RANGE:
        count += 1
        ov_s = str(ov).replace(".", "p")
        name = "best_ov{}_c{:02d}_tta".format(ov_s, int(c * 100))
        out_dir = Path(OUTPUT_BASE) / name
        if (out_dir / "labels").exists():
            print("[{}/{}] SKIP {}".format(count, total, name))
            continue
        cmd = [
            sys.executable, "src/inference.py",
            "--model", MODEL,
            "--source", SOURCE,
            "--output", str(out_dir),
            "--mode", "ellipse",
            "--conf", str(c),
            "--overlap", str(ov),
            "--tta",
        ]
        print("[{}/{}] RUN {}".format(count, total, name))
        subprocess.run(cmd, check=True)

# ── 评估 ──
print("\n" + "=" * 80)
print("{:<32s} {:>6s} {:>6s} {:>6s} {:>7s}".format(
    "Config", "F1", "Recall", "Prec", "CntErr"))
print("-" * 80)

results = []
for ov in OV_RANGE:
    for c in CONF_RANGE:
        ov_s = str(ov).replace(".", "p")
        name = "best_ov{}_c{:02d}_tta".format(ov_s, int(c * 100))
        pred_dir = Path(OUTPUT_BASE) / name / "labels"
        if not pred_dir.exists():
            continue
        proc = subprocess.run(
            [sys.executable, "src/eval.py",
             "--pred", str(pred_dir),
             "--gt", GT,
             "--images", IMAGES],
            capture_output=True, text=True,
        )
        f1 = rec = prec = cnt_err = 0.0
        for line in proc.stdout.splitlines():
            if "F1@0.5:" in line:
                f1 = float(line.split()[-1])
            elif "Recall@0.5:" in line:
                rec = float(line.split()[-1])
            elif "Precision@0.5:" in line:
                prec = float(line.split()[-1])
            elif "平均计数误差:" in line:
                cnt_err = float(line.split()[-1])

        results.append((name, f1, rec, prec, cnt_err))
        print("{:<32s} {:6.4f} {:6.4f} {:6.4f} {:6.1%}".format(
            name, f1, rec, prec, cnt_err))

# ── Top 5 ──
print("\n" + "=" * 80)
print("Top 5 (按 F1):")
print("-" * 80)
for i, (name, f1, rec, prec, cnt_err) in enumerate(
    sorted(results, key=lambda x: -x[1])[:5], 1
):
    print("{}. {:<28s} F1={:.4f}  Rec={:.4f}  Prec={:.4f}  Err={:.1%}".format(
        i, name, f1, rec, prec, cnt_err))

print("\nTop 5 (按计数误差):")
print("-" * 80)
for i, (name, f1, rec, prec, cnt_err) in enumerate(
    sorted(results, key=lambda x: x[4])[:5], 1
):
    print("{}. {:<28s} F1={:.4f}  Rec={:.4f}  Prec={:.4f}  Err={:.1%}".format(
        i, name, f1, rec, prec, cnt_err))
