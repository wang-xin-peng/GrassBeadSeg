"""v4 best.pt 大 overlap + conf 细调 + soft-nms"""
import subprocess, sys
from pathlib import Path

MODEL = "models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt"
SOURCE = "dataset/data_v4_aug/test/images"
OUTPUT_BASE = "outputs/v4_sweep_r3"
GT = "dataset/data_v4_aug/test/labels"
IMAGES = "dataset/data_v4_aug/test/images"

configs = []

# round1: large overlap × fine conf
for ov in [0.32, 0.35, 0.38, 0.40]:
    for c in [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42]:
        ov_s = str(ov).replace(".", "p")
        name = "ov{}_c{:02d}_tta".format(ov_s, int(c * 100))
        configs.append((name, ["--conf", str(c), "--overlap", str(ov), "--tta"]))

# round2: soft-nms trials (at promising regions from round1/grid results)
for ov, c in [(0.25, 0.38), (0.30, 0.41), (0.35, 0.42)]:
    ov_s = str(ov).replace(".", "p")
    name = "ov{}_c{:02d}_soft_tta".format(ov_s, int(c * 100))
    configs.append((name, ["--conf", str(c), "--overlap", str(ov), "--tta", "--soft-nms"]))

# round3: no-TTA at best combos
for ov, c in [(0.25, 0.38), (0.30, 0.41), (0.35, 0.42)]:
    ov_s = str(ov).replace(".", "p")
    name = "ov{}_c{:02d}_notta".format(ov_s, int(c * 100))
    configs.append((name, ["--conf", str(c), "--overlap", str(ov)]))

seen = {}
unique = []
for name, args in configs:
    if name not in seen:
        seen[name] = True
        unique.append((name, args))
configs = unique

total = len(configs)
for i, (name, extra_args) in enumerate(configs, 1):
    out_dir = Path(OUTPUT_BASE) / name
    if (out_dir / "labels").exists():
        print("[{}/{}] SKIP {}".format(i, total, name))
        continue
    cmd = [
        sys.executable, "src/inference.py",
        "--model", MODEL, "--source", SOURCE,
        "--output", str(out_dir), "--mode", "ellipse",
    ] + extra_args
    print("[{}/{}] RUN {}".format(i, total, name))
    subprocess.run(cmd, check=True)

print("\n" + "=" * 85)
print("{:<32s} {:>6s} {:>6s} {:>6s} {:>7s}".format(
    "Config", "F1", "Recall", "Prec", "CntErr"))
print("-" * 85)

results = []
for name, _ in configs:
    pred_dir = Path(OUTPUT_BASE) / name / "labels"
    if not pred_dir.exists():
        continue
    proc = subprocess.run(
        [sys.executable, "src/eval.py",
         "--pred", str(pred_dir), "--gt", GT, "--images", IMAGES],
        capture_output=True, text=True)
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

print("\nTop 5 (F1):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: -x[1])[:5], 1):
    print("{}. {:<28s} F1={:.4f} R={:.4f} P={:.4f} Err={:.1%}".format(i, n, f1, r, p, e))

print("\nTop 5 (计数误差):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: x[4])[:5], 1):
    print("{}. {:<28s} F1={:.4f} R={:.4f} P={:.4f} Err={:.1%}".format(i, n, f1, r, p, e))
