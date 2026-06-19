"""v4 best.pt 精细调参网格搜索"""
import subprocess, sys
from pathlib import Path

MODEL = "models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt"
SOURCE = "dataset/data_v4_aug/test/images"
OUTPUT_BASE = "outputs/v4_sweep_param"
GT = "dataset/data_v4_aug/test/labels"
IMAGES = "dataset/data_v4_aug/test/images"

configs = []

# conf 细调 (TTA)
for c in [0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43]:
    name = "c{:02d}_tta".format(int(c * 100))
    configs.append((name, ["--conf", str(c), "--tta"]))

# dedup_iou 扫描 (conf=0.40, TTA)
for diou in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
    name = "c040_tta_diou{}".format(str(diou).replace(".", "p"))
    configs.append((name, ["--conf", "0.40", "--tta", "--dedup-iou", str(diou)]))

# dedup_dist 扫描 (conf=0.40, TTA)
for ddist in [25, 30, 35, 40, 45, 50, 55]:
    name = "c040_tta_d{}".format(ddist)
    configs.append((name, ["--conf", "0.40", "--tta", "--dedup-dist", str(ddist)]))

# yolo_iou 扫描 (conf=0.40, TTA)
for yiou in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    name = "c040_tta_yiou{}".format(str(yiou).replace(".", "p"))
    configs.append((name, ["--conf", "0.40", "--tta", "--yolo-iou", str(yiou)]))

# overlap 扫描 (conf=0.40, TTA)
for ov in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    name = "c040_tta_ov{}".format(str(ov).replace(".", "p"))
    configs.append((name, ["--conf", "0.40", "--tta", "--overlap", str(ov)]))

total = len(configs)
print("共 {} 组实验\n".format(total))
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
print("{:<28s} {:>6s} {:>6s} {:>6s} {:>7s}".format(
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
    print("{:<28s} {:6.4f} {:6.4f} {:6.4f} {:6.1%}".format(
        name, f1, rec, prec, cnt_err))

print("\nTop 5 (F1):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: -x[1])[:5], 1):
    print("{}. {:<26s} F1={:.4f} R={:.4f} P={:.4f} Err={:.1%}".format(i, n, f1, r, p, e))

print("\nTop 5 (计数误差):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: x[4])[:5], 1):
    print("{}. {:<26s} F1={:.4f} R={:.4f} P={:.4f} Err={:.1%}".format(i, n, f1, r, p, e))
