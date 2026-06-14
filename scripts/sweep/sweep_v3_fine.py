"""v3 last.pt 精细调参网格搜索"""
import subprocess
import sys
from pathlib import Path

MODEL = "models/our_method/yolo11n-seg-v3/weights/last.pt"
SOURCE = "dataset/data_v5/test/images"
OUTPUT_BASE = "outputs/v3_sweep_fine"
GT = "dataset/data_v5/test/labels"
IMAGES = "dataset/data_v5/test/images"

# ── 全组合 ──
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

# ── 推理 ──
total = len(configs)
print("共 {} 组实验\n".format(total))
for i, (name, extra_args) in enumerate(configs, 1):
    out_dir = Path(OUTPUT_BASE) / name
    if (out_dir / "labels").exists():
        print("[{}/{}] SKIP {}".format(i, total, name))
        continue
    cmd = [
        sys.executable, "src/inference.py",
        "--model", MODEL,
        "--source", SOURCE,
        "--output", str(out_dir),
        "--mode", "ellipse",
    ] + extra_args
    print("[{}/{}] RUN {}".format(i, total, name))
    subprocess.run(cmd, check=True)

# ── 评估 ──
print("\n" + "=" * 85)
print("{:<28s} {:>6s} {:>6s} {:>6s} {:>7s}  {:>20s}".format(
    "Config", "F1", "Recall", "Prec", "CntErr", "TP/FP/FN"))
print("-" * 85)

results = []
for name, _ in configs:
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
    lines = proc.stdout.splitlines()
    f1 = rec = prec = cnt_err = 0.0
    tp_str = ""
    for line in lines:
        if "F1@0.5:" in line:
            f1 = float(line.split()[-1])
        elif "Recall@0.5:" in line:
            rec = float(line.split()[-1])
        elif "Precision@0.5:" in line:
            prec = float(line.split()[-1])
        elif "平均计数误差:" in line:
            cnt_err = float(line.split()[-1])
        elif "TP / FP / FN:" in line:
            tp_str = line.strip().split(":")[-1].strip()

    results.append((name, f1, rec, prec, cnt_err, tp_str))
    print("{:<28s} {:6.4f} {:6.4f} {:6.4f} {:6.1%}  {:>20s}".format(
        name, f1, rec, prec, cnt_err, tp_str))

# Top 5
print("\n" + "=" * 85)
print("Top 5 (按 F1):")
print("-" * 85)
for i, (name, f1, rec, prec, cnt_err, tp_str) in enumerate(
    sorted(results, key=lambda x: -x[1])[:5], 1
):
    print("{}. {:<26s} F1={:.4f}  Rec={:.4f}  Prec={:.4f}  CntErr={:.1%}".format(
        i, name, f1, rec, prec, cnt_err))

print("\nTop 5 (按计数误差):")
print("-" * 85)
for i, (name, f1, rec, prec, cnt_err, tp_str) in enumerate(
    sorted(results, key=lambda x: x[4])[:5], 1
):
    print("{}. {:<26s} F1={:.4f}  Rec={:.4f}  Prec={:.4f}  CntErr={:.1%}".format(
        i, name, f1, rec, prec, cnt_err))
