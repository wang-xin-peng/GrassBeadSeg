"""v4 best.pt 置信度 × overlap 网格搜索"""
import subprocess, sys
from pathlib import Path

MODEL = "models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt"
SOURCE = "dataset/data_v4_aug/test/images"
OUTPUT_BASE = "outputs/v4_sweep"
GT = "dataset/data_v4_aug/test/labels"
IMAGES = "dataset/data_v4_aug/test/images"

CONF_RANGE = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
OV_RANGE = [0.20, 0.25, 0.30]

total = len(CONF_RANGE) * len(OV_RANGE)
count = 0
for ov in OV_RANGE:
    for c in CONF_RANGE:
        count += 1
        name = f"ov{int(ov*100):02d}_c{int(c*100):02d}"
        out_dir = Path(OUTPUT_BASE) / name
        if (out_dir / "labels").exists():
            print(f"[{count}/{total}] SKIP {name}")
            continue
        cmd = [
            sys.executable, "src/inference.py",
            "--model", MODEL, "--source", SOURCE,
            "--output", str(out_dir), "--mode", "ellipse",
            "--conf", str(c), "--overlap", str(ov),
        ]
        print(f"[{count}/{total}] RUN {name}")
        subprocess.run(cmd, check=True)

# ── 评估 ──
print("\n" + "=" * 80)
print(f"{'Config':<24s} {'F1':>6s} {'Recall':>6s} {'Prec':>6s} {'CntErr':>7s}")
print("-" * 80)

results = []
for ov in OV_RANGE:
    for c in CONF_RANGE:
        name = f"ov{int(ov*100):02d}_c{int(c*100):02d}"
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
        print(f"{name:<24s} {f1:6.4f} {rec:6.4f} {prec:6.4f} {cnt_err:6.1%}")

print("\nTop 5 (F1):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: -x[1])[:5], 1):
    print(f"{i}. {n:<22s} F1={f1:.4f} R={r:.4f} P={p:.4f} Err={e:.1%}")

print("\nTop 5 (计数误差):")
for i, (n, f1, r, p, e) in enumerate(sorted(results, key=lambda x: x[4])[:5], 1):
    print(f"{i}. {n:<22s} F1={f1:.4f} R={r:.4f} P={p:.4f} Err={e:.1%}")
