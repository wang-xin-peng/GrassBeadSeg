"""
评估脚本：对比预测标注与 ground truth，计算召回率/计数误差。

使用方式：
    python src/eval.py \
        --gt dataset/data_v3_augmented/test/labels \
        --pred outputs/test_baseline/labels \
        --images dataset/data_v3_augmented/test/images
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_yolo_label(filepath):
    """解析 YOLO polygon 标注文件，返回 [(polygon_points, ...), ...]。
    每个 polygon 是 normalized 坐标的 N×2 数组。
    """
    polygons = []
    if not filepath.exists():
        return polygons

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:  # class_id + 至少 3 对坐标 = 7
                continue
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
            if len(coords) >= 3:
                polygons.append(coords)
    return polygons


def polygon_to_mask(polygon, h, w):
    """将归一化多边形转为 binary mask (H×W)。"""
    pts = (polygon * [w, h]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_mask_iou(mask_a, mask_b):
    """计算两个 binary mask 的 IoU。"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def evaluate_image(gt_polygons, pred_polygons, h, w, iou_threshold=0.5):
    """对单张图片计算匹配指标。

    Returns:
        dict with: gt_count, pred_count, tp, fp, fn, recall, precision, f1
    """
    gt_count = len(gt_polygons)
    pred_count = len(pred_polygons)

    if gt_count == 0 and pred_count == 0:
        return {
            "gt_count": 0, "pred_count": 0,
            "tp": 0, "fp": 0, "fn": 0,
            "recall": 1.0, "precision": 1.0, "f1": 1.0,
            "count_error": 0.0,
        }

    if gt_count == 0:
        return {
            "gt_count": 0, "pred_count": pred_count,
            "tp": 0, "fp": pred_count, "fn": 0,
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "count_error": 1.0,
        }

    if pred_count == 0:
        return {
            "gt_count": gt_count, "pred_count": 0,
            "tp": 0, "fp": 0, "fn": gt_count,
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "count_error": 1.0,
        }

    # 预计算所有 mask
    gt_masks = [polygon_to_mask(p, h, w) for p in gt_polygons]
    pred_masks = [polygon_to_mask(p, h, w) for p in pred_polygons]

    # 计算所有 GT-Pred IoU 对
    ious = []
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            iou = compute_mask_iou(gm, pm)
            if iou >= iou_threshold:
                ious.append((iou, i, j))

    # 贪心匹配（按 IoU 降序）
    ious.sort(key=lambda x: x[0], reverse=True)
    matched_gt = set()
    matched_pred = set()

    for iou, gi, pj in ious:
        if gi not in matched_gt and pj not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pj)

    tp = len(matched_gt)
    fn = gt_count - tp
    fp = pred_count - tp

    recall = tp / gt_count if gt_count > 0 else 0.0
    precision = tp / pred_count if pred_count > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    count_error = abs(pred_count - gt_count) / gt_count if gt_count > 0 else 0.0

    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "count_error": count_error,
    }


def main():
    parser = argparse.ArgumentParser(description="评估预测标注")
    parser.add_argument("--gt", required=True, help="Ground truth labels 目录")
    parser.add_argument("--pred", required=True, help="预测 labels 目录")
    parser.add_argument("--images", required=True, help="图片目录（用于读取尺寸）")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值 (default: 0.5)")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    img_dir = Path(args.images)

    if not gt_dir.exists():
        print(f"错误: GT 目录不存在: {gt_dir}")
        sys.exit(1)
    if not pred_dir.exists():
        print(f"错误: 预测目录不存在: {pred_dir}")
        sys.exit(1)

    gt_files = sorted([f for f in gt_dir.iterdir() if f.suffix == ".txt"])
    pred_files = sorted([f for f in pred_dir.iterdir() if f.suffix == ".txt"])

    # 按文件名匹配
    pred_map = {f.stem: f for f in pred_files}

    per_image = []
    total_tp = 0
    total_gt = 0
    total_pred = 0
    count_errors = []

    print(f"评估配置:")
    print(f"  GT 目录:     {gt_dir}")
    print(f"  预测目录:    {pred_dir}")
    print(f"  IoU 阈值:    {args.iou}")
    print(f"  GT 文件数:   {len(gt_files)}")
    print(f"  预测文件数:  {len(pred_files)}")
    print("=" * 70)

    for gt_file in gt_files:
        stem = gt_file.stem
        if stem not in pred_map:
            print(f"  [{stem}] 警告: 无对应预测文件，跳过")
            continue

        gt_polygons = parse_yolo_label(gt_file)
        pred_file = pred_map[stem]
        pred_polygons = parse_yolo_label(pred_file)

        # 读取图片尺寸
        img_path = img_dir / (stem + ".png")
        if not img_path.exists():
            img_path = img_dir / (stem + ".jpg")
        if img_path.exists():
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
        else:
            print(f"  [{stem}] 警告: 找不到图片，使用默认尺寸 640×640")
            h, w = 640, 640

        result = evaluate_image(gt_polygons, pred_polygons, h, w, iou_threshold=args.iou)
        per_image.append((stem, result))

        total_tp += result["tp"]
        total_gt += result["gt_count"]
        total_pred += result["pred_count"]
        count_errors.append(result["count_error"])

        print(f"  {stem}: GT={result['gt_count']:3d}  Pred={result['pred_count']:3d}  "
              f"TP={result['tp']:3d}  R={result['recall']:.3f}  P={result['precision']:.3f}  "
              f"Err={result['count_error']:.3f}")

    # ── 汇总 ──
    n = len(per_image)
    if n == 0:
        print("\n无可用数据，退出")
        sys.exit(1)

    total_fn = total_gt - total_tp
    total_fp = total_pred - total_tp

    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_f1 = (2 * overall_recall * overall_precision / (overall_recall + overall_precision)
                  if (overall_recall + overall_precision) > 0 else 0.0)
    avg_count_error = np.mean(count_errors) if count_errors else 0.0

    print("=" * 70)
    print(f"\n{'─' * 40}")
    print(f"  汇总统计 ({n} 张图片)")
    print(f"{'─' * 40}")
    print(f"  总 GT 数量:      {total_gt}")
    print(f"  总预测数量:      {total_pred}")
    print(f"  TP / FP / FN:    {total_tp} / {total_fp} / {total_fn}")
    print(f"{'─' * 40}")
    print(f"  Recall@{args.iou}:     {overall_recall:.4f}")
    print(f"  Precision@{args.iou}:  {overall_precision:.4f}")
    print(f"  F1@{args.iou}:          {overall_f1:.4f}")
    print(f"  平均计数误差:     {avg_count_error:.4f}")
    print(f"{'─' * 40}")

    # 每图计数对比
    print(f"\n  每图计数对比:")
    print(f"  {'图片':<30} {'GT':>5} {'Pred':>5} {'误差':>8}")
    for stem, r in per_image:
        diff = r["pred_count"] - r["gt_count"]
        sign = "+" if diff > 0 else ""
        print(f"  {stem:<30} {r['gt_count']:>5} {r['pred_count']:>5} {sign}{diff:>7}")


if __name__ == "__main__":
    main()
