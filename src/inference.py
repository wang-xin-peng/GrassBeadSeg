"""
YOLOv11-seg 推理脚本（our_method）。

支持三种推理模式：
  - baseline: 标准 YOLO 推理（Phase 1）
  - sahi:     SAHI 分块推理（Phase 2）
  - ellipse:  SAHI 分块 + 椭圆拟合（Phase 2）

输出 YOLO polygon 分割格式 (.txt)，兼容现有 pipeline。

使用方式：
    # Phase 1: 纯 YOLO 分割（无 crop，无拟合）
    python src/inference.py \
        --model models/our_method/yolo11s-seg-baseline/weights/best.pt \
        --source dataset/data_v3_augmented/test/images \
        --output outputs/test_baseline \
        --mode baseline

    # Phase 2: SAHI + 椭圆拟合
    python src/inference.py \
        --model models/our_method/yolo11s-seg-baseline/weights/best.pt \
        --source dataset/data_v3_augmented/test/images \
        --output outputs/test_ellipse \
        --mode ellipse
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 推理参数 ──────────────────────────────────────────
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SAHI_TILE_SIZE = 640
SAHI_OVERLAP = 0.2
CLASS_ID = 0
IMGSZ = 640
MAX_DET = 800


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def mask_to_yolo_polygon(mask, img_w, img_h, epsilon_factor=0.001):
    """将 binary mask 转为 YOLO polygon 格式字符串列表。"""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < 3:
            continue
        coords = []
        for pt in approx[:, 0, :]:
            x_norm = max(0.0, min(1.0, pt[0] / img_w))
            y_norm = max(0.0, min(1.0, pt[1] / img_h))
            coords.append(f"{x_norm:.6f} {y_norm:.6f}")
        polygons.append(f"{CLASS_ID} {' '.join(coords)}")
    return polygons


def mask_to_ellipse(mask):
    """凸包 + fitEllipse：将 mask 拟合为椭圆 mask。参数自由。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask

    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 5:
        return mask

    hull = cv2.convexHull(cnt)
    if hull.shape[0] < 5:
        return mask

    try:
        ellipse = cv2.fitEllipse(hull)
    except cv2.error:
        return mask

    result = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(result, ellipse, 255, -1)
    return result


def nms_masks(masks, confidences, iou_threshold=0.5):
    """基于 mask IoU 的 NMS。"""
    if len(masks) <= 1:
        return masks, confidences

    order = np.argsort(confidences)[::-1]
    keep = []

    for i in range(len(order)):
        idx = order[i]
        suppressed = False
        for k in keep:
            intersection = np.logical_and(masks[idx], masks[k]).sum()
            union = np.logical_or(masks[idx], masks[k]).sum()
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)

    return [masks[i] for i in keep], [confidences[i] for i in keep]


# ═══════════════════════════════════════════════════════
# 推理模式
# ═══════════════════════════════════════════════════════

def infer_baseline(model, image):
    """标准 YOLO 推理（整图 resize）。"""
    results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)
    h, w = image.shape[:2]

    masks = []
    confs = []
    if results[0].masks is not None:
        for mask_tensor, conf in zip(results[0].masks.data, results[0].boxes.conf):
            mask = mask_tensor.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
            confs.append(float(conf))

    return masks, confs


def infer_sahi(model, image):
    """SAHI 分块推理。"""
    h, w = image.shape[:2]
    tile_size = SAHI_TILE_SIZE
    stride = int(tile_size * (1 - SAHI_OVERLAP))

    all_masks = []
    all_confs = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            # Pad 不完整的 tile
            if th < tile_size or tw < tile_size:
                pad_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_tile[:th, :tw] = tile
                tile = pad_tile

            results = model(tile, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=tile_size, max_det=MAX_DET, verbose=False)

            if results[0].masks is not None:
                for mask_tensor, conf in zip(results[0].masks.data, results[0].boxes.conf):
                    mask = mask_tensor.cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

                    # 投影到原图坐标
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y:y + tile_size, x:x + tile_size] = mask[:th, :tw]
                    all_masks.append(full_mask)
                    all_confs.append(float(conf))

    # NMS 去重
    if len(all_masks) > 1:
        all_masks, all_confs = nms_masks(all_masks, all_confs, iou_threshold=IOU_THRESHOLD)

    return all_masks, all_confs


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def main():
    global CONF_THRESHOLD, SAHI_TILE_SIZE, SAHI_OVERLAP, MAX_DET

    parser = argparse.ArgumentParser(description="YOLOv11-seg 推理")
    parser.add_argument("--model", required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--source", required=True, help="输入图片目录或单张图片路径")
    parser.add_argument("--output", default=None, help="输出目录 (默认: 与 source 同级的 predictions/)")
    parser.add_argument("--mode", choices=["baseline", "sahi", "ellipse"], default="baseline",
                        help="推理模式 (default: baseline)")
    parser.add_argument("--device", default=0, help="设备 (0=cuda:0, cpu=cpu)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    parser.add_argument("--tile-size", type=int, default=SAHI_TILE_SIZE, help="SAHI tile size")
    parser.add_argument("--overlap", type=float, default=SAHI_OVERLAP, help="SAHI overlap")
    parser.add_argument("--max-det", type=int, default=MAX_DET, help="最大检测数 (default: 800)")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    SAHI_TILE_SIZE = args.tile_size
    SAHI_OVERLAP = args.overlap
    MAX_DET = args.max_det

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)

    source = Path(args.source)
    if source.is_dir():
        image_files = sorted([f for f in source.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
    else:
        image_files = [source]

    if not image_files:
        print(f"错误: 没有找到图片文件: {source}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = source.parent / "predictions"
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visualizations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Device 处理
    device = args.device
    if device != "cpu" and str(device).isdigit():
        import torch
        if not torch.cuda.is_available():
            print("警告: CUDA 不可用，使用 CPU")
            device = "cpu"

    model = YOLO(str(model_path))

    print(f"模型: {model_path.name}")
    print(f"模式: {args.mode}")
    print(f"置信度: {CONF_THRESHOLD}")
    print(f"设备: {device}")
    print(f"图片数: {len(image_files)}")
    print("=" * 60)

    total_detections = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  跳过: {img_path.name}")
            continue

        h, w = image.shape[:2]

        # 推理
        if args.mode == "baseline":
            masks, confs = infer_baseline(model, image)
        else:  # sahi / ellipse
            masks, confs = infer_sahi(model, image)

        # 椭圆拟合（仅在 ellipse 模式下）
        if args.mode == "ellipse":
            masks = [mask_to_ellipse(m) for m in masks]

        # 生成 YOLO polygon 标注
        yolo_lines = []
        for mask in masks:
            polygons = mask_to_yolo_polygon(mask, w, h)
            yolo_lines.extend(polygons)

        num_detections = len(yolo_lines)
        total_detections += num_detections

        # 保存
        label_path = labels_dir / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # 可视化（画 mask 轮廓在图上）
        vis_image = image.copy()
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(vis_dir / img_path.name), vis_image)

        print(f"  {img_path.name}: {num_detections} masks")

    print("=" * 60)
    print(f"推理完成！")
    print(f"总预测数: {total_detections}")
    print(f"标注文件: {labels_dir}")
    print(f"可视化: {vis_dir}")


if __name__ == "__main__":
    main()
    main()
