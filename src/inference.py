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
IOU_THRESHOLD = 0.5          # SAHI merge NMS 阈值
YOLO_IOU_THRESHOLD = 0.7     # YOLO 内部 NMS 阈值 (ultralytics 默认)
SAHI_TILE_SIZE = 640
SAHI_OVERLAP = 0.2
CLASS_ID = 0
IMGSZ = 640
MAX_DET = 800
DEDUP_IOU = 0.15             # 去重 IoU 阈值（0 表示禁用）
DEDUP_DIST = 30              # 去重中心距离阈值（像素）


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


def nms_masks(masks, confidences, iou_threshold=0.5, method="hard"):
    """基于 mask IoU 的 NMS（降采样加速版）。

    Args:
        method: "hard" (删除低分框) 或 "soft" (线性降分)
    """
    if len(masks) <= 1:
        return masks, confidences

    # 降采样用于 IoU 计算（保留原分辨率 mask 用于输出）
    h, w = masks[0].shape[:2]
    scale = min(480 / max(h, w), 1.0)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    small_masks = [cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_NEAREST) for m in masks]

    if method == "soft":
        return _soft_nms(masks, confidences, small_masks, iou_threshold)

    # Hard NMS
    order = np.argsort(confidences)[::-1]
    keep = []

    for i in range(len(order)):
        idx = order[i]
        suppressed = False
        for k in keep:
            intersection = np.bitwise_and(small_masks[idx], small_masks[k]).sum()
            union = np.bitwise_or(small_masks[idx], small_masks[k]).sum()
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)

    return [masks[i] for i in keep], [confidences[i] for i in keep]


def _soft_nms(masks, confidences, small_masks, iou_threshold):
    """Soft-NMS：线性降分而非直接删除。"""
    n = len(masks)
    scores = np.array(confidences, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(n, dtype=bool)

    for i_idx in range(n):
        i = order[i_idx]
        if suppressed[i]:
            continue
        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if suppressed[j]:
                continue
            intersection = np.bitwise_and(small_masks[i], small_masks[j]).sum()
            union = np.bitwise_or(small_masks[i], small_masks[j]).sum()
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                scores[j] *= (1.0 - iou)
                if scores[j] < CONF_THRESHOLD * 0.5:
                    suppressed[j] = True

    keep = [i for i in range(n) if not suppressed[i] and scores[i] >= CONF_THRESHOLD * 0.5]
    if not keep:
        keep = [int(np.argmax(confidences))]

    return [masks[i] for i in keep], [confidences[i] for i in keep]


def dedup_masks(masks, confidences, iou_threshold=0.15, distance_threshold=30):
    """基于中心距离 + IoU 的去重（二次过滤），处理 NMS 未能去除的碎片化重复。
    
    对于任意一对 mask，如果它们的中心距离 < distance_threshold 且
    mask IoU > iou_threshold，则抑制低置信度的那个。

    Args:
        iou_threshold: 较低的 IoU 阈值，用于捕获重叠碎片 (default: 0.15)
        distance_threshold: 中心像素距离阈值 (default: 30px)
    """
    if len(masks) <= 1:
        return masks, confidences

    n = len(masks)
    suppressed = [False] * n

    # 按置信度降序
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)

    # 降采样用于 IoU 计算
    h, w = masks[0].shape[:2]
    scale = min(480 / max(h, w), 1.0)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    small_masks = [cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_NEAREST) for m in masks]

    # 预计算所有 mask 的中心点（一次性，避免 O(n²) 重复计算）
    centers = [None] * n
    for i in range(n):
        ys, xs = np.where(masks[i] > 0)
        if len(ys) > 0:
            centers[i] = (ys.mean(), xs.mean())

    for i_idx, i in enumerate(order):
        if suppressed[i] or centers[i] is None:
            continue
        cy_i, cx_i = centers[i]

        for j in order[i_idx + 1:]:
            if suppressed[j] or centers[j] is None:
                continue
            cy_j, cx_j = centers[j]

            # 中心距离检查
            center_dist = np.sqrt((cy_i - cy_j) ** 2 + (cx_i - cx_j) ** 2)
            if center_dist > distance_threshold:
                continue

            # IoU 检查（降采样加速）
            intersection = np.bitwise_and(small_masks[i], small_masks[j]).sum()
            union = np.bitwise_or(small_masks[i], small_masks[j]).sum()
            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                suppressed[j] = True

    keep = [i for i in range(n) if not suppressed[i] and confidences[i] >= CONF_THRESHOLD * 0.5]
    if not keep:
        keep = [int(np.argmax(confidences))]

    return [masks[i] for i in keep], [confidences[i] for i in keep]


# ═══════════════════════════════════════════════════════
# 推理模式
# ═══════════════════════════════════════════════════════

def infer_baseline(model, image):
    """标准 YOLO 推理（整图 resize）。"""
    results = model(image, conf=CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, imgsz=IMGSZ, max_det=MAX_DET, verbose=False)
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


def _sahi_extract_masks(model, image):
    """SAHI tile + 投影（不做 NMS），返回 (all_masks, all_confs)。"""
    h, w = image.shape[:2]
    tile_size = SAHI_TILE_SIZE
    stride = int(tile_size * (1 - SAHI_OVERLAP))

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_tiles = len(y_steps) * len(x_steps)

    all_masks = []
    all_confs = []
    tile_idx = 0

    for y in y_steps:
        for x in x_steps:
            tile_idx += 1
            tile = image[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            # Pad 不完整的 tile
            if th < tile_size or tw < tile_size:
                pad_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_tile[:th, :tw] = tile
                tile = pad_tile

            results = model(tile, conf=CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, imgsz=tile_size, max_det=MAX_DET, verbose=False)

            n_det = len(results[0].masks.data) if results[0].masks is not None else 0
            print(f"    tile {tile_idx}/{total_tiles}: {n_det} detections", flush=True)

            if results[0].masks is not None:
                for mask_tensor, conf in zip(results[0].masks.data, results[0].boxes.conf):
                    mask = mask_tensor.cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

                    # 投影到原图坐标
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y:y + tile_size, x:x + tile_size] = mask[:th, :tw]
                    all_masks.append(full_mask)
                    all_confs.append(float(conf))

    return all_masks, all_confs


def _tta_consensus(orig_masks, orig_confs, flip_masks, flip_confs, iou_threshold=0.3):
    """共识 TTA：只保留原图和翻转图都检测到的预测（IoU > threshold）。
    
    降采样加速，保留置信度更高的版本。
    """
    if not orig_masks or not flip_masks:
        return orig_masks[:], orig_confs[:]

    h, w = orig_masks[0].shape[:2]
    scale = min(480 / max(h, w), 1.0)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))

    s_orig = [cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_NEAREST) for m in orig_masks]
    s_flip = [cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_NEAREST) for m in flip_masks]

    used_flip = set()
    result_masks = []
    result_confs = []

    for i, so in enumerate(s_orig):
        best_iou, best_j = 0, -1
        for j, sf in enumerate(s_flip):
            if j in used_flip:
                continue
            intersection = np.bitwise_and(so, sf).sum()
            union = np.bitwise_or(so, sf).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou and iou > iou_threshold:
                best_iou, best_j = iou, j

        if best_j >= 0:
            used_flip.add(best_j)
            # 取置信度更高的版本
            if orig_confs[i] >= flip_confs[best_j]:
                result_masks.append(orig_masks[i])
                result_confs.append(orig_confs[i])
            else:
                result_masks.append(flip_masks[best_j])
                result_confs.append(flip_confs[best_j])

    return result_masks, result_confs


def infer_sahi(model, image, tta=False, nms_method="hard", dedup_iou=0, dedup_dist=30):
    """SAHI 分块推理，可选 TTA（水平翻转 + 合并 NMS）+ 二次去重。

    Args:
        tta: 启用水平翻转 TTA
        nms_method: "hard" 或 "soft"
        dedup_iou: 去重 IoU 阈值（0 表示禁用）
        dedup_dist: 去重中心距离阈值（像素）
    """
    masks, confs = _sahi_extract_masks(model, image)

    if tta:
        flipped = cv2.flip(image, 1)
        f_masks, f_confs = _sahi_extract_masks(model, flipped)
        f_masks = [cv2.flip(m, 1) for m in f_masks]
        masks, confs = _tta_consensus(masks, confs, f_masks, f_confs, iou_threshold=0.3)
        print(f"    TTA consensus: {len(masks)} masks kept", flush=True)

    # NMS 去重
    if len(masks) > 1:
        print(f"    NMS ({nms_method}): {len(masks)} masks...", flush=True)
        masks, confs = nms_masks(masks, confs, iou_threshold=IOU_THRESHOLD, method=nms_method)
        print(f"    NMS done: {len(masks)} masks kept", flush=True)

    # 二次去重（中心距离 + IoU），捕获 NMS 未能去除的碎片化重复
    if dedup_iou > 0 and len(masks) > 1:
        print(f"    Dedup (iou>{dedup_iou}, dist<={dedup_dist}): {len(masks)} masks...", flush=True)
        masks, confs = dedup_masks(masks, confs, iou_threshold=dedup_iou, distance_threshold=dedup_dist)
        print(f"    Dedup done: {len(masks)} masks kept", flush=True)

    return masks, confs


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def main():
    global CONF_THRESHOLD, SAHI_TILE_SIZE, SAHI_OVERLAP, MAX_DET, YOLO_IOU_THRESHOLD, DEDUP_IOU, DEDUP_DIST

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
    parser.add_argument("--tta", action="store_true", help="启用 TTA（水平翻转 + 合并 NMS）")
    parser.add_argument("--soft-nms", action="store_true", help="使用 Soft-NMS 替代硬 NMS")
    parser.add_argument("--yolo-iou", type=float, default=YOLO_IOU_THRESHOLD,
                        help=f"YOLO 内部 NMS IoU 阈值 (default: {YOLO_IOU_THRESHOLD})")
    parser.add_argument("--dedup-iou", type=float, default=DEDUP_IOU,
                        help=f"去重 IoU 阈值（0 禁用，default: {DEDUP_IOU}）")
    parser.add_argument("--dedup-dist", type=float, default=DEDUP_DIST,
                        help=f"去重中心距离阈值，像素（default: {DEDUP_DIST}）")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    SAHI_TILE_SIZE = args.tile_size
    SAHI_OVERLAP = args.overlap
    MAX_DET = args.max_det
    YOLO_IOU_THRESHOLD = args.yolo_iou
    DEDUP_IOU = args.dedup_iou
    DEDUP_DIST = args.dedup_dist
    nms_method = "soft" if args.soft_nms else "hard"

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
    print(f"YOLO IoU: {YOLO_IOU_THRESHOLD}")
    print(f"TTA: {'启用' if args.tta else '关闭'}")
    print(f"NMS:  {'Soft' if nms_method == 'soft' else 'Hard'}")
    print(f"去重: {'关闭' if DEDUP_IOU <= 0 else f'IoU>{DEDUP_IOU}, Dist<={DEDUP_DIST}px'}")
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
            masks, confs = infer_sahi(model, image, tta=args.tta, nms_method=nms_method,
                                      dedup_iou=DEDUP_IOU, dedup_dist=DEDUP_DIST)

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

