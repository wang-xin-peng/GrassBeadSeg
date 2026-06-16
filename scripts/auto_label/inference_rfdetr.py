"""
运行 RF-DETR 推理，对原始图片进行自动标注。

使用方式：
  1. 训练完成后 models/rfdetr_seg_large/trained/checkpoint_best_total.pth 将存在
  2. conda activate gbseg && python scripts/auto_label/inference_rfdetr.py

输出：在 dataset/auto_labeled/ 下生成 YOLO 分割格式的标注文件。
"""

import os
import sys
import cv2
import numpy as np
import supervision as sv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "rfdetr_seg_large" / "trained" / "checkpoint_best_total.pth"
RAW_IMAGES_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_LABELS_DIR = PROJECT_ROOT / "dataset" / "auto_labeled" / "labels"
OUTPUT_IMAGES_DIR = PROJECT_ROOT / "dataset" / "auto_labeled" / "images"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "auto_labeled"
DATA_V2_DIR = PROJECT_ROOT / "dataset" / "data_v2"

DETECTION_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
CLASS_ID = 0


def get_already_labeled_set():
    labeled = set()
    for split in ["train", "valid", "test"]:
        label_dir = DATA_V2_DIR / split / "labels"
        if not label_dir.exists():
            continue
        for f in os.listdir(label_dir):
            if f.endswith(".txt"):
                parts = f.split("_")
                if parts and parts[0].isdigit():
                    labeled.add(parts[0])
                elif "_" in f:
                    number_part = f.split("_")[0]
                    if number_part.isdigit():
                        labeled.add(number_part)
    return labeled


def mask_to_yolo_polygon(mask, img_width, img_height, epsilon_factor=0.001):
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
            x_norm = pt[0] / img_width
            y_norm = pt[1] / img_height
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            coords.append(f"{x_norm:.6f} {y_norm:.6f}")
        polygons.append(f"{CLASS_ID} {' '.join(coords)}")
    return polygons


def main():
    if not CHECKPOINT_PATH.exists():
        alt_paths = [
            PROJECT_ROOT / "models" / "rfdetr_seg_large" / "trained" / "checkpoint_best_ema.pth",
            PROJECT_ROOT / "models" / "rfdetr_seg_large" / "trained" / "checkpoint_best_regular.pth",
        ]
        checkpoint_path = None
        for ap in alt_paths:
            if ap.exists():
                checkpoint_path = ap
                break
        if checkpoint_path is None:
            print(f"错误: 未找到训练好的检查点")
            print(f"请先运行 scripts/auto_label/train_rfdetr.py 完成训练")
            print(f"或者在以下位置放置检查点文件: {CHECKPOINT_PATH}")
            sys.exit(1)
    else:
        checkpoint_path = CHECKPOINT_PATH

    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    all_image_files = sorted([
        f for f in os.listdir(RAW_IMAGES_DIR)
        if Path(f).suffix.lower() in image_extensions
    ])

    if not all_image_files:
        print(f"错误: {RAW_IMAGES_DIR} 中没有找到图片文件")
        sys.exit(1)

    labeled_set = get_already_labeled_set()
    image_files = []
    for f in all_image_files:
        stem = Path(f).stem
        if stem in labeled_set:
            print(f"  跳过(已在 data_v2 中标注): {f}")
        else:
            image_files.append(f)

    print(f"找到 {len(image_files)} 张未标注图片 (已跳过 {len(all_image_files) - len(image_files)} 张)")

    try:
        from rfdetr import RFDETRSegLarge
        import torch
    except ImportError:
        print("错误: 请先安装 rfdetr: pip install rfdetr")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {device}")

    model = RFDETRSegLarge(pretrain_weights=str(checkpoint_path), num_queries=800, num_select=800, device=device)

    # 尝试优化推理（某些版本可能返回 None）
    try:
        optimized_model = model.optimize_for_inference()
        if optimized_model is not None:
            model = optimized_model
            print("推理优化已启用")
        else:
            print("警告: optimize_for_inference() 返回 None，继续使用原始模型")
    except Exception as e:
        print(f"警告: 推理优化失败 ({e})，继续使用原始模型")

    print("模型加载完成")
    print(f"检测阈值: {DETECTION_THRESHOLD}")
    print("=" * 60)

    # 检查模型内部结构和设备
    try:
        print(f"model 类型: {type(model)}")
        print(f"model 属性: {[a for a in dir(model) if not a.startswith('_')]}")
        for attr in ['model', 'net', 'module', 'backbone']:
            if hasattr(model, attr):
                sub = getattr(model, attr)
                params = list(sub.parameters()) if hasattr(sub, 'parameters') else []
                if params:
                    print(f"model.{attr} 设备: {params[0].device}")
                    break
    except Exception as e:
        print(f"设备检测失败: {e}")

    total_detections = 0
    for img_filename in image_files:
        img_path = RAW_IMAGES_DIR / img_filename
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  跳过(无法读取): {img_filename}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        detections = model.predict(image_rgb, threshold=DETECTION_THRESHOLD)
        detections = detections.with_nms(threshold=IOU_THRESHOLD)

        num_detections = len(detections)
        total_detections += num_detections

        yolo_lines = []
        if num_detections > 0 and detections.mask is not None:
            for mask in detections.mask:
                polygons = mask_to_yolo_polygon(mask, w, h)
                yolo_lines.extend(polygons)

        base_name = Path(img_filename).stem
        label_path = OUTPUT_LABELS_DIR / f"{base_name}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        cv2.imwrite(str(OUTPUT_IMAGES_DIR / img_filename), image)
        num_polygons = len(yolo_lines)
        print(f"  {img_filename}: {num_detections} objects, {num_polygons} polygons")

    data_yaml_content = f"""train: ../auto_labeled/images
val: ../auto_labeled/images
test: ../auto_labeled/images

nc: 1
names: ['GlassBeadSeg']
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml_content)

    print("=" * 60)
    print(f"自动标注完成！")
    print(f"总处理图片: {len(image_files)} 张")
    print(f"总检测目标: {total_detections}")
    print(f"标注文件保存至: {OUTPUT_LABELS_DIR}")
    print(f"配置文件: {yaml_path}")


if __name__ == "__main__":
    main()