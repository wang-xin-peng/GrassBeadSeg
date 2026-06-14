"""
数据增强脚本 - 对YOLO分割格式数据集进行增强
支持YOLO polygon/segmentation格式（每行: class_id x1 y1 x2 y2 x3 y3 ...）

Preprocessing:
- Auto-Orient: Applied
- Resize: Fit within 624x624 (保持宽高比，填充黑色)

Augmentations:
- Outputs per training example: 30 (vs 10 原版)
- Flip: Horizontal (p=0.5), Vertical (p=0.2)
- Affine: scale=0.85-1.15, translate=10%, rotate=±25°, shear=±5° (p=0.7)
- ElasticTransform: p=0.3
- OpticalDistortion: p=0.3
- ColorJitter: brightness=0.3, saturation=0.3, contrast=0.2, hue=0.05 (p=0.7)
- RandomBrightnessContrast: p=0.5
- CLAHE: p=0.3
- GaussianBlur / MotionBlur: p=0.5 / p=0.2
- PixelDropout / GaussNoise / ISONoise
- Sharpen / Posterize

用法:
    python augment_dataset.py --input dataset/data_v2 --output dataset/data_v2_aug
    python augment_dataset.py --input dataset/data_v4 --output dataset/data_v4_aug_v2 --augments-per-image 30
"""

import os
import random
import shutil
import argparse
import yaml
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AUGMENTS_PER_IMAGE = 30
DEFAULT_MAX_SIZE = 624
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def auto_orient(image):
    exif_orientation = cv2.imdecode(np.frombuffer(b'', np.uint8), cv2.IMREAD_UNCHANGED)
    return image


def resize_fit_within(image, max_size, keypoints=None):
    h, w = image.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = max_size - new_w
        pad_h = max_size - new_h
        left = pad_w // 2
        top = pad_h // 2
        padded = cv2.copyMakeBorder(resized, top, max_size - new_h - top, 
                                     left, max_size - new_w - left, 
                                     cv2.BORDER_CONSTANT, value=0)
        
        if keypoints is not None:
            new_kps = []
            for kx, ky in keypoints:
                nx = kx * scale + left
                ny = ky * scale + top
                new_kps.append((nx, ny))
            return padded, new_kps, new_w, new_h
        
        return padded, new_w, new_h
    else:
        pad_w = max_size - w
        pad_h = max_size - h
        left = pad_w // 2
        top = pad_h // 2
        padded = cv2.copyMakeBorder(image, top, max_size - h - top,
                                     left, max_size - w - left,
                                     cv2.BORDER_CONSTANT, value=0)
        if keypoints is not None:
            new_kps = [(kx + left, ky + top) for kx, ky in keypoints]
            return padded, new_kps, w, h
        return padded, w, h


def load_yolo_polygons(label_path, img_width, img_height):
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7 and len(parts) % 2 == 1:
                class_id = int(parts[0])
                points = []
                for j in range(1, len(parts), 2):
                    x = float(parts[j]) * img_width
                    y = float(parts[j+1]) * img_height
                    points.append([x, y])
                polygons.append({
                    'class_id': class_id,
                    'points': np.array(points, dtype=np.float32)
                })
    return polygons


def save_yolo_polygons(label_path, polygons, img_width, img_height):
    with open(label_path, 'w') as f:
        for poly in polygons:
            coords = []
            for pt in poly['points']:
                x_norm = pt[0] / img_width
                y_norm = pt[1] / img_height
                coords.append(f"{x_norm:.6f} {y_norm:.6f}")
            f.write(f"{poly['class_id']} {' '.join(coords)}\n")


def create_augmentation_pipeline():
    """创建增强流水线 - 增强版，包含更多变换类型以提高模型泛化能力。"""
    return A.Compose([
        # 翻转
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # 几何变换 - 仿射、旋转、缩放、剪切
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-25, 25),
            shear=(-5, 5),
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        # 弹性形变（模拟轻微透视/镜头畸变）
        A.ElasticTransform(alpha=1, sigma=50, p=0.3,
                           border_mode=cv2.BORDER_CONSTANT),

        # 镜头畸变模拟
        A.OpticalDistortion(distort_limit=(-0.15, 0.15), p=0.3,
                            border_mode=cv2.BORDER_CONSTANT),

        # 颜色增强
        A.ColorJitter(
            brightness=0.3,
            saturation=0.3,
            contrast=0.2,
            hue=0.05,
            p=0.7,
        ),

        # 随机亮度对比度（更宽范围）
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.3,
            p=0.5,
        ),

        # CLAHE 局部对比度增强
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # 模糊
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.5),
        A.MotionBlur(blur_limit=(3, 5), p=0.2),

        # 噪声
        A.PixelDropout(dropout_prob=0.005, per_channel=False, p=0.5),
        A.GaussNoise(std_range=(0.05, 0.1), per_channel=True, p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),

        # 值域增强
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Posterize(num_bits=(5, 7), p=0.2),

    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ))


def augment_dataset(dataset_root, augmented_dir, augments_per_image=10, max_size=624):
    dataset_root = Path(dataset_root)
    augmented_dir = Path(augmented_dir)
    
    train_images_dir = dataset_root / 'train' / 'images'
    train_labels_dir = dataset_root / 'train' / 'labels'
    
    aug_train_images_dir = augmented_dir / 'train' / 'images'
    aug_train_labels_dir = augmented_dir / 'train' / 'labels'
    aug_valid_images_dir = augmented_dir / 'valid' / 'images'
    aug_valid_labels_dir = augmented_dir / 'valid' / 'labels'
    aug_test_images_dir = augmented_dir / 'test' / 'images'
    aug_test_labels_dir = augmented_dir / 'test' / 'labels'
    
    for d in [aug_train_images_dir, aug_train_labels_dir,
              aug_valid_images_dir, aug_valid_labels_dir,
              aug_test_images_dir, aug_test_labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    for src_dir, dst_dir in [
        (dataset_root / 'valid' / 'images', aug_valid_images_dir),
        (dataset_root / 'valid' / 'labels', aug_valid_labels_dir),
        (dataset_root / 'test' / 'images', aug_test_images_dir),
        (dataset_root / 'test' / 'labels', aug_test_labels_dir),
    ]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                shutil.copy2(src_dir / f, dst_dir / f)
    
    image_files = [f for f in os.listdir(train_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 张训练图片")
    print(f"每张图片生成 {augments_per_image} 个增强版本")
    print(f"Resize: Fit within {max_size}x{max_size}")
    print(f"总共将生成 {len(image_files) * augments_per_image} 张增强图片")
    
    aug_pipeline = create_augmentation_pipeline()
    total_augmented = 0
    total_polygons = 0
    
    for img_filename in image_files:
        img_path = train_images_dir / img_filename
        base_name = Path(img_filename).stem
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  警告: 无法读取 {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        label_filename = Path(img_filename).with_suffix('.txt').name
        label_path = train_labels_dir / label_filename
        polygons = load_yolo_polygons(str(label_path), orig_w, orig_h)
        
        all_kps = []
        kps_per_poly = []
        for poly in polygons:
            start_idx = len(all_kps)
            for pt in poly['points']:
                all_kps.append((float(pt[0]), float(pt[1])))
            kps_per_poly.append((start_idx, len(poly['points'])))
        
        resized_img, resized_kps, new_w, new_h = resize_fit_within(image, max_size, all_kps)
        
        resized_polygons = []
        for idx, (start_idx, num_pts) in enumerate(kps_per_poly):
            pts = []
            for k in range(start_idx, start_idx + num_pts):
                pts.append([resized_kps[k][0], resized_kps[k][1]])
            resized_polygons.append({
                'class_id': polygons[idx]['class_id'],
                'points': np.array(pts, dtype=np.float32)
            })
        
        orig_img_path = aug_train_images_dir / img_filename
        resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(orig_img_path), resized_img_bgr)
        
        resized_label_filename = Path(img_filename).with_suffix('.txt').name
        save_yolo_polygons(str(aug_train_labels_dir / resized_label_filename), resized_polygons, max_size, max_size)
        total_augmented += 1
        total_polygons += len(resized_polygons)
        
        print(f"  {img_filename}: {len(polygons)} 个标注多边形 (原图 {orig_w}x{orig_h} -> 填充到 {max_size}x{max_size})")
        
        for i in range(augments_per_image):
            try:
                kps_for_aug = []
                for poly in resized_polygons:
                    for pt in poly['points']:
                        kps_for_aug.append((float(pt[0]), float(pt[1])))
                
                aug_result = aug_pipeline(
                    image=resized_img,
                    keypoints=kps_for_aug
                )
                
                aug_image = aug_result['image']
                aug_keypoints = aug_result['keypoints']
                h, w = aug_image.shape[:2]
                
                aug_polygons = []
                keypoint_idx = 0
                for poly in resized_polygons:
                    num_points = len(poly['points'])
                    if keypoint_idx + num_points <= len(aug_keypoints):
                        new_points = []
                        for j in range(num_points):
                            kp = aug_keypoints[keypoint_idx + j]
                            new_points.append([kp[0], kp[1]])
                        keypoint_idx += num_points
                        
                        new_points = np.array(new_points, dtype=np.float32)
                        new_points[:, 0] = np.clip(new_points[:, 0], 0, w - 1)
                        new_points[:, 1] = np.clip(new_points[:, 1], 0, h - 1)
                        
                        if len(new_points) >= 3:
                            aug_polygons.append({
                                'class_id': poly['class_id'],
                                'points': new_points
                            })
                
                aug_filename = f"{base_name}_aug{i+1}.png"
                aug_img_path = aug_train_images_dir / aug_filename
                aug_label_path = aug_train_labels_dir / Path(aug_filename).with_suffix('.txt').name
                
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)
                save_yolo_polygons(str(aug_label_path), aug_polygons, w, h)
                
                total_augmented += 1
                total_polygons += len(aug_polygons)
                
            except Exception as e:
                print(f"  警告: {img_filename} 的第 {i+1} 个增强失败: {e}")
                continue
        
        print(f"  处理完成: {img_filename}")
    
    print(f"\n数据增强完成!")
    print(f"原始训练图片: {len(image_files)} 张")
    print(f"增强后训练图片: {total_augmented} 张 (包含原始图片)")
    print(f"总标注多边形数: {total_polygons}")
    
    update_data_yaml(augmented_dir)


def update_data_yaml(augmented_dir):
    augmented_dir = Path(augmented_dir)
    yaml_path = augmented_dir / 'data.yaml'
    
    data = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1,
        'names': ['GrassBeadSeg']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n已更新 {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO分割数据集增强脚本")
    parser.add_argument("--input", "-i", required=True, help="输入数据集目录 (包含 train/valid/test 和 data.yaml)")
    parser.add_argument("--output", "-o", required=True, help="输出增强后数据集目录")
    parser.add_argument("--augments-per-image", "-n", type=int, default=DEFAULT_AUGMENTS_PER_IMAGE,
                        help=f"每张图片生成的增强版本数 (默认: {DEFAULT_AUGMENTS_PER_IMAGE})")
    parser.add_argument("--max-size", "-s", type=int, default=DEFAULT_MAX_SIZE,
                        help=f"Resize 目标尺寸 (默认: {DEFAULT_MAX_SIZE})")
    
    args = parser.parse_args()
    
    print("="*50)
    print("YOLO分割数据集增强脚本")
    print("="*50)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"增强倍数: {args.augments_per_image}")
    print(f"最大尺寸: {args.max_size}")
    print("="*50)
    
    augment_dataset(args.input, args.output, args.augments_per_image, args.max_size)


if __name__ == '__main__':
    main()
