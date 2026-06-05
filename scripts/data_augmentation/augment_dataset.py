"""
数据增强脚本 - 对YOLO分割格式数据集进行增强
支持YOLO polygon/segmentation格式（每行: class_id x1 y1 x2 y2 x3 y3 ...）
增强配置（与Roboflow一致）：

Preprocessing:
- Auto-Orient: Applied
- Resize: Fit within 624x624 (保持宽高比，填充黑色)

Augmentations:
- Outputs per training example: 10
- Flip: Horizontal
- Rotation: Between -15 and +15
- Saturation: Between -25% and +25%
- Brightness: Between -25% and +25%
- Blur: Up to 0.5px
- Noise: Up to 0.34% of pixels
"""

import os
import random
import shutil
import yaml
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "data_v3"
AUGMENTED_DIR = PROJECT_ROOT / "dataset" / "data_v3_augmented"
AUGMENTS_PER_IMAGE = 10
MAX_SIZE = 624
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
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(
            brightness=0.25,
            saturation=0.25,
            contrast=0.0,
            hue=0.0,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=1.0),
        A.PixelDropout(dropout_prob=0.0034, per_channel=False, p=1.0),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ))


def augment_dataset():
    train_images_dir = DATASET_ROOT / 'train' / 'images'
    train_labels_dir = DATASET_ROOT / 'train' / 'labels'
    
    aug_train_images_dir = AUGMENTED_DIR / 'train' / 'images'
    aug_train_labels_dir = AUGMENTED_DIR / 'train' / 'labels'
    aug_valid_images_dir = AUGMENTED_DIR / 'valid' / 'images'
    aug_valid_labels_dir = AUGMENTED_DIR / 'valid' / 'labels'
    aug_test_images_dir = AUGMENTED_DIR / 'test' / 'images'
    aug_test_labels_dir = AUGMENTED_DIR / 'test' / 'labels'
    
    for d in [aug_train_images_dir, aug_train_labels_dir,
              aug_valid_images_dir, aug_valid_labels_dir,
              aug_test_images_dir, aug_test_labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    for src_dir, dst_dir in [
        (DATASET_ROOT / 'valid' / 'images', aug_valid_images_dir),
        (DATASET_ROOT / 'valid' / 'labels', aug_valid_labels_dir),
        (DATASET_ROOT / 'test' / 'images', aug_test_images_dir),
        (DATASET_ROOT / 'test' / 'labels', aug_test_labels_dir),
    ]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                shutil.copy2(src_dir / f, dst_dir / f)
    
    image_files = [f for f in os.listdir(train_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 张训练图片")
    print(f"每张图片生成 {AUGMENTS_PER_IMAGE} 个增强版本")
    print(f"Resize: Fit within {MAX_SIZE}x{MAX_SIZE}")
    print(f"总共将生成 {len(image_files) * AUGMENTS_PER_IMAGE} 张增强图片")
    
    aug_pipeline = create_augmentation_pipeline()
    total_augmented = 0
    total_polygons = 0
    
    for img_filename in image_files:
        img_path = train_images_dir / img_filename
        base_name = Path(img_filename).stem
        
        image = cv2.imread(str(img_path))
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
        
        resized_img, resized_kps, new_w, new_h = resize_fit_within(image, MAX_SIZE, all_kps)
        
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
        save_yolo_polygons(str(aug_train_labels_dir / resized_label_filename), resized_polygons, MAX_SIZE, MAX_SIZE)
        total_augmented += 1
        total_polygons += len(resized_polygons)
        
        print(f"  {img_filename}: {len(polygons)} 个标注多边形 (原图 {orig_w}x{orig_h} -> 填充到 {MAX_SIZE}x{MAX_SIZE})")
        
        for i in range(AUGMENTS_PER_IMAGE):
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
    
    update_data_yaml()


def update_data_yaml():
    yaml_path = AUGMENTED_DIR / 'data.yaml'
    
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


if __name__ == '__main__':
    print("="*50)
    print("YOLO分割数据集增强脚本")
    print("="*50)
    augment_dataset()
