"""
生成 data_v4 数据集：合并人工标注 + RF-DETR 自动标注，重新划分，离线增强。

结构：
    dataset/data_v4/
    ├── train/images/     # 28 张原始 + 增强后 ~150 张
    ├── train/labels/
    ├── valid/images/     # 6 张（不做增强）
    ├── valid/labels/
    ├── test/images/      # 6 张（不做增强）
    ├── test/labels/
    └── data.yaml

使用方式：
    python scripts/data_preparation/prepare_data_v4.py
"""

import os
import sys
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "data_v4"

# 源数据
DATA_V3_DIR = PROJECT_ROOT / "dataset" / "data_v3"
AUTO_LABELED_DIR = PROJECT_ROOT / "dataset" / "auto_labeled_v3"

# 划分比例
VALID_SIZE = 6
TEST_SIZE = 6
TRAIN_SIZE = 40 - VALID_SIZE - TEST_SIZE  # 28

# 增强后 train 目标数量
TARGET_TRAIN_AUGMENTED = 150

# 固定随机种子保证可复现
random.seed(42)
np.random.seed(42)

# ── 数据增强策略 ──────────────────────────────────────
# 只对 train 做增强，valid/test 保持原样
AUGMENTATION = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
], bbox_params=None, keypoint_params=None)


def collect_all_images():
    """收集所有 40 张有标注的图片路径。"""
    all_images = []

    # data_v3: 01-20（人工标注）
    for split in ["train", "valid", "test"]:
        img_dir = DATA_V3_DIR / split / "images"
        label_dir = DATA_V3_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                all_images.append((str(img_path), str(label_path), "manual"))

    # auto_labeled_v3: 21-40（自动标注）
    img_dir = AUTO_LABELED_DIR / "images"
    label_dir = AUTO_LABELED_DIR / "labels"
    for img_path in sorted(img_dir.glob("*.png")):
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            all_images.append((str(img_path), str(label_path), "auto"))

    return all_images


def split_dataset(all_images):
    """划分 train/valid/test，确保人工标注和自动标注均匀分布。"""
    manual = [x for x in all_images if x[2] == "manual"]
    auto = [x for x in all_images if x[2] == "auto"]

    random.shuffle(manual)
    random.shuffle(auto)

    # valid/test 各取一半人工一半自动
    valid_manual = manual[:VALID_SIZE // 2]
    valid_auto = auto[:VALID_SIZE // 2]
    valid = valid_manual + valid_auto

    test_manual = manual[VALID_SIZE // 2:VALID_SIZE // 2 + TEST_SIZE // 2]
    test_auto = auto[VALID_SIZE // 2:VALID_SIZE // 2 + TEST_SIZE // 2]
    test = test_manual + test_auto

    # 剩余全部给 train
    train_manual = [x for x in manual if x not in valid and x not in test]
    train_auto = [x for x in auto if x not in valid and x not in test]
    train = train_manual + train_auto

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    return train, valid, test


def copy_files(file_list, dest_img_dir, dest_label_dir):
    """复制图片和标注到目标目录。"""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path, _ in file_list:
        shutil.copy2(img_path, dest_img_dir / Path(img_path).name)
        shutil.copy2(label_path, dest_label_dir / Path(label_path).name)


def augment_train(train_images, dest_img_dir, dest_label_dir):
    """对 train 集做离线增强。"""
    n_original = len(train_images)
    n_needed = max(0, TARGET_TRAIN_AUGMENTED - n_original)

    if n_needed <= 0:
        print(f"  train 原始数量 {n_original} 已 >= 目标 {TARGET_TRAIN_AUGMENTED}，无需增强")
        return

    print(f"  train 原始: {n_original} 张，需要增强到 {TARGET_TRAIN_AUGMENTED} 张")

    # 计算每个原始图需要生成几张增强图
    n_per_image = n_needed // n_original + 1
    aug_count = 0

    for img_path, label_path, src_type in train_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 读取标注
        with open(label_path, "r") as f:
            lines = f.read().strip().split("\n")
        lines = [l for l in lines if l.strip()]

        for i in range(n_per_image):
            if aug_count >= n_needed:
                break

            # 增强图片
            augmented = AUGMENTATION(image=img)
            aug_img = augmented["image"]

            # 保存增强后的图片和标注（标注不变，因为几何变换用参数保留位置）
            # 注意：albumentations 的 HorizontalFlip/VerticalFlip/Rotate90 会改变图片坐标
            # 但 polygon 标注需要同步变换。为了简化，这里只做像素级增强（亮度、对比度、噪声等）
            # 几何变换需要重新计算 polygon 坐标，比较复杂。先只做像素级增强。

            # 实际上上面定义了 HorizontalFlip 等几何变换，需要同步变换 polygon
            # 这里简化：只保存像素级增强版本
            pass

        aug_count += 1

    print(f"  增强完成，共生成 {aug_count} 张")


def augment_train_simple(train_images, dest_img_dir, dest_label_dir):
    """简化版增强：只做像素级变换（不改变 polygon 坐标）。"""
    pixel_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.12), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
    ])

    n_original = len(train_images)
    n_needed = max(0, TARGET_TRAIN_AUGMENTED - n_original)

    if n_needed <= 0:
        print(f"  train 原始数量 {n_original} 已 >= 目标 {TARGET_TRAIN_AUGMENTED}，无需增强")
        return

    print(f"  train 原始: {n_original} 张，增强到 {TARGET_TRAIN_AUGMENTED} 张")

    n_per_image = n_needed // n_original + 1
    aug_count = 0
    base_name_map = {}

    for img_path, label_path, src_type in train_images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = Path(img_path).stem

        for i in range(n_per_image):
            if aug_count >= n_needed:
                break

            # 像素级增强
            aug_img = pixel_aug(image=img)["image"]

            # 新文件名
            new_stem = f"{stem}_aug{i:03d}"
            new_img_path = dest_img_dir / f"{new_stem}.png"
            new_label_path = dest_label_dir / f"{new_stem}.txt"

            cv2.imwrite(str(new_img_path), aug_img)
            shutil.copy2(label_path, new_label_path)

            aug_count += 1

    print(f"  增强完成，共生成 {aug_count} 张增强图")


def generate_data_yaml():
    """生成 data.yaml。"""
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: 1
names: ['GrassBeadSeg']
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return str(yaml_path)


def main():
    print("=" * 60)
    print("生成 data_v4 数据集")
    print("=" * 60)

    # 1. 收集所有图片
    all_images = collect_all_images()
    print(f"总共收集到 {len(all_images)} 张有标注的图片")

    manual_count = sum(1 for _, _, t in all_images if t == "manual")
    auto_count = sum(1 for _, _, t in all_images if t == "auto")
    print(f"  人工标注: {manual_count} 张")
    print(f"  自动标注: {auto_count} 张")

    # 2. 划分
    train, valid, test = split_dataset(all_images)
    print(f"\n划分结果:")
    print(f"  train: {len(train)} 张")
    print(f"  valid: {len(valid)} 张")
    print(f"  test:  {len(test)} 张")

    # 3. 清理并创建目录
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 4. 复制文件
    print("\n复制 train...")
    copy_files(train, DATASET_DIR / "train" / "images", DATASET_DIR / "train" / "labels")

    print("复制 valid...")
    copy_files(valid, DATASET_DIR / "valid" / "images", DATASET_DIR / "valid" / "labels")

    print("复制 test...")
    copy_files(test, DATASET_DIR / "test" / "images", DATASET_DIR / "test" / "labels")

    # 5. 增强 train
    print("\n增强 train...")
    augment_train_simple(train, DATASET_DIR / "train" / "images", DATASET_DIR / "train" / "labels")

    # 6. 生成 data.yaml
    yaml_path = generate_data_yaml()
    print(f"\ndata.yaml 生成: {yaml_path}")

    # 7. 统计
    print("\n" + "=" * 60)
    print("data_v4 生成完成！")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        n_img = len(list((DATASET_DIR / split / "images").glob("*.png")))
        n_lbl = len(list((DATASET_DIR / split / "labels").glob("*.txt")))
        print(f"  {split:5s}: {n_img:4d} images, {n_lbl:4d} labels")
    print("=" * 60)


if __name__ == "__main__":
    main()
