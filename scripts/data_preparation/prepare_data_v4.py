"""
生成 data_v4 数据集：合并人工标注 + RF-DETR 自动标注，重新划分，复用 augment_dataset.py 做增强。

结构：
    dataset/data_v4/
    ├── train/images/     # 28 张原始 + 增强后 ~280 张（10x）
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
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "data_v4"
RAW_DIR = PROJECT_ROOT / "dataset" / "data_v4_raw"

# 源数据
DATA_V3_DIR = PROJECT_ROOT / "dataset" / "data_v3"
AUTO_LABELED_DIR = PROJECT_ROOT / "dataset" / "auto_labeled_v3"

# 划分比例
VALID_SIZE = 6
TEST_SIZE = 6
TRAIN_SIZE = 40 - VALID_SIZE - TEST_SIZE  # 28

# 增强倍数（复用 augment_dataset.py 的默认值）
AUGMENTS_PER_IMAGE = 10

# 固定随机种子保证可复现
random.seed(42)


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
    if img_dir.exists():
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


def generate_data_yaml(data_dir):
    """生成 data.yaml。"""
    yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['GrassBeadSeg']
"""
    yaml_path = Path(data_dir) / "data.yaml"
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

    # 3. 清理旧目录
    for d in [DATASET_DIR, RAW_DIR]:
        if d.exists():
            shutil.rmtree(d)

    # 4. 先把原始数据组织到 raw 目录（标准 YOLO 结构）
    print("\n组织原始数据到临时目录...")
    copy_files(train, RAW_DIR / "train" / "images", RAW_DIR / "train" / "labels")
    copy_files(valid, RAW_DIR / "valid" / "images", RAW_DIR / "valid" / "labels")
    copy_files(test, RAW_DIR / "test" / "images", RAW_DIR / "test" / "labels")
    generate_data_yaml(RAW_DIR)

    # 5. 调用 augment_dataset.py 进行增强
    augment_script = PROJECT_ROOT / "scripts" / "data_augmentation" / "augment_dataset.py"
    cmd = [
        sys.executable,
        str(augment_script),
        "--input", str(RAW_DIR),
        "--output", str(DATASET_DIR),
        "--augments-per-image", str(AUGMENTS_PER_IMAGE),
    ]
    print(f"\n调用增强脚本: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("增强脚本执行失败")
        sys.exit(1)

    # 6. 清理临时 raw 目录
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
        print(f"\n已清理临时目录: {RAW_DIR}")

    # 7. 统计
    print("\n" + "=" * 60)
    print("data_v4 生成完成！")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        n_img = len(list((DATASET_DIR / split / "images").glob("*.png")))
        n_lbl = len(list((DATASET_DIR / split / "labels").glob("*.txt")))
        print(f"  {split:5s}: {n_img:4d} images, {n_lbl:4d} labels")
    print("=" * 60)
    print(f"\n接下来可以用以下命令训练:")
    print(f"  python src/train.py --data dataset/data_v4/data.yaml --model nano")


if __name__ == "__main__":
    main()
