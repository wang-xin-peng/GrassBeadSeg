import os
import sys
import torch
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = PROJECT_ROOT / "models" / "rfdetr_seg_large" / "pretrained_300.pth"
DATASET_DIR = PROJECT_ROOT / "dataset" / "data_v3_augmented"
OUTPUT_DIR = PROJECT_ROOT / "models" / "rfdetr_seg_large" / "trained"

EPOCHS = 200
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4
RESOLUTION = 624
NUM_QUERIES = 800


def expand_pretrained_weights(src_path, dst_path, target_num_queries=600, group_detr=13):
    checkpoint = torch.load(str(src_path), map_location="cpu")
    ckpt_model = checkpoint["model"]

    first_key = next(k for k in ckpt_model if "refpoint_embed.weight" in k)
    src_rows = ckpt_model[first_key].shape[0]
    src_num_queries = src_rows // group_detr

    if src_num_queries == target_num_queries:
        torch.save(checkpoint, str(dst_path))
        return

    print(f"扩展 query 参数: {src_num_queries} -> {target_num_queries} (每组 {src_num_queries} -> {target_num_queries})")

    for key in ["refpoint_embed.weight", "query_feat.weight"]:
        tensor = ckpt_model[key]
        groups = tensor.chunk(group_detr, dim=0)
        factor = target_num_queries / src_num_queries
        if factor == int(factor):
            expanded = torch.cat(
                [g.repeat_interleave(int(factor), dim=0) for g in groups], dim=0
            )
        else:
            expanded = torch.cat(
                [g.repeat((target_num_queries + src_num_queries - 1) // src_num_queries, 1)[:target_num_queries] for g in groups],
                dim=0,
            )
        ckpt_model[key] = expanded

    torch.save(checkpoint, str(dst_path))
    print(f"扩展后的预训练权重已保存: {dst_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        print(f"错误: 数据集路径不存在: {DATASET_DIR}")
        sys.exit(1)

    if not PRETRAINED_PATH.exists():
        print(f"错误: 预训练权重不存在: {PRETRAINED_PATH}")
        print("请先在本地有网络的环境下运行 models/download_pretrained.py 下载权重")
        sys.exit(1)

    try:
        from rfdetr import RFDETRSegLarge
    except ImportError:
        print("错误: 请先安装 rfdetr: pip install rfdetr")
        sys.exit(1)

    EXPANDED_PATH = PROJECT_ROOT / "models" / "rfdetr_seg_large" / f"pretrained_{NUM_QUERIES}.pth"
    expand_pretrained_weights(PRETRAINED_PATH, EXPANDED_PATH, target_num_queries=NUM_QUERIES)

    print("=" * 60)
    print("RF-DETR 分割模型训练")
    print("=" * 60)
    print(f"预训练权重: {EXPANDED_PATH}")
    print(f"数据集: {DATASET_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练轮数: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRAD_ACCUM_STEPS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"分辨率: {RESOLUTION}")
    print(f"查询数量: {NUM_QUERIES}")
    print(f"最大检测数: {NUM_QUERIES}")
    print("=" * 60)

    model = RFDETRSegLarge(pretrain_weights=str(EXPANDED_PATH), num_queries=NUM_QUERIES, num_select=NUM_QUERIES)

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        resolution=RESOLUTION,
        output_dir=str(OUTPUT_DIR),
        early_stopping=True,
        early_stopping_patience=30,
        early_stopping_min_delta=0.001,
        tensorboard=True,
    )

    print(f"\n训练完成！检查点保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()