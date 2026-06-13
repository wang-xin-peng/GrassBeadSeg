"""
合并 data_v3（20张）+ 21-30（10张新标注）→ data_v5（30张）。

策略：
  - 保持原 test/valid 不变（01,08 在 valid；02,03 在 test），确保评估可比
  - 原 train（04-07,09-20）+ 全部新标注（21-30）→ 新 train（26张）

使用方式：
    python scripts/data_preparation/prepare_data_v5.py
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_V3 = PROJECT_ROOT / "dataset" / "data_v3"
SRC_NEW = PROJECT_ROOT / "dataset" / "21-30"
DST = PROJECT_ROOT / "dataset" / "data_v5"


def copy_split(src_dir, dst_split):
    """复制 src_dir 下的 images/ 和 labels/ 到 dst_split/"""
    for sub in ["images", "labels"]:
        src = src_dir / sub
        if src.exists():
            dst = DST / dst_split / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(str(f), str(dst / f.name))


def main():
    if DST.exists():
        shutil.rmtree(DST)

    # test 和 valid 保持原样
    copy_split(SRC_V3 / "test", "test")
    copy_split(SRC_V3 / "valid", "valid")

    # train: 原 train + 全部新标注
    copy_split(SRC_V3 / "train", "train")
    copy_split(SRC_NEW, "train")

    # 统计
    def count_img(d):
        p = DST / d / "images"
        return len(list(p.iterdir())) if p.exists() else 0

    t_train = count_img("train")
    t_valid = count_img("valid")
    t_test = count_img("test")
    total = t_train + t_valid + t_test

    # data.yaml
    (DST / "data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        "names: ['GrassBeadSeg']\n"
    )

    print(f"data_v5 生成完成: train={t_train}, valid={t_valid}, test={t_test} (总计 {total})")
    print(f"输出目录: {DST}")


if __name__ == "__main__":
    main()
