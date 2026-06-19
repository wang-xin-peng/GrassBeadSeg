# GlassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 路线演进

从 10 张原始人工标注出发，经过 4 个版本迭代，逐步提升数据规模、增强策略和推理管线，最终将 F1@0.5 从 0.560 提升至 0.752（+34.3%）。

| 版本      | 人工标注(总数) |     训练增强     | 训练图数 | 在线增强      | 推理策略              |     F1@0.5     |     Recall     |    Precision    |    计数误差    | 最优权重          |
| --------- | :------------: | :--------------: | :------: | ------------- | --------------------- | :-------------: | :-------------: | :-------------: | :------------: | ----------------- |
| v1 (nano) |     10 张     | 10× (5 种变换) |    66    | mosaic only   | baseline / ellipse    |      0.560      |      0.554      |      0.566      |      2.5%      | best.pt           |
| v2 (nano) |     20 张     | 15× (15 种变换) |   256   | +6 种在线增强 | SAHI + ellipse + 去重 |      0.647      |      0.662      |      0.633      |      4.5%      | last.pt           |
| v3 (nano) |     30 张     | 15× (15 种变换) |   416   | 同 v2         | + TTA                 |      0.738      |      0.766      |      0.712      |      7.7%      | last.pt           |
| v4 (nano) |     40 张     | 15× (15 种变换) |   576   | 同 v2         | + 精细调参            | **0.752** | **0.767** | **0.737** | **4.1%** | **best.pt** |

> **逐版提升**: v1→v2 +15.5% | v2→v3 +14.1% | v3→v4 +1.9% | 总计 v1→v4 **+34.3%**（0.560→0.752）

### 预训练模型选择

四个版本均使用 **YOLOv11n-seg**（2.6M 参数）作为基础模型，加载 **COCO 预训练权重**（`yolo11n-seg.pt`）。v1/v2 同时对比了 YOLOv11s-seg（9.4M 参数）：

- **v1-small**：F1=0.628，略高于 nano (0.560)，但推理速度和显存占用明显更大
- **v2-small**：计数误差高达 50.1%（nano 仅 4.5%），说明小数据集上 large model 过拟合严重
- v2 之后确定 nano 为最优选择，弃用 small

结论：在 10~40 张小样本场景下，**2.6M 的 nano 比 9.4M 的 small 更抗过拟合**，且推理更快、显存更省。预训练权重均来自 COCO 实例分割任务，提供了良好的底层特征提取能力。

### v1: Baseline

- **数据集**：`data_v1` — 10 张人工标注（train=6, valid=2, test=2）
- **离线增强**：`data_v1_aug` — 每图 10×，共 66 张训练图（6 原始 + 60 增强）。5 种 albumentations 变换：HorizontalFlip, Rotate±15°, ColorJitter, GaussianBlur, PixelDropout
- **在线增强**：仅 YOLO 默认 mosaic=1.0，无 scale/rotate/shear/mixup/copy_paste
- **推理**：baseline 或 ellipse 模式，conf=0.3，无任何去重后处理
- **训练**：200 epochs, AdamW + cosine lr + warmup=3, batch=8
- **结果**：v1-nano F1=0.560；v1-small (9.4M) F1=0.628（nano 在小数据上更抗过拟合）

### v2: 数据翻倍 + 增强全面升级

- **数据集**：`data_v2` — 20 张人工标注（train=16, valid=2, test=2），新增 10 张标注
- **离线增强**：`data_v2_aug` — 每图 15×，共 256 张训练图（16 原始 + 240 增强）。变换从 5→15 种：新增 VerticalFlip, Affine(scale+translate+rotate+shear), ElasticTransform, OpticalDistortion, RandomBrightnessContrast, CLAHE, MotionBlur, GaussNoise, ISONoise, Sharpen, Posterize
- **在线增强**：从 1→7 种。新增 scale=0.9, degrees=15, shear=5, perspective=0.0005, flipud=0.1, mixup=0.1, copy_paste=0.1
- **推理管线重构**：SAHI 分块推理 (640×640) + ellipse fitting + 二次去重（中心距离 ≤30px 且 mask IoU > 0.15）
- **训练**：200 epochs, AdamW + cosine lr + warmup=3, batch=8
- **结果**：
  - v2-nano best.pt：F1=0.669, Recall=0.771, Precision=0.591, 但过检严重导致计数误差 30.4%
  - **v2-nano last.pt（推荐）**：F1=0.647, Recall=0.662, Precision=0.633, 计数误差 4.5%
  - 首次发现 **last.pt > best.pt**：best.pt 的 mAP 虽高但过拟合导致大量误检

### v3: 扩大数据 + TTA

- **数据集**：`data_v3` — 30 张人工标注（train=26, valid=2, test=2），新增 10 张标注
- **离线增强**：`data_v3_aug` — 每图 15×，共 416 张训练图（26 原始 + 390 增强）。albumentations pipeline
- **在线增强**：与 v2 相同
- **推理新增加 TTA**（水平翻转共识）
- **训练**：200 epochs, AdamW + cosine lr, batch=16
- **结果**：
  - **v3 last.pt 最佳 F1**：0.738 (overlap=0.32, conf=0.38, TTA 开)，best.pt 最佳 F1 仅 0.726
  - v3 最低计数误差：F1=0.731, Err=1.5% (overlap=0.30, conf=0.41, TTA)
  - 再次确认 last.pt > best.pt

### v4:  扩大数据 + 精细参数搜索

- **数据集**：`data_v4` — 40 张纯人工标注（train=36, valid=2, test=2），新增 10 张标注，全部人工标注无混合自动标注
- **离线增强**：`data_v4_aug` — 每图 15×，共 576 张训练图（36 原始 + 540 增强）。albumentations pipeline
- **在线增强**：与 v2/v3 相同
- **训练**：200 epochs, AdamW + cosine lr, batch=16
- **推理参数通过 6 轮 sweep 扫描精细确定**：

  - sweep_v4.py：overlap × conf 粗扫描（21 组）
  - sweep_v4_ov_conf.py：ov × conf 交叉搜索（30 组）
  - sweep_v4_conf.py：置信度 + TTA 扫描（10 组）
  - sweep_v4_fine.py：五维精细调参（33 组，含 dedup_iou/dist/yolo_iou/overlap）
  - sweep_v4_r3.py：大 overlap + TTA + soft-nms（38 组）
- **v4 best.pt 最佳结果**：F1=0.752 (overlap=0.35, conf=0.35, TTA 开)
- v4 最低计数误差：F1=0.744, Err=1.9% (overlap=0.40, conf=0.37, TTA)
- **v4 首次 best.pt > last.pt**：更多纯人工标注数据使 best.pt 的过拟合问题得到缓解
- dedup/dist/yolo_iou 在 v4 上已全面饱和：18 组 c040_tta 变体全部得到相同的 F1=0.7371

---

## 推荐方案

| 场景                  | 权重    | overlap | conf | TTA |  F1  | 计数误差 |
| --------------------- | ------- | :-----: | :--: | :-: | :---: | :------: |
| **v4 综合最佳** | best.pt |  0.35  | 0.35 | 开 | 0.752 |   4.1%   |
| v4 最低计数误差       | best.pt |  0.40  | 0.37 | 开 | 0.744 |   1.9%   |
| v3 最高 F1            | last.pt |  0.32  | 0.38 | 开 | 0.738 |   7.7%   |
| v3 最低计数误差       | last.pt |  0.30  | 0.41 | 开 | 0.731 |   1.5%   |
| v2 可用               | last.pt |  0.20  | 0.30 | 关 | 0.647 |   4.5%   |

---

## 数据版本一览

| 版本 | 数据集目录  | 人工标注 | train / valid / test | 增强目录        | 增强倍数 | 训练图数 |
| ---- | ----------- | :------: | :------------------: | --------------- | :------: | :------: |
| v1   | `data_v1` |    10    |      6 / 2 / 2      | `data_v1_aug` |   10×   |    66    |
| v2   | `data_v2` |    20    |      16 / 2 / 2      | `data_v2_aug` |   15×   |   256   |
| v3   | `data_v3` |    30    |      26 / 2 / 2      | `data_v3_aug` |   15×   |   416   |
| v4   | `data_v4` |    40    |      36 / 2 / 2      | `data_v4_aug` |   15×   |   576   |

---

## 推理流水线

```
原始图像 (1920×1080)
       │
       ▼
┌  阶段一: SAHI 分块推理  ──────────────────────────
│  滑动窗口 640×640, overlap=0.35
│  YOLOv11n-seg (2.6M) 逐块推理
│  输出 → polygon mask 集合
└─────────────────────────────────────────────────
       │
       ▼
┌  阶段二: 去重  ──────────────────────────────────
│  ① NMS        mask IoU 抑制重叠框 (iou=0.7)
│  ② 二次去重    中心距离 + IoU 过滤碎片 (dist≤30, iou>0.15)
│  ③ TTA        水平翻转共识（原图 + 翻转图交集）
└─────────────────────────────────────────────────
       │
       ▼
┌  阶段三: 椭圆拟合  ─────────────────────────────
│  ④ findContours 提取轮廓
│  ⑤ convexHull   凸包填充阴影凹陷
│  ⑥ fitEllipse   光滑椭圆拟合
│  输出 → 椭圆掩膜 (.txt, YOLO 格式)
└─────────────────────────────────────────────────
```

### 关键参数

| 阶段 | 参数               |   默认   |      v4 推荐      |      v3 推荐      | 说明                         |
| ---- | ------------------ | :------: | :---------------: | :---------------: | ---------------------------- |
| SAHI | `--overlap`      |   0.2   |  **0.35**  |  **0.32**  | 分块重叠率，v3/v4 需更大覆盖 |
| SAHI | `--conf`         |   0.3   |  **0.35**  |  **0.38**  | 置信度阈值                   |
| 去重 | `--yolo-iou`     |   0.7   |        0.7        |        0.7        | NMS IoU（v4 已饱和）         |
| 去重 | `--dedup-iou`    |   0.15   |       0.15       |       0.15       | 二次去重 IoU（v4 已饱和）    |
| 去重 | `--dedup-dist`   |    30    |        30        |        30        | 中心距离阈值 px（v4 已饱和） |
| 去重 | `--tta`          |    关    |   **开**   |   **开**   | 水平翻转共识                 |
| 椭圆 | `--mode ellipse` | baseline | **ellipse** | **ellipse** | 启用椭圆拟合                 |

---

## 项目结构

```
GlassBeadSeg/
├── src/
│   ├── train.py                  # YOLOv11-seg 训练脚本（支持 --resume）
│   ├── inference.py              # 推理脚本（SAHI / ellipse / TTA / Soft-NMS / dedup）
│   └── eval.py                   # Greedy IoU 评估脚本
├── scripts/
│   ├── data_augmentation/
│   │   └── augment_dataset.py    # 离线增强（albumentations, 可配置倍数）
│   ├── auto_label/
│   │   ├── train_rfdetr.py       # RF-DETR 训练
│   │   └── inference_rfdetr.py   # RF-DETR 自动标注推理
│   ├── new_images/
│   │   └── test_new_images.py    # 新图片抽样测试脚本
│   └── sweep/
│       ├── compare_models.py      # 多模型自动对比
│       ├── sweep_best_pt.py       # best.pt / last.pt 对比扫描 (v1/v2)
│       ├── sweep_best_pt_v2.py    # v2 专属 best.pt / last.pt 对比
│       ├── sweep_v3_best.py       # v3 best.pt overlap 扫描
│       ├── sweep_v3_conf.py       # v3 置信度 + TTA 扫描
│       ├── sweep_v3_ov_conf.py    # v3 overlap × conf 交叉搜索
│       ├── sweep_v3_fine.py       # v3 五维精细调参
│       ├── sweep_v3_r3.py         # v3 大 overlap + soft-nms
│       ├── sweep_v4.py            # v4 overlap × conf 粗扫描
│       ├── sweep_v4_conf.py       # v4 置信度 + TTA 扫描
│       ├── sweep_v4_ov_conf.py    # v4 overlap × conf 交叉搜索
│       ├── sweep_v4_fine.py       # v4 五维精细调参
│       └── sweep_v4_r3.py         # v4 大 overlap + soft-nms
├── dataset/
│   ├── raw/                      # 原始未标注图片
│   ├── new_images/               # 新增原始图片
│   ├── data_v1 / data_v1_aug/    # v1 标注及增强 (10 张, 10×)
│   ├── data_v2 / data_v2_aug/    # v2 标注及增强 (20 张, 15×)
│   ├── data_v3 / data_v3_aug/    # v3 标注及增强 (30 张, 15×)
│   ├── data_v4 / data_v4_aug/    # v4 标注及增强 (40 张, 15×)
│   └── auto_labeled/             # RF-DETR 自动标注结果
├── models/
│   ├── download_pretrained.py      # 下载 rfdetr_seg_large 预训练权重
│   ├── yolo11n-seg.pt             # YOLOv11n-seg COCO 预训练权重
│   ├── yolo11s-seg.pt             # YOLOv11s-seg COCO 预训练权重
│   ├── glass_bead_seg/
│   │   ├── yolo11n-seg-v1/        # v1 nano 模型
│   │   ├── yolo11n-seg-v2/        # v2 nano 模型
│   │   ├── yolo11n-seg-v3/        # v3 nano 模型
│   │   ├── yolo11n-seg-v4/        # v4 nano 模型
│   │   ├── yolo11s-seg-v1/        # v1 small 模型
│   │   └── yolo11s-seg-v2/        # v2 small 模型
│   └── rfdetr_seg_large/          # RF-DETR 模型 (预训练 + fine-tune 权重)
├── outputs/                      # 推理输出
├── logs/
│   ├── sweep/                    # 参数搜索评估日志
│   └── train/                    # 训练日志
└── requirements.txt
```

---

## 环境配置

```bash
conda create -n gbseg python=3.10 -y
conda activate gbseg
pip install -r requirements.txt
```

---

## 使用流程

### 1. 数据准备

- `dataset/data_v4/`（40 张人工标注，train=36/valid=2/test=2）

生成增强数据：

```bash
python scripts/data_augmentation/augment_dataset.py \
  --input dataset/data_v4 --output dataset/data_v4_aug --augments-per-image 15
```

### 2. 训练

```bash
# 从头训练 v4 nano
python src/train.py --data dataset/data_v4_aug --model n --name v4

# 离线环境可加 --no-amp
python src/train.py --data dataset/data_v4_aug --model n --name v4 \
  --batch 16 --workers 16 --device 0 --no-amp
```

### 3. 推理

```bash
# v4 best.pt + TTA（最高 F1=0.752）
python src/inference.py \
  --model models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt \
  --source <images> \
  --output outputs/v4_result \
  --mode ellipse \
  --overlap 0.35 \
  --conf 0.35 \
  --tta

# v4 最高计数精度（计数误差 1.9%）
python src/inference.py \
  --model models/glass_bead_seg/yolo11n-seg-v4/weights/best.pt \
  --source <images> \
  --output outputs/v4_result \
  --mode ellipse \
  --overlap 0.40 \
  --conf 0.37 \
  --tta

# v3 last.pt + TTA（F1=0.738, Err=7.7%）
python src/inference.py \
  --model models/glass_bead_seg/yolo11n-seg-v3/weights/last.pt \
  --source <images> \
  --output outputs/result \
  --mode ellipse \
  --overlap 0.32 \
  --conf 0.38 \
  --tta

# 查看所有选项
python src/inference.py --help
```

### 4. 评估

```bash
# 评估 v4 结果
python src/eval.py \
  --pred outputs/v4_result/labels \
  --gt dataset/data_v4/test/labels \
  --images dataset/data_v4/test/images

# 评估 v3 结果
python src/eval.py \
  --pred outputs/result/labels \
  --gt dataset/data_v3/test/labels \
  --images dataset/data_v3/test/images
```

### 5. 参数搜索

```bash
# v4
python scripts/sweep/sweep_v4.py          # overlap × conf 粗扫描
python scripts/sweep/sweep_v4_conf.py     # 置信度 + TTA 扫描
python scripts/sweep/sweep_v4_ov_conf.py  # overlap × conf 交叉搜索
python scripts/sweep/sweep_v4_fine.py     # 五维精细调参
python scripts/sweep/sweep_v4_r3.py       # 大 overlap + soft-nms

# v3
python scripts/sweep/sweep_v3_best.py     # best.pt overlap 扫描
python scripts/sweep/sweep_v3_conf.py     # 置信度 + TTA 扫描
python scripts/sweep/sweep_v3_ov_conf.py  # overlap × conf 交叉搜索
python scripts/sweep/sweep_v3_fine.py     # 五维精细调参
python scripts/sweep/sweep_v3_r3.py       # 大 overlap + soft-nms
```

### 6. RF-DETR 自动标注

```bash
python scripts/auto_label/inference_rfdetr.py
```

项目中使用 RF-DETR 对原始图片进行自动标注以加速数据扩展的路径。RF-DETR 在 data_aug_v1上完成训练后，对 30 张新图片生成了自动标注结果。由于存在
~15% 的漏标率，直接混入训练集会拉低模型 precision，只能用于辅助人工标注。

---

## 关键设计决策

### 数据策略

- **数据量是核心瓶颈**：v1(10张)→v4(40张)，F1 从 0.560→0.752（+34.3%）。40 张下 nano (2.6M) 比 small (9.4M) 更抗过拟合，继续扩展数据量仍有空间
- **v1→v2 数据翻倍 + 增强升级**：标注从 10→20 张（+100%），离线增强从 10×→15×、变换从 5→15 种、在线增强从 1→7 种、推理管线从 baseline 升级到 SAHI+ellipse+dedup，F1 从 0.560→0.647（+15.5%）
- **离线增强的边际效应**：v1(10×) vs v2(15×) 在同数据上提升显著，但主要贡献来自变换种类增加（5→15 种）而非单纯扩大倍数
- **纯人工标注优于混合标注**：自动标注的漏标率会拉低 precision。v4 全部采用纯人工标注或经过人工修复的自动标注，precision 从 v3 的 0.712→0.737（+2.5pp）

### 推理策略

- **SAHI 的必要性**：1920×1080 直接推理显存不足，分块后逐块处理。相邻分块会产生同一珠子的碎片化重复检测
- **两层去重**：NMS 抑制重叠框 → 中心距离+IoU 二次去重消除跨块碎片
- **凸包 + 椭圆**：YOLO 输出的多边形受阴影/反光影响呈锯齿状甚至裂块。convexHull 填平凹陷，fitEllipse 还原玻璃珠的圆形本质
- **TTA 仍然刚需**：v4 关闭 TTA，F1 仅掉 ~0.006 但计数误差从 2-4% 飙升至 16.5%，说明 TTA 对计数精度的保护作用不可替代
- **v3/v4 需要大 overlap**：最优 overlap v4=0.35 / v3=0.32，较默认值 0.2 大幅提升，更大的分块重叠可覆盖边界珠子

### 训练策略

- **last.pt vs best.pt**：
  - v2/v3 上 last.pt 优于 best.pt：best.pt 在验证集上 mAP 更高但因过拟合导致大量误检，计数误差严重
  - v4 上 best.pt 优于 last.pt：更多数据（40 张）缓解了过拟合，best.pt 的 precision 达到 0.737
- **v4 去重参数全面饱和**：dedup_iou (0.05~0.18)、dedup_dist (25~55)、yolo_iou (0.45~0.75) 在大范围内扫描，18 组 c040_tta 变体全部得到相同 F1=0.7371，说明 SAHI 本身的重复检测问题已在 v4 训练质量下大幅减少
- **Soft-NMS 无效**：在 v4 上 3 组 soft-nms 试验结果与 hard-nms 完全相同，无增益

### 工程优化

- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级
- **v1-small (9.4M) 过拟合严重**：小数据集上 nano (2.6M) 始终优于 small，v2-small 计数误差高达 50.1%

---

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
