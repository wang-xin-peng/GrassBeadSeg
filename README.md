# GrassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 技术路线（our_method）

```
原始图像 (1920×1080)
       │
       ▼
  SAHI 滑动窗口分块 (640×640, 20% overlap)
       │
       ▼
  YOLOv11n-seg 推理 (2.6M 参数, COCO 预训练)
       │
       ▼
  Convex Hull 凸包 + fitEllipse 椭圆拟合
       │
       ▼
  Mask IoU NMS 去重
       │
       ▼
  最终椭圆掩膜输出
```

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv11n-seg (2.6M) |
| 训练数据 | 20 张人工标注（data_v3），10 倍增强 |
| 推理模式 | SAHI + ellipse |
| 置信度阈值 | 0.3 |
| Recall@0.5 | 0.512 |
| Precision@0.5 | 0.557 |
| F1@0.5 | 0.534 |
| 平均计数误差 | 7.6% |

## 项目结构

```
GrassBeadSeg/
├── src/
│   ├── train.py              # YOLOv11-seg 训练脚本
│   ├── inference.py          # 推理脚本（baseline / sahi / ellipse）
│   └── eval.py               # Greedy IoU 评估脚本
├── scripts/
│   ├── data_augmentation/
│   │   └── augment_dataset.py # 几何 + 像素增强（albumentations）
│   ├── data_preparation/
│   │   └── prepare_data_v4.py # data_v4 数据集生成
│   ├── train/
│   │   └── train_rfdetr.py    # RF-DETR 训练
│   └── inference/
│       └── inference_rfdetr.py # RF-DETR 自动标注推理
├── dataset/
│   ├── raw/                   # 40 张原始未标注图片
│   ├── data_v3/               # 20 张人工标注数据集
│   │   ├── train/ (16 张), valid/ (2 张), test/ (2 张)
│   │   └── data.yaml
│   ├── data_v3_augmented/     # 增强后数据集（~200 张 train）
│   ├── data_v4/               # 人工 + 自动标注混合数据集
│   └── auto_labeled_v*/       # RF-DETR 自动标注结果
├── models/
│   ├── our_method/            # YOLOv11 训练权重
│   │   ├── yolo11n-seg-baseline_v1/  # nano 最佳权重
│   │   └── yolo11s-seg-baseline_v1/  # small 权重
│   └── rfdetr_seg_large/      # RF-DETR 权重及预训练模型
├── outputs/                   # 推理输出（标签 + 可视化）
├── logs/                      # 训练与评估日志
└── requirements.txt
```

## 环境配置

```bash
conda create -n gbseg python=3.10 -y
conda activate gbseg
pip install -r requirements.txt
```

服务器部署时使用 `opencv-python-headless` 以避免 libGL 依赖。

## 使用流程

### 1. 数据准备

**方式一：使用已有数据集 data_v3**

直接使用 `dataset/data_v3_augmented/`（20 张人工标注 + 10 倍增强）。

**方式二：从头生成**

生成 data_v4（40 张混合标注，不推荐用于训练）：

```bash
python scripts/data_preparation/prepare_data_v4.py
```

对任意 YOLO 格式数据集做增强：

```bash
python scripts/data_augmentation/augment_dataset.py \
  --input dataset/data_v3 --output dataset/data_v3_augmented
```

### 2. 训练

```bash
# 默认配置（data_v3_augmented, nano + small）
python src/train.py

# 指定数据集
python src/train.py --data dataset/data_v3_augmented --model n

# 服务器 A800 配置
python src/train.py --model n --batch 32 --workers 16 --no-amp
```

### 3. 推理

```bash
# SAHI + ellipse 模式（推荐）
python src/inference.py \
  --model models/our_method/yolo11n-seg-baseline_v1/weights/best.pt \
  --source dataset/data_v3/test/images \
  --output outputs/test_result \
  --mode ellipse --conf 0.3

# 支持三种模式: baseline / sahi / ellipse
python src/inference.py --help
```

### 4. 评估

```bash
python src/eval.py \
  --pred outputs/test_result/labels \
  --gt dataset/data_v3/test/labels \
  --images dataset/data_v3/test/images
```

### 5. RF-DETR 自动标注（可选）

```bash
# 用于批量标注新图片
python scripts/inference/inference_rfdetr.py
```

RF-DETR v3 自动标注 recall ~85%，可用于辅助标注，但注意漏标噪声会影响训练质量，纯人工标注在精确计数任务上效果更优。

## 关键设计决策

- **小模型优于大模型**：20 张训练数据下，nano (2.6M) F1=0.534 > small (9.4M) F1=0.301，小容量更抗过拟合
- **椭圆拟合优于原始分割**：ConvexHull + fitEllipse 消除阴影凹陷和碎片化，减少重复检测，计数误差降至 7.6%
- **纯人工标注优于混合标注**：自动标注的 ~15% 漏标率会显著拉低模型 precision（0.557 → 0.390）
- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级，对 NMS 效果无影响

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
