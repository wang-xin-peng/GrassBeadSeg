# GrassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 最优模型 — YOLOv11n-seg v2

| 指标 | v1 (baseline) | **v2 (推荐)** |
|------|:---:|:---:|
| 权重 | yolo11n-seg-baseline_v1 | **yolo11n-seg-v2 (last.pt)** |
| Recall@0.5 | 0.554 | **0.662** |
| Precision@0.5 | 0.566 | **0.633** |
| F1@0.5 | 0.560 | **0.647** |
| 计数误差 | 2.5% | **4.5%** |

v2 改进点：
- 离线增强：20× per image（v1 为 10×），更强的 albumentations pipeline
- 在线增强：scale=0.9, degrees=15, shear=5, perspective, mixup, copy_paste
- 训练 200 epochs，AdamW + cosine lr

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

## 完整流程

```plaintext
YOLO 输出 polygon mask
        │
        ▼
① 提取轮廓 (cv2.findContours)
        │
        ▼
② 凸包填充凹陷 (cv2.convexHull)
        │
        ▼
③ 拟合椭圆 (cv2.fitEllipse)
        │
        ▼
④ 画椭圆 mask 替换原始 polygon
```

说明：

- YOLO 直接输出的分割是**不规则多边形**，受阴影、反光影响，珠子会被切成锯齿状甚至裂成几块
- **凸包**先把这些凹陷填平——把"被阴影劈开的"重新算成一个整体
- **fitEllipse** 再把凸包拟合成光滑椭圆——因为玻璃珠本身就是圆的，椭圆比多边形更接近真实形状

## 项目结构

```
GrassBeadSeg/
├── src/
│   ├── train.py              # YOLOv11-seg 训练脚本（支持 --resume 续训）
│   ├── inference.py          # 推理脚本（baseline / sahi / ellipse / TTA / Soft-NMS）
│   └── eval.py               # Greedy IoU 评估脚本
├── scripts/
│   ├── data_augmentation/
│   │   └── augment_dataset.py # 几何 + 像素增强（albumentations, 20×）
│   ├── data_preparation/
│   │   └── prepare_data_v4.py # data_v4 数据集生成
│   ├── train/
│   │   └── train_rfdetr.py    # RF-DETR 训练
│   ├── inference/
│   │   └── inference_rfdetr.py # RF-DETR 自动标注推理
│   └── eval/
│       └── compare_models.py  # v1/v2 模型自动对比脚本
├── dataset/
│   ├── raw/                   # 40 张原始未标注图片
│   ├── data_v3/               # 20 张人工标注 + 增强数据集
│   ├── data_v3_augmented/     # v1 增强数据（10×）
│   ├── data_v3_aug_v2/        # v2 增强数据（20×）
│   ├── data_v4/               # 人工 + 自动标注混合
│   └── auto_labeled_v*/       # RF-DETR 自动标注结果
├── models/
│   ├── our_method/
│   │   ├── yolo11n-seg-baseline_v1/  # v1 nano 权重
│   │   ├── yolo11s-seg-baseline_v1/  # v1 small 权重
│   │   ├── yolo11n-seg-v2/           # ★ v2 nano 权重（推荐）
│   │   └── yolo11s-seg-v2/           # v2 small 权重
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

直接使用 `dataset/data_v3/`（20 张人工标注）。

**方式二：从头生成增强数据**

```bash
python scripts/data_augmentation/augment_dataset.py \
  --input dataset/data_v3 --output dataset/data_v3_augmented
```

### 2. 训练

```bash
# 从头训练 nano
python src/train.py --data dataset/data_v3_aug_v2 --model n

# 续训（从 last.pt 恢复）
python src/train.py --data dataset/data_v3_aug_v2 --model n --resume

# 服务器 A800 配置
python src/train.py --data dataset/data_v3_aug_v2 --model n \
  --batch 32 --workers 16 --device 0 --no-amp
```

### 3. 推理

```bash
# SAHI + ellipse 模式（推荐）
python src/inference.py \
  --model models/our_method/yolo11n-seg-v2/weights/last.pt \
  --source dataset/data_v3_augmented/test/images \
  --output outputs/test_result \
  --mode ellipse --conf 0.3

# 启用 TTA（水平翻转，提升 recall 2-5%）
python src/inference.py --mode ellipse --tta ...

# 查看所有选项
python src/inference.py --help
```

### 4. 评估

```bash
python src/eval.py \
  --pred outputs/test_result/labels \
  --gt dataset/data_v3_augmented/test/labels \
  --images dataset/data_v3_augmented/test/images
```

### 5. v1 vs v2 自动对比

```bash
python scripts/eval/compare_models.py
# 自动推理 + 评估 + 打印 F1/Recall/Precision 对比表
```

### 6. RF-DETR 自动标注（可选）

```bash
python scripts/inference/inference_rfdetr.py
```

RF-DETR v3 自动标注 recall ~85%，可用于辅助标注，但注意漏标噪声会影响训练质量，纯人工标注在精确计数任务上效果更优。

## 关键设计决策

- **nano > small**：20 张训练数据下，nano (2.6M) F1=0.647 > small (9.4M) F1=0.628，小容量更抗过拟合
- **椭圆拟合优于原始分割**：ConvexHull + fitEllipse 消除阴影凹陷和碎片化，减少重复检测
- **纯人工标注优于混合标注**：自动标注的 ~15% 漏标率会显著拉低模型 precision
- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级
- **last.pt > best.pt**：best.pt 追求最高 mAP 导致过检（30% 计数误差），last.pt 更平衡（4.5% 计数误差）

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
