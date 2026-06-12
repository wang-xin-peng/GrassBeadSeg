# GrassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 最优模型 — YOLOv11n-seg v2

| 指标 | v1 (baseline) | **v2 last.pt** | **v2 best.pt + 优化** |
|------|:---:|:---:|:---:|
| 权重 | yolo11n-seg-baseline_v1 | yolo11n-seg-v2 (last.pt) | yolo11n-seg-v2 (best.pt) |
| Recall@0.5 | 0.554 | **0.662** | **0.736** |
| Precision@0.5 | 0.566 | **0.633** | **0.695** |
| F1@0.5 | 0.560 | **0.647** | **0.715** |
| 计数误差 | 2.5% | **4.5%** | **5.9%** |

**推荐方案**：

- 如需**最高计数精度**（3.3% 误差）→ `last.pt` + dedup + TTA
- 如需**最高 F1 / 检出率**（0.715）→ `best.pt` + 去重 + TTA + conf=0.34（下详）

v2 改进点：
- 离线增强：20× per image（v1 为 10×），更强的 albumentations pipeline
- 在线增强：scale=0.9, degrees=15, shear=5, perspective, mixup, copy_paste
- 训练 200 epochs，AdamW + cosine lr

## 技术路线（our_method）

```
原始图像 (1920×1080)
       │
       ▼
  SAHI 滑动窗口分块 (640×640, 15% overlap)
       │
       ▼
  YOLOv11n-seg 推理 (2.6M 参数, COCO 预训练)
       │
       ▼
  Mask IoU NMS (Hard) 去重
       │
       ▼
  中心距离 + IoU 二次去重 (dedup)
       │
       ▼
  [可选] TTA 水平翻转共识过滤
       │
       ▼
  Convex Hull 凸包 + fitEllipse 椭圆拟合
       │
       ▼
  最终椭圆掩膜输出
```

## 完整流程

```plaintext
YOLO 输出 polygon mask
        │
        ▼
① NMS 去重 — 基于 mask IoU 抑制重叠检测
        │
        ▼
② 二次去重 — 基于中心距离 + IoU 去除碎片化重复
        │
        ▼
③ 提取轮廓 (cv2.findContours)
        │
        ▼
④ 凸包填充凹陷 (cv2.convexHull)
        │
        ▼
⑤ 拟合椭圆 (cv2.fitEllipse)
        │
        ▼
⑥ 画椭圆 mask 替换原始 polygon
```

说明：

- **NMS + 二次去重**：SAHI 分块推理会在相邻块产生同一珠子的碎片化重复检测，通过 mask IoU NMS（IoU>0.5）和中心距离+IoU 去重（dist<40px, IoU>0.10）两层过滤
- YOLO 直接输出的分割是**不规则多边形**，受阴影、反光影响，珠子会被切成锯齿状甚至裂成几块
- **凸包**先把这些凹陷填平——把"被阴影劈开的"重新算成一个整体
- **fitEllipse** 再把凸包拟合成光滑椭圆——因为玻璃珠本身就是圆的，椭圆比多边形更接近真实形状
- **TTA（水平翻转共识）**：原图和翻转图各推理一次，只保留两边都检测到的预测，提升 recall 2-3pp

## 项目结构

```
GrassBeadSeg/
├── src/
│   ├── train.py              # YOLOv11-seg 训练脚本（支持 --resume 续训）
│   ├── inference.py          # 推理脚本（baseline / sahi / ellipse / TTA / Soft-NMS / dedup）
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
│       ├── compare_models.py  # v1/v2 模型自动对比（支持 --dedup-iou）
│       ├── sweep_best_pt.py   # best.pt 去重参数网格搜索
│       └── sweep_best_pt_v2.py # best.pt 第二轮优化搜索
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
├── outputs/                   # 推理输出（标签 + 可视化 + 评估日志）
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
# ★ 推荐：best.pt + 去重 + TTA（最高 F1=0.715，计数误差 5.9%）
python src/inference.py \
  --model models/our_method/yolo11n-seg-v2/weights/best.pt \
  --source <images> \
  --output outputs/result \
  --mode ellipse \
  --yolo-iou 0.5 \
  --dedup-iou 0.10 \
  --dedup-dist 40 \
  --overlap 0.15 \
  --conf 0.34 \
  --tta

# 最高计数精度（last.pt，计数误差 3.3%）
python src/inference.py \
  --model models/our_method/yolo11n-seg-v2/weights/last.pt \
  --source <images> \
  --output outputs/result \
  --mode ellipse \
  --dedup-iou 0.15 \
  --tta

# 查看所有选项
python src/inference.py --help
```

关键参数说明：

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--yolo-iou` | 0.7 | 0.5 | YOLO 内部 NMS，降低可减少块内重复 |
| `--dedup-iou` | 0.15 | 0.10 | 去重 IoU 阈值（0=禁用） |
| `--dedup-dist` | 30 | 40 | 去重中心距离阈值（px） |
| `--overlap` | 0.2 | 0.15 | SAHI 分块重叠率，降低减少重复 |
| `--conf` | 0.3 | 0.34 | 置信度阈值 |
| `--tta` | 关闭 | 启用 | 翻转共识，提升 recall |

### 4. 评估

```bash
python src/eval.py \
  --pred outputs/result/labels \
  --gt dataset/data_v3_augmented/test/labels \
  --images dataset/data_v3_augmented/test/images
```

### 5. 模型自动对比

```bash
# 基础对比
python scripts/eval/compare_models.py

# 带去重 + TTA 的对比
python scripts/eval/compare_models.py --dedup-iou 0.15

# best.pt 参数网格搜索
python scripts/eval/sweep_best_pt.py
```

### 6. RF-DETR 自动标注（可选）

```bash
python scripts/inference/inference_rfdetr.py
```

RF-DETR v3 自动标注 recall ~85%，可用于辅助标注，但注意漏标噪声会影响训练质量，纯人工标注在精确计数任务上效果更优。

## 关键设计决策

- **中心距离+IoU 去重是关键**：SAHI 分块产生大量碎片化重复，仅靠 mask IoU NMS 不足以去除（碎片间 IoU 只有 0.1-0.4）。引入中心距离约束后，去重叠检测效果显著，F1 +0.03，计数误差从 30.4% 降至 5.9%
- **TTA 翻转共识**：原图+翻转图共识过滤，天然抑制 SAHI 碎片带来的假阳，同时恢复部分被提高的置信度阈值误杀的真检测
- **nano > small**：20 张训练数据下，nano (2.6M) F1=0.715 > small (9.4M) F1=0.628，小容量更抗过拟合
- **椭圆拟合优于原始分割**：ConvexHull + fitEllipse 消除阴影凹陷和碎片化，减少重复检测
- **纯人工标注优于混合标注**：自动标注的 ~15% 漏标率会显著拉低模型 precision
- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
