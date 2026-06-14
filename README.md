# GrassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 最终效果

使用的预训练模型：YOLOv11n-seg

| 指标          |   v3 last.pt + 优化   |
| ------------- | :--------------------: |
| Recall@0.5    |    **0.766**    |
| Precision@0.5 |    **0.712**    |
| F1@0.5        |    **0.738**    |
| 计数误差      | **3.6%**（平衡） |

**推荐方案**：

- 如需**最高 F1 / 检出率**（0.738）→ `last.pt` + overlap=0.32 + conf=0.38 + TTA
- 如需**最高计数精度**（1.5% 误差）→ `last.pt` + overlap=0.30 + conf=0.41 + TTA

v3 训练配置：

- 数据集：30 张人工标注（train=26, valid=2, test=2）
- 离线增强：20× per image，albumentations pipeline
- 在线增强：scale=0.9, degrees=15, shear=5, perspective, mixup, copy_paste
- 模型：YOLOv11n-seg (2.6M)，COCO 预训练
- 训练：200 epochs，AdamW + cosine lr，batch=16

## 推理流水线

```
原始图像 (1920×1080)
       │
       ▼
┌  阶段一: SAHI 分块推理  ──────────────────────────
│  滑动窗口 640×640, overlap=0.32
│  YOLOv11n-seg (2.6M) 逐块推理
│  输出 → polygon mask 集合
└─────────────────────────────────────────────────
       │
       ▼
┌  阶段二: 去重  ──────────────────────────────────
│  ①  NMS        mask IoU 抑制重叠框 (iou=0.7)
│  ②  二次去重    中心距离 + IoU 过滤碎片 (dist<40, iou>0.10)
│  ③  TTA        水平翻转共识
└─────────────────────────────────────────────────
       │
       ▼
┌  阶段三: 椭圆拟合  ─────────────────────────────
│  ④  findContours 提取轮廓
│  ⑤  convexHull   凸包填充阴影凹陷
│  ⑥  fitEllipse   光滑椭圆拟合
│  输出 → 椭圆掩膜 (.txt, YOLO 格式)
└─────────────────────────────────────────────────
```

### 关键参数

| 阶段 | 参数               | 默认     | v3 推荐           | 说明                        |
| ---- | ------------------ | -------- | ----------------- | --------------------------- |
| SAHI | `--overlap`      | 0.2      | **0.32**    | 分块重叠率，v3 需要更大覆盖 |
| SAHI | `--conf`         | 0.3      | **0.38**    | 置信度阈值                  |
| 去重 | `--yolo-iou`     | 0.7      | 0.7               | NMS IoU，v3 已饱和          |
| 去重 | `--dedup-iou`    | 0.15     | 0.15              | 二次去重 IoU，v3 已饱和     |
| 去重 | `--dedup-dist`   | 30       | 30                | 中心距离阈值 (px)           |
| 去重 | `--tta`          | 关       | **开**      | 水平翻转共识                |
| 椭圆 | `--mode ellipse` | baseline | **ellipse** | 启用椭圆拟合                |

### 设计说明

- **SAHI 的必要性**: 1920×1080 直接推理显存不足，分块后逐块处理。但相邻分块会产生同一珠子的碎片化重复检测
- **两层去重**: NMS 抑制重叠框 → 中心距离+IoU 二次去重消除跨块碎片。v3 训练质量提升后，去重参数已接近饱和
- **凸包 + 椭圆**: YOLO 输出的多边形受阴影/反光影响呈锯齿状甚至裂块。convexHull 填平凹陷，fitEllipse 还原玻璃珠的圆形本质
- **TTA 稳定有效**: 原图与水平翻转图各推理一次，只保留两边共识的预测，稳定提升 recall 2-3pp，不可省略
- **last.pt > best.pt**: v3 上 last.pt 在所有维度均优于 best.pt

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
│   ├── auto_label/
│   │   ├── train_rfdetr.py      # RF-DETR 训练
│   │   └── inference_rfdetr.py   # RF-DETR 自动标注推理
│   └── sweep/
│       ├── sweep_v3_conf.py     # v3 置信度 + TTA 扫描
│       ├── sweep_v3_fine.py     # v3 五维精细调参
│       ├── sweep_v3_ov_conf.py  # v3 overlap × conf 交叉搜索
│       ├── sweep_v3_r3.py       # v3 大 overlap + soft-nms
│       └── sweep_v3_best.py     # v3 best.pt overlap 扫描
├── dataset/
│   ├── raw/                     # 40 张原始未标注图片
│   ├── data_v3/                 # ★ 30 张人工标注（v3 训练集）
│   ├── data_v3_aug/             # v3 增强数据
│   └── auto_labeled/            # RF-DETR 自动标注结果
├── models/
│   ├── grass_bead_seg/
│   │   └── yolo11n-seg-v3/          # ★ v3 最优模型
│   └── rfdetr_seg_large/        # RF-DETR 权重及预训练模型
├── outputs/                     # 推理输出
├── logs/                        # 训练与评估日志
└── requirements.txt
```

## 环境配置

```bash
conda create -n gbseg python=3.10 -y
conda activate gbseg
pip install -r requirements.txt
```

## 使用流程

### 1. 数据准备

**方式一：使用已有数据集 data_v3**

直接使用 `dataset/data_v3/`（30 张人工标注，train=26/valid=2/test=2）。

**方式二：从头生成增强数据**

```bash
python scripts/data_augmentation/augment_dataset.py \
  --input dataset/data_v3 --output dataset/data_v3_aug --augments-per-image 20
```

### 2. 训练

```bash
# 从头训练 v3 nano
python src/train.py --data dataset/data_v3_aug --model n --name v3

# 离线环境可以加 --no-amp
python src/train.py --data dataset/data_v3_aug --model n --name v3 \
  --batch 16 --workers 16 --device 0 --no-amp
```

### 3. 推理

```bash
# ★ 推荐：v3 last.pt + TTA（最高 F1=0.738）
python src/inference.py \
  --model models/grass_bead_seg/yolo11n-seg-v3/weights/last.pt \
  --source <images> \
  --output outputs/result \
  --mode ellipse \
  --overlap 0.32 \
  --conf 0.38 \
  --tta

# 最高计数精度（计数误差 1.5%）
python src/inference.py \
  --model models/grass_bead_seg/yolo11n-seg-v3/weights/last.pt \
  --source <images> \
  --output outputs/result \
  --mode ellipse \
  --overlap 0.30 \
  --conf 0.41 \
  --tta

# 查看所有选项
python src/inference.py --help
```

参数说明见上方 [推理流水线](#推理流水线) 的关键参数表。

### 4. 评估

```bash
python src/eval.py \
  --pred outputs/result/labels \
  --gt dataset/data_v3/test/labels \
  --images dataset/data_v3/test/images
```

### 5. 模型自动对比 / 参数搜索

```bash
# v3 置信度 + TTA 扫描
python scripts/sweep/sweep_v3_conf.py

# v3 五维精细调参
python scripts/sweep/sweep_v3_fine.py

# v3 overlap × conf 交叉搜索
python scripts/sweep/sweep_v3_ov_conf.py

# v3 大 overlap + soft-nms
python scripts/sweep/sweep_v3_r3.py

# v3 best.pt overlap 扫描
python scripts/sweep/sweep_v3_best.py
```

### 6. RF-DETR 自动标注

```bash
python scripts/auto_label/inference_rfdetr.py
```

RF-DETR 自动标注，用于辅助标注。

## 关键设计决策

- **数据量是核心瓶颈**：30 张人工标注下，nano (2.6M) 比 small (9.4M) 更抗过拟合，进一步扩展数据量仍有提升空间
- **v3 需要大 overlap**：最优 overlap=0.32，较默认值 0.2 大幅提升，更大的分块重叠可覆盖边界珠子
- **dedup/dist/yolo_iou 已饱和**：训练质量提升后，SAHI 自身产生的重复检测大幅减少，去重参数在全范围内几乎无差异
- **TTA 仍然刚需**：关闭 TTA 直接掉 0.015-0.019 F1，不可省略
- **last.pt > best.pt**：last.pt 在所有维度（F1、Recall、计数精度）均优于 best.pt
- **椭圆拟合优于原始分割**：ConvexHull + fitEllipse 消除阴影凹陷和碎片化，减少重复检测
- **纯人工标注优于混合标注**：自动标注的 ~15% 漏标率会显著拉低模型 precision
- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
