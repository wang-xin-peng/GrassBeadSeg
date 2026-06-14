# GrassBeadSeg

玻璃珠高密度粘连图像的实例分割项目。核心任务是对图像中每一颗玻璃珠生成独立的像素级掩膜，实现个体分离、轮廓提取与精确计数。适用于材料科学、工业质检及颗粒分析中的高通量形态学分析。

## 最优模型 — YOLOv11n-seg v3

| 指标          |        v2 best.pt        | **v3 last.pt + 优化** |
| ------------- | :----------------------: | :-------------------------: |
| 权重          | yolo11n-seg-v2 (best.pt) |  yolo11n-seg-v3 (last.pt)  |
| Recall@0.5    |          0.736          |       **0.766**       |
| Precision@0.5 |          0.695          |       **0.712**       |
| F1@0.5        |          0.715          |       **0.738**       |
| 计数误差      |           5.9%           |   **3.6%**（平衡）   |

**推荐方案**：

- 如需**最高 F1 / 检出率**（0.738）→ `last.pt` + ov=0.32 + conf=0.38 + TTA
- 如需**最高计数精度**（1.5% 误差）→ `last.pt` + ov=0.30 + conf=0.41 + TTA

v3 改进点：

- 训练数据：30 张人工标注（vs v2 的 20 张），data_v3 数据集
- 离线增强：20× per image，albumentations pipeline
- 在线增强：scale=0.9, degrees=15, shear=5, perspective, mixup, copy_paste
- 训练 200 epochs，AdamW + cosine lr
- v2 → v3：F1 +0.023，Recall +0.030，计数误差 -2.3pp

## 技术路线（grass_bead_seg）

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
│   ├── auto_label/
│   │   ├── train_rfdetr.py      # RF-DETR 训练
│   │   └── inference_rfdetr.py   # RF-DETR 自动标注推理
│   └── sweep/
│       ├── compare_models.py    # v1/v2 模型自动对比
│       ├── sweep_best_pt.py     # best.pt 去重参数网格搜索
│       ├── sweep_best_pt_v2.py  # best.pt 第二轮优化搜索
│       ├── sweep_v3_conf.py     # v3 置信度 + TTA 扫描
│       ├── sweep_v3_fine.py     # v3 五维精细调参
│       ├── sweep_v3_ov_conf.py  # v3 overlap × conf 交叉搜索
│       ├── sweep_v3_r3.py       # v3 大 overlap + soft-nms
│       └── sweep_v3_best.py     # v3 best.pt overlap 扫描
├── dataset/
│   ├── raw/                     # 40 张原始未标注图片
│   ├── data_v2/                 # 20 张人工标注（v2 训练集）
│   ├── data_v2_aug/             # v2 增强数据（20×）
│   ├── data_v3/                 # ★ 30 张人工标注（v3 训练集）
│   └── auto_labeled/            # RF-DETR 自动标注结果
├── models/
│   ├── grass_bead_seg/
│   │   ├── yolo11n-seg-baseline/    # nano 基线权重
│   │   ├── yolo11s-seg-baseline/    # small 基线权重
│   │   ├── yolo11n-seg-v1/          # v1 nano 权重
│   │   ├── yolo11s-seg-v1/          # v1 small 权重
│   │   ├── yolo11n-seg-v2/          # v2 nano 权重
│   │   ├── yolo11s-seg-v2/          # v2 small 权重
│   │   └── yolo11n-seg-v3/          # v3 nano 权重
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

关键参数说明：

| 参数             | 默认值 | 推荐值 | 说明                             |
| ---------------- | ------ | ------ | -------------------------------- |
| `--overlap`    | 0.2    | 0.32   | SAHI 分块重叠率，v3 需要更大重叠 |
| `--conf`       | 0.3    | 0.38   | 置信度阈值                       |
| `--tta`        | 关闭   | 启用   | 翻转共识，提升 F1 +0.015-0.02    |
| `--yolo-iou`   | 0.7    | 0.7    | v3 上已饱和，默认即可            |
| `--dedup-iou`  | 0.15   | 0.15   | v3 上已饱和，默认即可            |
| `--dedup-dist` | 30     | 30     | v3 上已饱和，默认即可            |

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

- **数据扩展是最有效的提升**：v2 (20 张) → v3 (30 张)，F1 +0.023，Recall +0.030，计数误差 -2.3pp。瓶颈仍在数据量
- **v3 需要更大 overlap**：v2 最优 overlap=0.15，v3 最优 overlap=0.32。v3 的冗余检测减少后，需要更大的分块重叠来覆盖边界珠子
- **dedup/dist/yolo_iou 对 v3 已饱和**：v3 训练质量提升后，SAHI 自身产生的重复检测大幅减少，dedup、NMS 参数在全范围内几乎无差异。Soft-NMS 同样无效果
- **TTA 仍然刚需**：关闭 TTA 直接掉 0.015-0.019 F1，翻转共识过滤机制仍然有效
- **last.pt > best.pt**：v3 上 last.pt 在所有维度（F1、Recall、计数精度）均优于 best.pt
- **nano > small**：30 张训练数据下，nano (2.6M) 比 small (9.4M) 更抗过拟合
- **椭圆拟合优于原始分割**：ConvexHull + fitEllipse 消除阴影凹陷和碎片化，减少重复检测
- **纯人工标注优于混合标注**：自动标注的 ~15% 漏标率会显著拉低模型 precision
- **降采样 NMS**：全分辨率 mask IoU 计算为 O(n²)，降采样到 ~480px 后速度从数分钟降至秒级

## 许可证

本项目使用 MIT 许可证。依赖的 RF-DETR 使用 Apache 2.0 许可证，详见 NOTICE 文件。
