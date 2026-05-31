# GrassBeadSeg

一个面向高密度粘连玻璃珠图像的实例分割项目，旨在解决玻璃珠密集堆积、相互接触及遮挡场景下的个体级解析问题。

项目核心任务是为图像中每一颗玻璃珠生成独立的像素级掩膜与空间定位，实现从粘连区域中精确分离个体边界，并支持端到端的自动检测与计数。该任务区别于语义分割的类别级标注，专注于密集场景下的个体识别、轮廓提取与数量统计，适用于材料科学、工业质检及颗粒分析等领域的高通量玻璃珠形态学分析。

## 项目结构

```
GrassBeadSeg/
├── dataset/
│   ├── raw/                      # 40张原始未标注图片
│   ├── data_v1/                  # Roboflow标注数据集（10张）
│   │   ├── train/ (6张)
│   │   ├── valid/ (2张)
│   │   ├── test/  (2张)
│   │   └── data.yaml
│   ├── data_v1_augmented/        # 数据增强后的数据集
│   │   ├── train/ (66张)
│   │   ├── valid/ (2张)
│   │   └── test/  (2张)
│   └── auto_labeled/             # 自动标注输出
│       ├── images/
│       ├── labels/
│       └── data.yaml
├── models/
│   ├── download_pretrained.py     # 预训练权重下载脚本
│   ├── rfdetr_seg_large_pretrained.pth  # 下载的预训练权重
│   └── trained/                  # 训练输出检查点
│       ├── checkpoint_best_total.pth
│       └── ...
├── scripts/
│   ├── data_augmentation/
│   │   └── augment_dataset.py    # 数据增强
│   ├── train/
│   │   └── train_rfdetr.py       # RF-DETR 训练脚本
│   └── inference/
│       └── inference_rfdetr.py   # RF-DETR 自动标注脚本
├── NOTICE                        # Apache 2.0 组件声明
├── LICENSE                       # MIT 许可证
└── README.md
```

## 许可证

本项目本身使用 **MIT 许可证**（详见 LICENSE 文件）。

本项目使用了 [RF-DETR](https://github.com/roboflow/rf-detr)（Apache 2.0 许可证）进行模型训练和推理。
详见 NOTICE 文件中的 Attribution 声明。

## 环境准备

```bash
# 激活 conda 环境
conda activate gbseg

# 安装 RF-DETR（含训练依赖）
pip install rfdetr

# 如需 TensorBoard 日志
pip install "rfdetr[metrics]"
```

## 使用流程

### 1. 下载预训练权重（需联网）

```bash
E:\conda_envs\gbseg\python.exe models\download_pretrained.py
```

### 2. 训练模型（在服务器上运行）

```bash
E:\conda_envs\gbseg\python.exe scripts\train\train_rfdetr.py
```

训练参数可通过脚本顶部的常量调整。
训练默认启用 early stopping，mAP 连续 30 轮无提升将自动停止。

### 3. 自动标注（在服务器上运行）

```bash
E:\conda_envs\gbseg\python.exe scripts\inference\inference_rfdetr.py
```

读取 `dataset/raw/` 中的 40 张原始图片，输出自动标注到 `dataset/auto_labeled/`。

## 工作流说明

1. **本地机器（有网络）**：运行 `download_pretrained.py` 下载预训练权重
2. **传输到服务器**：将整个项目（含 `models/rfdetr_seg_large_pretrained.pth`）复制到服务器
3. **服务器（可 pip 安装，无法访问外网）**：
   - 安装依赖：`pip install rfdetr`
   - 运行训练：`python scripts/train/train_rfdetr.py`
   - 运行推理：`python scripts/inference/inference_rfdetr.py`