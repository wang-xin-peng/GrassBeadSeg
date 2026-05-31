# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """玻璃珠检测脚本 - 基于官方文档"""

# import os
# import base64
# from dotenv import load_dotenv
# from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# # 加载 .env
# load_dotenv()

# API_KEY = os.getenv("ROBOFLOW_API_KEY")
# IMAGE_PATH = "dataset/11.png"
# MODEL_ID = "grassbeadseg/8"

# # 读取图片
# with open(IMAGE_PATH, "rb") as f:
#     image_base64 = base64.b64encode(f.read()).decode("utf-8")

# # 配置参数（明确写出来）
# config = InferenceConfiguration(
#     confidence_threshold=0.5,
#     max_detections=600,      # 目标600
#     iou_threshold=0.5
# )

# # 创建客户端并应用配置
# client = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key=API_KEY
# ).configure(config)  # 链式调用，直接应用配置

# # 推理
# result = client.infer(image_base64, model_id=MODEL_ID)

# print(f"检测到 {len(result['predictions'])} 个目标")

from inference import get_model

# 这行代码会自动下载并缓存模型权重到本地
model = get_model("grassbeadseg/8", api_key="Eef6s9abKXF4P9PRFR2d")

# 推理在本地执行，不经过云端
results = model.infer("dataset/11.png")[0]
print(f"检测到 {len(results.predictions)} 个目标")
