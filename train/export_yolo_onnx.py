"""
导出 YOLOv8n 为 ONNX 格式（在 Windows 本地电脑运行）

用法:
    pip install ultralytics
    python train/export_yolo_onnx.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR

OUTPUT_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")


def main():
    from ultralytics import YOLO

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("下载并导出 YOLOv8n 为 ONNX...")
    model = YOLO("yolov8n.pt")  # 自动下载 yolov8n.pt
    model.export(format="onnx", imgsz=640, opset=13)

    # ultralytics 导出到 yolov8n.onnx（同目录），移动到 models/
    import shutil
    src = "yolov8n.onnx"
    if os.path.exists(src):
        shutil.move(src, OUTPUT_PATH)

    print(f"已导出: {OUTPUT_PATH}")
    print("下一步: 将 models/yolov8n.onnx 复制到树莓派的 cat/models/ 目录")


if __name__ == "__main__":
    main()
