"""
导出训练好的模型为 ONNX 格式

用法（在 Windows 本地电脑运行）:
    python train/export_onnx.py
"""

import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, NUM_CLASSES, INPUT_SIZE

PTH_PATH = os.path.join(MODELS_DIR, "cat_classifier.pth")
ONNX_PATH = os.path.join(MODELS_DIR, "cat_classifier.onnx")


def load_model():
    """从 .pth 文件加载模型"""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )

    checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载模型: {PTH_PATH}")
    print(f"训练时最佳验证准确率: {checkpoint['best_val_acc']:.4f}")
    return model


def export_onnx(model):
    """导出为 ONNX 格式"""
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX 模型已导出: {ONNX_PATH}")


def verify_onnx(model):
    """验证 ONNX 模型与 PyTorch 模型输出一致"""
    # 检查 ONNX 模型结构
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型结构验证通过")

    # 对比输出
    import onnxruntime as ort

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        pt_output = model(dummy).numpy()

    session = ort.InferenceSession(ONNX_PATH)
    ort_output = session.run(None, {"input": dummy.numpy()})[0]

    diff = np.max(np.abs(pt_output - ort_output))
    print(f"PyTorch vs ONNX 最大输出差异: {diff:.6e}")
    if diff < 1e-5:
        print("输出一致性验证通过")
    else:
        print("[警告] 输出差异较大，请检查模型")

    # 文件大小
    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"ONNX 模型大小: {size_mb:.1f} MB")


def main():
    if not os.path.exists(PTH_PATH):
        print(f"[错误] 未找到训练好的模型: {PTH_PATH}")
        print("请先运行 python train/train_classifier.py 进行训练")
        sys.exit(1)

    model = load_model()
    export_onnx(model)

    try:
        verify_onnx(model)
    except ImportError:
        print("[提示] 安装 onnxruntime 可验证输出一致性: pip install onnxruntime")

    print("\n下一步: 将 models/cat_classifier.onnx 复制到树莓派的 Cat/models/ 目录下")


if __name__ == "__main__":
    main()
