"""
猫识别模块 — 使用 ONNX 模型区分自家猫咪
"""

import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, CLASS_NAMES, INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, CLASSIFIER_CONFIDENCE,
)

ONNX_PATH = os.path.join(MODELS_DIR, "cat_classifier.onnx")


class CatClassifier:
    def __init__(self):
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                f"未找到分类模型: {ONNX_PATH}\n"
                "请先在本地电脑训练模型，然后将 cat_classifier.onnx 复制到 models/ 目录"
            )
        self.session = ort.InferenceSession(ONNX_PATH)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[分类器] ONNX 模型已加载: {ONNX_PATH}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """将裁剪后的猫图像预处理为模型输入"""
        # BGR -> RGB -> PIL -> resize
        img = Image.fromarray(image[:, :, ::-1]).resize((INPUT_SIZE, INPUT_SIZE))
        # 转 numpy, 归一化
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        # HWC -> CHW -> NCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
        return arr.astype(np.float32)

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """
        识别裁剪后的猫图像

        Args:
            crop: BGR 图像 (裁剪后的猫区域)

        Returns:
            (类别名, 置信度) — 置信度低于阈值时返回 ("unknown", 置信度)
        """
        input_data = self._preprocess(crop)
        outputs = self.session.run(None, {self.input_name: input_data})[0]

        # softmax
        exp = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp / exp.sum()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = CLASS_NAMES[idx]

        # 置信度不够高时归为 unknown
        if conf < CLASSIFIER_CONFIDENCE:
            return "unknown", conf

        return label, conf
