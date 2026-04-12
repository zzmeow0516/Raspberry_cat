"""
猫检测模块 — 使用 YOLOv8n ONNX 模型检测画面中的猫
（纯 onnxruntime 推理，无需 torch/ultralytics）
"""

import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, YOLO_CONFIDENCE, COCO_CAT_CLASS

ONNX_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")
INPUT_SIZE = 640


class CatDetector:
    def __init__(self):
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                f"未找到检测模型: {ONNX_PATH}\n"
                "请在 Windows 电脑上运行 python train/export_yolo_onnx.py，"
                "然后将 models/yolov8n.onnx 复制到树莓派"
            )
        self.session = ort.InferenceSession(ONNX_PATH)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[检测器] YOLOv8n ONNX 已加载: {ONNX_PATH}")

    def _preprocess(self, frame: np.ndarray):
        """BGR 图像 → YOLOv8 输入 tensor，返回 (tensor, scale, pad)"""
        h, w = frame.shape[:2]

        # letterbox：等比缩放到 640x640，不足处填灰
        scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_top = (INPUT_SIZE - new_h) // 2
        pad_left = (INPUT_SIZE - new_w) // 2

        from PIL import Image
        img = Image.fromarray(frame[:, :, ::-1])  # BGR→RGB
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = np.array(img)

        # HWC → NCHW, 归一化到 [0, 1]
        tensor = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return tensor, scale, pad_top, pad_left

    def _postprocess(self, output, scale, pad_top, pad_left, orig_h, orig_w):
        """
        YOLOv8 ONNX 输出: [1, 84, 8400]
        84 = 4 (cx,cy,w,h) + 80 classes
        """
        pred = output[0].transpose(1, 0)  # [8400, 84]
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        # 取最大类别得分
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # 只保留猫（class 15）且置信度达标
        mask = (class_ids == COCO_CAT_CLASS) & (confidences >= YOLO_CONFIDENCE)
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2（letterbox 坐标系）
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 还原到原始图像坐标
        x1 = np.clip((x1 - pad_left) / scale, 0, orig_w).astype(int)
        y1 = np.clip((y1 - pad_top) / scale, 0, orig_h).astype(int)
        x2 = np.clip((x2 - pad_left) / scale, 0, orig_w).astype(int)
        y2 = np.clip((y2 - pad_top) / scale, 0, orig_h).astype(int)

        # NMS
        indices = self._nms(x1, y1, x2, y2, confidences)

        return [
            {"box": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
             "confidence": float(confidences[i])}
            for i in indices
        ]

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_threshold=0.45):
        """简单 NMS"""
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_threshold]
        return keep

    def detect(self, frame: np.ndarray) -> list:
        """
        检测图像中的猫

        Args:
            frame: BGR 图像 (numpy array)

        Returns:
            [{"box": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        orig_h, orig_w = frame.shape[:2]
        tensor, scale, pad_top, pad_left = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})
        return self._postprocess(output[0], scale, pad_top, pad_left, orig_h, orig_w)
