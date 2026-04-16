"""
猫咪识别喂食器 — 主程序

用法:
    python rpi/main.py               # 无预览（纯检测）
    python rpi/main.py --preview     # 开启摄像头实时预览窗口

流程:
    摄像头抓帧 → YOLOv8n 检测猫 → 裁剪猫区域 → MobileNetV2 识别 → 控制舵机喂食
"""

import argparse
import signal
import sys
import time
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_RESOLUTION, DETECTION_INTERVAL

from cat_detector import CatDetector
from cat_classifier import CatClassifier
from servo import ServoController

# 各类别对应的 BGR 颜色
LABEL_COLORS = {
    "cat_a":   (0, 255, 0),    # 绿色
    "cat_b":   (255, 128, 0),  # 蓝橙色
    "unknown": (0, 0, 255),    # 红色
}


def init_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
    ))
    controls={"FrameRate": 50}
    cam.start()
    time.sleep(2)
    print(f"[摄像头] 已启动 ({CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]})")
    return cam


def crop_region(frame: np.ndarray, box: list) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def draw_detections(frame_bgr, cats_with_labels):
    """在 BGR 图像上绘制检测框和标签"""
    import cv2
    vis = frame_bgr.copy()
    for cat, label, conf in cats_with_labels:
        x1, y1, x2, y2 = cat["box"]
        color = LABEL_COLORS.get(label, (128, 128, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        # 标签背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="开启实时预览窗口")
    args = parser.parse_args()

    if args.preview:
        try:
            import cv2
        except ImportError:
            print("[错误] 预览模式需要 OpenCV，请先运行: sudo apt install python3-opencv")
            sys.exit(1)

    print("=" * 50)
    print("猫咪识别喂食器")
    print("=" * 50)

    camera = init_camera()
    detector = CatDetector()
    classifier = CatClassifier()
    servo = ServoController()

    if args.preview:
        import cv2
        cv2.namedWindow("Cat Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cat Monitor", CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1])
        print("[预览] 按 Q 键退出")

    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n正在关闭...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n开始检测... (Ctrl+C 退出)\n")

    try:
        while running:
            frame_rgb = camera.capture_array()
            frame_bgr = frame_rgb[:, :, ::-1].copy()

            cats = detector.detect(frame_bgr)
            cats_with_labels = []

            if cats:
                print(f"[检测] 发现 {len(cats)} 只猫")
                for i, cat in enumerate(cats):
                    crop = crop_region(frame_bgr, cat["box"])
                    if crop.size == 0:
                        continue
                    label, conf = classifier.classify(crop)
                    cats_with_labels.append((cat, label, conf))
                    print(
                        f"  猫#{i+1}: {label} "
                        f"(识别: {conf:.2f}, 检测: {cat['confidence']:.2f})"
                    )
                    if label in ("cat_a", "cat_b"):
                        if servo.is_cooling_down:
                            print(f"  -> 冷却中 (剩余 {servo.cooldown_remaining:.0f}s)")
                        else:
                            print(f"  -> 确认是自家猫 [{label}]，触发喂食！")
                            servo.trigger_feed()

            if args.preview:
                import cv2
                vis = draw_detections(frame_bgr, cats_with_labels)
                cv2.imshow("Cat Monitor", vis)
                # 按 Q 退出，waitKey 1ms 不阻塞主循环
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(DETECTION_INTERVAL)

    finally:
        print("\n清理资源...")
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
        servo.cleanup()
        camera.stop()
        print("已退出")


if __name__ == "__main__":
    main()
