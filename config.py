"""共享配置"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 数据集 ──
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── 分类标签 ──
CLASS_NAMES = ["cat_a", "cat_b", "unknown"]
NUM_CLASSES = len(CLASS_NAMES)

# ── 模型参数 ──
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── 训练参数 ──
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
EARLY_STOP_PATIENCE = 7

# ── 推理参数 ──
YOLO_CONFIDENCE = 0.5       # YOLOv8 猫检测置信度阈值
CLASSIFIER_CONFIDENCE = 0.8  # 分类器置信度阈值
COCO_CAT_CLASS = 15          # COCO 数据集中 cat 的类别编号

# ── 舵机参数 ──
SERVO_GPIO_PIN = 18          # GPIO18 (物理 Pin 12)
SERVO_OPEN_DURATION = 1.5    # 正转时长（秒）
SERVO_CLOSE_DURATION = 1.5   # 反转时长（秒）
COOLDOWN_SECONDS = 5       # 冷却时间（秒），防止重复触发

# ── 摄像头参数 ──
CAMERA_RESOLUTION = (320, 240)
DETECTION_INTERVAL = 1.0     # 检测间隔（秒）
