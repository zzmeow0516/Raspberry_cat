dataset，图片放在本地windows电脑跑模型，跑完的数据传回rasp.

Windows 电脑上（如果还没装 ultralytics）：
pip install ultralytics
python train/train_classifier.py
python train/export_yolo_onnx.py

会在 models/ 下生成 yolov8n.onnx，把它复制到树莓派的 Cat/models/

rasp:
cd work/Cat/

sudo pigpiod

source  venv/bin/activate

python rpi/main.py

-------------------------
项目结构

     Cat/
     ├── train/                       # 【本地电脑运行】训练代码
     │   ├── requirements.txt         # pip install -r requirements.txt
     │   ├── prepare_dataset.py       # 数据预处理与增强
     │   ├── train_classifier.py      # MobileNetV2 迁移学习训练
     │   └── export_onnx.py           # 导出 ONNX 模型
     ├── dataset/                     # 【用户手动放入】猫咪照片
     │   ├── cat_a/                   # 猫咪A 照片 100-200 张
     │   ├── cat_b/                   # 猫咪B 照片 100-200 张
     │   └── unknown/                 # 陌生猫/其他猫 50-100 张
     ├── models/                      # 训练好的模型文件
     ├── rpi/                         # 【树莓派运行】推理+控制代码
     │   ├── requirements.txt         # 树莓派依赖
     │   ├── cat_detector.py          # YOLOv8n 猫检测
     │   ├── cat_classifier.py        # MobileNetV2 猫识别
     │   ├── servo.py                 # SG90 舵机控制
     │   └── main.py                  # 主程序入口
     └── config.py                    # 共享配置（GPIO引脚、阈值等）

-------------------------
     硬件接线

     ┌─────────────────┬─────────────────────┐
     │      连接       │        引脚         │
     ├─────────────────┼─────────────────────┤
     │ 舵机红线 (VCC)  │ RPi 5V (Pin 2)      │
     ├─────────────────┼─────────────────────┤
     │ 舵机棕线 (GND)  │ RPi GND (Pin 6)     │
     ├─────────────────┼─────────────────────┤
     │ 舵机橙线 (信号) │ RPi GPIO18 (Pin 12) │
     ├─────────────────┼─────────────────────┤
     │ CSI 摄像头      │ CSI 排线接口        │
     └─────────────────┴─────────────────────┘
