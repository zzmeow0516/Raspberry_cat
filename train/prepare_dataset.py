"""
数据预处理与加载

用法：被 train_classifier.py 导入，不单独运行
"""

import os
import sys

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_DIR, CLASS_NAMES, INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, TRAIN_SPLIT,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan_dataset():
    """扫描 dataset/ 目录，返回 (图片路径列表, 标签列表)"""
    paths, labels = [], []
    for idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[警告] 目录不存在: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return paths, labels


class CatDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# 训练集增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 验证集只做 resize + 归一化
val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders():
    """返回 (train_loader, val_loader, class_counts)"""
    paths, labels = _scan_dataset()
    if len(paths) == 0:
        raise FileNotFoundError(
            f"在 {DATASET_DIR} 下未找到任何图片。\n"
            f"请将猫咪照片放入 cat_a/, cat_b/, unknown/ 子目录中。"
        )

    # 统计各类数量
    class_counts = {}
    for lbl in labels:
        name = CLASS_NAMES[lbl]
        class_counts[name] = class_counts.get(name, 0) + 1
    print(f"数据集统计: {class_counts}，共 {len(paths)} 张图片")

    # 分层划分训练/验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=42,
    )
    print(f"训练集: {len(train_paths)} 张, 验证集: {len(val_paths)} 张")

    train_ds = CatDataset(train_paths, train_labels, train_transform)
    val_ds = CatDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_counts
