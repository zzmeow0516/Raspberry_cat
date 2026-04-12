"""
MobileNetV2 迁移学习 — 猫咪分类器训练

用法（在 Windows 本地电脑运行）:
    cd Cat
    pip install -r train/requirements.txt
    python train/train_classifier.py
"""

import os
import sys
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, NUM_CLASSES, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
)
from train.prepare_dataset import get_dataloaders

DEVICE = torch.device("cpu")  # Windows Intel GPU 用 CPU 训练
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "cat_classifier.pth")


def build_model():
    """加载预训练 MobileNetV2，替换分类头"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # 冻结特征提取层
    for param in model.features.parameters():
        param.requires_grad = False

    # 替换分类头
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def plot_history(train_losses, val_losses, train_accs, val_accs):
    """保存训练曲线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")

    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=100)
    print(f"训练曲线已保存: {save_path}")
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 50)
    print("猫咪分类器训练")
    print("=" * 50)

    # 加载数据
    train_loader, val_loader, class_counts = get_dataloaders()

    # 构建模型
    model = build_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型参数: 总计 {total:,}, 可训练 {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> 新的最佳模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  -> Early stopping (连续 {EARLY_STOP_PATIENCE} 轮无提升)")
                break

    # 保存最佳模型
    if best_model_state is not None:
        torch.save({
            "model_state_dict": best_model_state,
            "class_names": ["cat_a", "cat_b", "unknown"],
            "best_val_acc": best_val_acc,
        }, MODEL_SAVE_PATH)
        print(f"\n最佳模型已保存: {MODEL_SAVE_PATH}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
    else:
        print("\n[错误] 没有保存任何模型")

    # 绘制训练曲线
    plot_history(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
