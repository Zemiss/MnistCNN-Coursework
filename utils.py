import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def save_metrics(metrics, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_curves(metrics, loss_path: Path, accuracy_path: Path) -> None:
    df = pd.DataFrame(metrics)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST CNN Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["test_acc"], marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST CNN Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.close()


def _markdown_metrics_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in ["train_loss", "train_acc", "test_loss", "test_acc"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    columns = list(display_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _trend_text(df: pd.DataFrame) -> tuple[str, str, str]:
    loss_down = df["test_loss"].iloc[-1] < df["test_loss"].iloc[0]
    acc_up = df["test_acc"].iloc[-1] > df["test_acc"].iloc[0]
    overfit_gap = df["train_acc"].iloc[-1] - df["test_acc"].iloc[-1]

    loss_text = "整体呈下降趋势" if loss_down else "未表现出稳定下降趋势"
    acc_text = "整体呈上升趋势" if acc_up else "未表现出稳定上升趋势"
    if overfit_gap > 0.05:
        overfit_text = "训练准确率明显高于测试准确率，存在一定过拟合迹象"
    else:
        overfit_text = "训练准确率与测试准确率差距较小，未观察到明显过拟合迹象"

    return loss_text, acc_text, overfit_text


def generate_report(metrics_csv: Path, report_path: Path, config: dict) -> None:
    df = pd.read_csv(metrics_csv)
    final_test_acc = df["test_acc"].iloc[-1]
    loss_text, acc_text, overfit_text = _trend_text(df)

    content = f"""# MNIST 手写数字分类实验报告

## 1. 实验目的

使用 PyTorch 和 torchvision 构建一个小型 CNN 模型，对 MNIST 手写数字数据集进行 0-9 十分类实验，并记录训练过程中的损失和准确率变化。

## 2. 数据集介绍

MNIST 数据集包含灰度手写数字图像，每张图像大小为 1 x 28 x 28。训练集包含 60000 张图像，测试集包含 10000 张图像，类别为数字 0 到 9。

## 3. 数据预处理方式

数据通过 `torchvision.datasets.MNIST` 下载和加载。预处理流程为：

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 4. CNN 网络结构

| 层次 | 结构 | 输出说明 |
| --- | --- | --- |
| Input | 1 x 28 x 28 | 灰度图像输入 |
| Conv1 | Conv2d(1, 16, kernel_size=3, padding=1) + ReLU + MaxPool2d(2) | 16 x 14 x 14 |
| Conv2 | Conv2d(16, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2) | 32 x 7 x 7 |
| Flatten | Flatten | 1568 |
| FC1 | Linear(32 * 7 * 7, 128) + ReLU | 128 |
| FC2 | Linear(128, 10) | 10 类 logits |

## 5. 训练参数

| 参数 | 取值 |
| --- | --- |
| batch_size | {config["batch_size"]} |
| epochs | {config["epochs"]} |
| learning_rate | {config["learning_rate"]} |
| optimizer | Adam |
| loss function | CrossEntropyLoss |
| device | {config["device"]} |
| seed | {config["seed"]} |

## 6. 损失函数说明

实验使用 `nn.CrossEntropyLoss()` 作为损失函数。模型最后一层直接输出 logits，不在 `forward` 中使用 `softmax` 或 `log_softmax`。

## 7. 训练结果表格

{_markdown_metrics_table(df)}

## 8. 训练曲线

![Loss Curve](loss_curve.png)

![Accuracy Curve](accuracy_curve.png)

## 9. 简短结果分析

- Loss：{loss_text}。
- Accuracy：{acc_text}。
- 最终测试准确率：{final_test_acc:.4f}。
- 过拟合迹象：{overfit_text}。

## 10. 实验总结

本实验使用一个小型 CNN 完成 MNIST 手写数字分类任务。训练过程记录了每个 epoch 的训练集和测试集损失、准确率，并自动保存模型、指标文件、曲线图和实验报告草稿，便于后续复现实验和撰写正式报告。
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
