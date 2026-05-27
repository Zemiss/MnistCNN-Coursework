import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch: int,
    log_interval: int,
    train_losses: list,
    train_counter: list,
    checkpoint_path: Path,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        running_loss += loss.item() * batch_size
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        total += batch_size

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * train_loader.batch_size)
                + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, test_loader, device, test_losses=None):
    model.eval()
    test_loss = 0.0
    correct = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        test_loss += F.nll_loss(output, target, reduction="sum").item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    if test_losses is not None:
        test_losses.append(test_loss)

    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * test_acc,
        )
    )
    return test_loss, test_acc


def save_metrics(metrics, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_loss_curve(train_counter, train_losses, test_counter, test_losses, output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("MNIST Training Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_curve(metrics, output_path: Path) -> None:
    df = pd.DataFrame(metrics)
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["test_acc"], marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_ground_truth_examples(example_data, example_targets, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(output_path, dpi=150)
    plt.close()


@torch.no_grad()
def save_prediction_examples(model, example_data, device, output_path: Path) -> None:
    model.eval()
    output = model(example_data.to(device))
    predictions = output.data.max(1, keepdim=True)[1].cpu()

    plt.figure(figsize=(7, 4))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Prediction: {}".format(predictions[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(output_path, dpi=150)
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


def generate_report(metrics_csv: Path, report_path: Path, config: dict) -> None:
    df = pd.read_csv(metrics_csv)
    final_test_acc = df["test_acc"].iloc[-1]
    loss_down = df["test_loss"].iloc[-1] < df["test_loss"].iloc[0]
    acc_up = df["test_acc"].iloc[-1] > df["test_acc"].iloc[0]
    overfit_gap = df["train_acc"].iloc[-1] - df["test_acc"].iloc[-1]

    loss_text = "测试集 loss 整体下降" if loss_down else "测试集 loss 未呈现稳定下降"
    acc_text = "测试集 accuracy 整体上升" if acc_up else "测试集 accuracy 未呈现稳定上升"
    overfit_text = (
        "训练准确率明显高于测试准确率，存在一定过拟合迹象"
        if overfit_gap > 0.05
        else "训练准确率与测试准确率差距较小，未观察到明显过拟合迹象"
    )

    content = f"""# MNIST 手写数字识别 CNN 实验报告

## 1. 实验目的

本实验参考 PyTorch Tutorial 中的 ConvNet MNIST 示例，使用 PyTorch 和 torchvision 在 MNIST 数据集上训练一个卷积神经网络，完成 0-9 手写数字十分类任务，并自动生成实验报告所需的指标、曲线图和模型文件。

## 2. 数据构成

MNIST 数据集包含 70000 张灰度手写数字图像，其中训练集 60000 张，测试集 10000 张。每张图像大小为 1 x 28 x 28，标签为 0-9 中的一个数字。

本实验使用 `torchvision.datasets.MNIST` 直接下载和加载数据：

- 训练集：`train=True`
- 测试集：`train=False`
- 训练集 DataLoader：`shuffle=True`
- 测试集 DataLoader：`shuffle=True`

## 3. 数据预处理

图像先转换为 tensor，再使用 MNIST 全局均值和标准差进行标准化：

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 4. 网络结构

本实验采用教程中的 `ConvNet` 结构。网络包含两个卷积层和两个全连接层，最后输出 10 个类别的 log probability。

| 层次 | 结构 | 输出尺寸 |
| --- | --- | --- |
| Input | MNIST 灰度图像 | 1 x 28 x 28 |
| Conv1 | Conv2d(1, 10, kernel_size=5) + ReLU + MaxPool2d(2, 2) | 10 x 12 x 12 |
| Conv2 | Conv2d(10, 20, kernel_size=3) + ReLU | 20 x 10 x 10 |
| Flatten | view(batch_size, -1) | 2000 |
| FC1 | Linear(20 * 10 * 10, 500) + ReLU | 500 |
| FC2 | Linear(500, 10) | 10 |
| Output | log_softmax(dim=1) | 10 类 log probability |

## 5. 训练方式与参数

| 参数 | 取值 |
| --- | --- |
| batch_size | {config["batch_size"]} |
| epochs | {config["epochs"]} |
| optimizer | Adam |
| loss function | NLLLoss |
| log_interval | {config["log_interval"]} |
| random_seed | {config["random_seed"]} |
| device | {config["device"]} |

训练时每个 epoch 执行一次完整训练集迭代，然后在测试集上评估平均损失和准确率。训练过程会保存模型参数、优化器参数和 checkpoint。

## 6. 损失函数

模型最后一层使用：

```python
F.log_softmax(out, dim=1)
```

因此损失函数使用负对数似然损失：

```python
F.nll_loss(output, target)
```

测试阶段使用 `torch.no_grad()`，避免保存计算图，从而降低内存开销。

## 7. 初始测试结果

训练前，随机初始化模型在测试集上的表现为：

- initial test loss: {config["initial_test_loss"]:.4f}
- initial test accuracy: {config["initial_test_acc"]:.4f}

## 8. 训练结果

{_markdown_metrics_table(df)}

## 9. 曲线图与样例图

### 样例真实标签

![Ground Truth Examples](sample_ground_truth.png)

### Loss 变化曲线

![Loss Curve](loss_curve.png)

### Accuracy 变化曲线

![Accuracy Curve](accuracy_curve.png)

### 训练后预测样例

![Prediction Examples](sample_predictions.png)

## 10. 结果分析

- Loss：{loss_text}。
- Accuracy：{acc_text}。
- 最终测试准确率：{final_test_acc:.4f}。
- 过拟合迹象：{overfit_text}。

## 11. 输出文件

- `metrics.csv`：每个 epoch 的 train loss、train accuracy、test loss、test accuracy。
- `loss_curve.png`：训练和测试 loss 曲线。
- `accuracy_curve.png`：训练和测试 accuracy 曲线。
- `sample_ground_truth.png`：测试样例真实标签。
- `sample_predictions.png`：训练后预测样例。
- `mnist_cnn.pth`：最终模型参数。
- `optimizer.pth`：最终优化器参数。
- `checkpoint.pth`：训练过程 checkpoint。

## 12. 实验总结

本实验完整覆盖了一个图像分类项目的基本流程：加载数据集、预处理、定义 CNN、训练模型、测试模型、保存指标和图表。实验结果表明，简单的两层卷积网络已经能在 MNIST 上取得较高准确率。由于 MNIST 数据集较简单，本实验更适合作为 PyTorch 图像分类流程模板，而不是复杂真实场景模型性能的代表。
"""
    report_path.write_text(content, encoding="utf-8")
