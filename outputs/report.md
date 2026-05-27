# MNIST 手写数字分类实验报告

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
| batch_size | 64 |
| epochs | 10 |
| learning_rate | 0.001 |
| optimizer | Adam |
| loss function | CrossEntropyLoss |
| device | cuda |
| seed | 42 |

## 6. 损失函数说明

实验使用 `nn.CrossEntropyLoss()` 作为损失函数。模型最后一层直接输出 logits，不在 `forward` 中使用 `softmax` 或 `log_softmax`。

## 7. 训练结果表格

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 1 | 0.1696 | 0.9500 | 0.0578 | 0.9810 |
| 2 | 0.0525 | 0.9842 | 0.0356 | 0.9881 |
| 3 | 0.0370 | 0.9885 | 0.0355 | 0.9888 |
| 4 | 0.0272 | 0.9913 | 0.0360 | 0.9881 |
| 5 | 0.0216 | 0.9930 | 0.0344 | 0.9888 |
| 6 | 0.0152 | 0.9949 | 0.0333 | 0.9897 |
| 7 | 0.0137 | 0.9954 | 0.0342 | 0.9894 |
| 8 | 0.0107 | 0.9966 | 0.0422 | 0.9872 |
| 9 | 0.0093 | 0.9967 | 0.0429 | 0.9876 |
| 10 | 0.0080 | 0.9974 | 0.0438 | 0.9885 |

## 8. 训练曲线

![Loss Curve](loss_curve.png)

![Accuracy Curve](accuracy_curve.png)

## 9. 简短结果分析

- Loss：整体呈下降趋势。
- Accuracy：整体呈上升趋势。
- 最终测试准确率：0.9885。
- 过拟合迹象：训练准确率与测试准确率差距较小，未观察到明显过拟合迹象。

## 10. 实验总结

本实验使用一个小型 CNN 完成 MNIST 手写数字分类任务。训练过程记录了每个 epoch 的训练集和测试集损失、准确率，并自动保存模型、指标文件、曲线图和实验报告草稿，便于后续复现实验和撰写正式报告。
