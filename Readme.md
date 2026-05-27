# MNIST CNN 手写数字分类实验

本项目使用 PyTorch 和 torchvision 完成 MNIST 手写数字 0-9 十分类实验。代码会自动下载 MNIST 数据集，训练一个小型 CNN，并生成实验报告所需的模型、指标 CSV、训练曲线图和 Markdown 报告草稿。

## 项目结构

```text
.
├── model.py
├── train.py
├── utils.py
├── requirements.txt
├── data/
└── outputs/
    ├── metrics.csv
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── mnist_cnn.pth
    └── report.md
```

## 环境配置

建议在 `myenv` 虚拟环境中运行：

```powershell
conda activate myenv
pip install -r requirements.txt
```

如果不激活环境，也可以直接使用：

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe -m pip install -r requirements.txt
```

## 运行实验

在项目根目录执行：

```powershell
python train.py
```

或指定 `myenv` 的 Python：

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe train.py
```

脚本会自动完成：

- 下载并加载 MNIST 数据集
- 训练 `SmallCNN` 模型
- 逐 epoch 打印训练集和测试集指标
- 保存模型权重、指标 CSV、loss 曲线、accuracy 曲线
- 生成 Markdown 实验报告草稿

## 模型结构

`SmallCNN` 结构如下：

| 层次 | 结构 |
| --- | --- |
| Input | 1 x 28 x 28 |
| Conv1 | Conv2d(1, 16, kernel_size=3, padding=1) + ReLU + MaxPool2d(2) |
| Conv2 | Conv2d(16, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2) |
| Flatten | Flatten |
| FC1 | Linear(32 * 7 * 7, 128) + ReLU |
| FC2 | Linear(128, 10) |

最后一层直接输出 logits，损失函数使用 `nn.CrossEntropyLoss()`，因此 `forward` 中不使用 `softmax` 或 `log_softmax`。

## 训练配置

| 参数 | 取值 |
| --- | --- |
| batch_size | 64 |
| epochs | 10 |
| learning_rate | 0.001 |
| optimizer | Adam |
| loss function | CrossEntropyLoss |
| seed | 42 |
| device | cuda 优先，否则 cpu |

## 输出文件

训练完成后会在 `outputs/` 下生成：

- `metrics.csv`：每个 epoch 的 `train_loss`、`train_acc`、`test_loss`、`test_acc`
- `loss_curve.png`：训练集和测试集 loss 曲线
- `accuracy_curve.png`：训练集和测试集 accuracy 曲线
- `mnist_cnn.pth`：模型权重
- `report.md`：实验报告草稿

## 当前实验结果

最近一次完整训练的最终测试准确率：

```text
test_acc = 0.9885
```

第 10 个 epoch 指标：

| epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- |
| 10 | 0.007995 | 0.997417 | 0.043808 | 0.9885 |

## 复现实验说明

本项目设置了随机种子 `seed=42`，并配置了 cuDNN 的确定性选项，以尽量保证结果可复现。不同硬件、CUDA/cuDNN 版本或 PyTorch 版本下，最终指标可能存在轻微差异。
