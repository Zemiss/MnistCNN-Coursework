# 深度学习原理课程作业：MNIST-CNN

![Python >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)
![PyTorch >=2.2](https://img.shields.io/badge/PyTorch->=2.2-yellow.svg)

>本项目用 PyTorch 训练一个简单的卷积神经网络，用于识别 MNIST 手写数字。训练脚本会保存训练指标、曲线图和测试准确率最高的模型权重。

参考教程：

[MNIST数据集手写数字识别](https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/)

## 目录

- [环境配置](#环境配置)
- [配置系统](#配置系统)
- [训练](#训练)
- [测试](#测试)
- [项目结构](#项目结构)
- [模型结构](#模型结构)
- [训练配置](#训练配置)
- [输出文件](#输出文件)


## 环境配置

```powershell
conda create -n mnist-cnn python=3.9 -y
pip install -r requirements.txt
```

## 配置系统

项目使用 YAML 配置文件 `configs/default.yaml` 作为默认配置。所有训练和测试参数都可在此文件中设置，也可通过命令行参数覆盖。

**修改配置的方法：**

1. 直接修改 `configs/default.yaml` 中的参数：

```yaml
paths:
  train_data_dir: data
  outputs_dir: outputs
  model_path: model/mnist_cnn.pth
  test_data_dir: null

train:
  batch_size: 512
  epochs: 20
  log_interval: 30
  seed: 1
```

2. 使用命令行参数来临时覆盖配置文件中的值：

```powershell
python train.py --batch-size 256 --epochs 10 --train-data-dir custom_data --outputs-dir custom_outputs
```

## 训练

**使用配置文件:**

```powershell
python train.py
```

**使用命令行参数:**

```powershell
python train.py --train-data-dir data --outputs-dir outputs --model-path model\mnist_cnn.pth
```

## 测试 

### MNIST 数据集

```powershell
python test.py --test-data-dir data --model-path model\mnist_cnn.pth --use-mnist --num-samples 20
```

### 图片文件夹

测试脚本接收一个图片文件夹，会逐张预测其中的图片。支持的图片格式包括 `.png`、`.jpg`、`.jpeg`、`.bmp`。

```powershell
python test.py --test-data-dir path\to\digit_images --model-path model\mnist_cnn.pth
```

把 `path\to\digit_images` 换成测试图片文件夹路径。


## 项目结构

```text
.
|-- data/
|-- train.py
|-- test.py
|-- requirements.txt
|-- configs/
|   `-- default.yaml
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- model.py
|   `-- utils.py
|-- model/
|   `-- mnist_cnn.pth
`-- outputs/
    |-- metrics.csv
    |-- loss_curve.png
    |-- accuracy_curve.png
    |-- sample_ground_truth.png
    `-- sample_predictions.png
```

## 模型结构

`src/model.py` 定义了 `ConvNet`：

| 层 | 定义 | 输出 |
| --- | --- | --- |
| Input | MNIST 灰度图 | 1 x 28 x 28 |
| Conv1 | `Conv2d(1, 10, kernel_size=5)` + ReLU + MaxPool2d(2, 2) | 10 x 12 x 12 |
| Conv2 | `Conv2d(10, 20, kernel_size=3)` + ReLU | 20 x 10 x 10 |
| Flatten | `view(batch_size, -1)` | 2000 |
| FC1 | `Linear(20 * 10 * 10, 500)` + ReLU | 500 |
| FC2 | `Linear(500, 10)` | 10 |
| Output | `log_softmax(dim=1)` | 10 个类别的 log probability |

## 训练配置

| 参数 | 值 |
| --- | --- |
| batch size | 512 |
| epochs | 20 |
| optimizer | Adam |
| loss | `F.nll_loss` |
| seed | 1 |
| device | 有 CUDA 用 CUDA，否则用 CPU |

## 输出文件

| 文件 | 说明 |
| --- | --- |
| `model\mnist_cnn.pth` | 测试准确率最高的模型权重 |
| `outputs\metrics.csv` | 每个 epoch 的训练和测试指标 |
| `outputs\loss_curve.png` | loss 曲线 |
| `outputs\accuracy_curve.png` | accuracy 曲线 |
| `outputs\sample_ground_truth.png` | 测试样例真实标签 |
| `outputs\sample_predictions.png` | 模型预测样例 |
