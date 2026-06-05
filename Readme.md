# MNIST CNN 手写数字识别

这个项目用 PyTorch 训练一个简单的卷积神经网络，用于识别 MNIST 手写数字。训练脚本会保存训练指标、曲线图和测试准确率最高的模型权重。

参考教程：

https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/

## 环境

项目使用 Conda 环境 `myenv`。建议先进入项目根目录再运行脚本：

```powershell
conda activate myenv
cd C:\Users\12445\Desktop\Mnist-CNN
```

如果 `conda activate myenv` 不能用，也可以直接使用环境里的 Python：

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe
```

## 安装依赖

```powershell
conda activate myenv
cd C:\Users\12445\Desktop\Mnist-CNN
pip install -r main\requirements.txt
```

## 配置系统

项目使用 YAML 配置文件 `main/configs/default.yaml` 作为默认配置。所有训练和测试参数都可在此文件中设置，也可通过命令行参数覆盖。

### 配置文件位置

```
main/configs/default.yaml
```

### 配置优先级

1. **命令行参数** (最高优先级)
2. **配置文件** (默认值)
3. **代码中的硬编码默认值** (最低优先级)

### 修改配置的方法

#### 方法 1：编辑配置文件
直接修改 `main/configs/default.yaml` 中的参数：

```yaml
batch_size: 512
n_epochs: 20
train_data_dir: data
outputs_dir: main\outputs
model_path: main\model\mnist_cnn.pth
```

#### 方法 2：命令行参数（覆盖配置文件）
使用命令行参数来临时覆盖配置文件中的值：

```powershell
python -m main.train --batch-size 256 --epochs 10 --train_data_dir custom_data --outputs_dir custom_outputs
```

## 训练

### 使用配置文件（推荐）
使用默认配置文件运行训练：

```powershell
python -m main.train
```

### 使用命令行参数
用命令行参数覆盖配置：

```powershell
python -m main.train --train_data_dir data --outputs_dir main\outputs --model_path main\model\mnist_cnn.pth
```

### 使用命令行参数指定特定参数
修改特定参数而保持其他参数使用配置文件的默认值：

```powershell
# 只修改批大小和 epoch 数
python -m main.train --batch-size 256 --epochs 15

# 只修改输出目录
python -m main.train --outputs_dir my_custom_outputs
```

### 可用的训练参数

| 参数 | 命令行选项 | 配置键 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 数据目录 | `--train_data_dir` / `--data-dir` | `train_data_dir` | `data` | MNIST 数据集保存位置 |
| 输出目录 | `--outputs_dir` / `--outputs-dir` | `outputs_dir` | `main\outputs` | 训练指标、曲线图、样例图片输出位置 |
| 模型路径 | `--model_path` / `--model-path` | `model_path` | `main\model\mnist_cnn.pth` | best 模型权重保存位置 |
| 批大小 | `--batch-size` | `batch_size` | `512` | 每个批次的样本数 |
| 训练轮数 | `--epochs` | `n_epochs` | `20` | 训练的总轮数 |
| 日志间隔 | `--log-interval` | `log_interval` | `30` | 打印日志的间隔 |
| 随机种子 | `--seed` | `random_seed` | `1` | 随机种子，用于重现结果 |

训练过程中只保存测试准确率最高的模型，文件为：

```text
main\model\mnist_cnn.pth
```

不会再保存 `optimizer.pth` 或 `checkpoint.pth`。

## 测试图片文件夹

测试脚本接收一个图片文件夹，会逐张预测其中的图片。支持的图片格式包括 `.png`、`.jpg`、`.jpeg`、`.bmp`。

### 使用配置文件运行测试
先在 `main/configs/default.yaml` 中设置 `test_data_dir`：

```yaml
test_data_dir: path/to/digit_images
```

然后运行：
```powershell
python -m main.test
```

### 使用命令行参数运行测试
直接指定测试数据目录：

```powershell
python -m main.test --test_data_dir path\to\digit_images --model_path main\model\mnist_cnn.pth
```

把 `path\to\digit_images` 换成你的测试图片文件夹路径。

### 可用的测试参数

| 参数 | 命令行选项 | 配置键 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 数据目录 | `--test_data_dir` / `--data-dir` | `test_data_dir` | (必需) | 测试图片文件夹或 MNIST 数据集目录 |
| 模型路径 | `--model_path` / `--model` | `model_path` | `main\model\mnist_cnn.pth` | 要加载的模型权重 |
| 使用 MNIST | `--use-mnist` | `use_mnist` | `false` | 使用 MNIST 数据集进行测试 |
| 样本数量 | `--num-samples` | `num_samples` | `10` | 测试样本数量（仅用于 MNIST） |

## 测试 MNIST 数据集

也可以直接用 MNIST 测试集来测试模型。

### 使用配置文件
在 `main/configs/default.yaml` 中设置：

```yaml
test_data_dir: data
use_mnist: true
num_samples: 20
```

然后运行：
```powershell
python -m main.test
```

### 使用命令行参数
```powershell
python -m main.test --test_data_dir data --model_path main\model\mnist_cnn.pth --use-mnist --num-samples 20
```

## 常用完整命令

```
conda activate myenv
cd C:\Users\12445\Desktop\Mnist-CNN
```

**训练：**

python -m main.train

```
python -m main.train --batch-size 256 --epochs 10 --outputs_dir custom_outputs
```

**测试：**

python -m main.test

```
python -m main.test --test_data_dir data --model_path main\model\mnist_cnn.pth --use-mnist --num-samples 20
```

**测试（用自己的图片）：**

```
python -m main.test --test_data_dir path\to\digit_images --model_path main\model\mnist_cnn.pth
```


## 项目结构

```text
.
|-- data/
`-- main/
    |-- train.py
    |-- test.py
    |-- requirements.txt
    |-- src/
    |   |-- __init__.py
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

`main/src/model.py` 定义了 `ConvNet`：

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
| `main\model\mnist_cnn.pth` | 测试准确率最高的模型权重 |
| `main\outputs\metrics.csv` | 每个 epoch 的训练和测试指标 |
| `main\outputs\loss_curve.png` | loss 曲线 |
| `main\outputs\accuracy_curve.png` | accuracy 曲线 |
| `main\outputs\sample_ground_truth.png` | 测试样例真实标签 |
| `main\outputs\sample_predictions.png` | 模型预测样例 |
