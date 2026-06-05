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

## 训练

```powershell
python -m main.train --train_data_dir data --outputs_dir main\outputs --model_path main\model\mnist_cnn.pth
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--train_data_dir` | MNIST 数据集保存位置 |
| `--outputs_dir` | 训练指标、曲线图、样例图片输出位置 |
| `--model_path` | best 模型权重保存位置 |

训练过程中只保存测试准确率最高的模型，文件为：

```text
main\model\mnist_cnn.pth
```

不会再保存 `optimizer.pth` 或 `checkpoint.pth`。

## 测试图片文件夹

测试脚本接收一个图片文件夹，会逐张预测其中的图片。支持的图片格式包括 `.png`、`.jpg`、`.jpeg`、`.bmp`。

```powershell
python -m main.test --test_data_dir path\to\digit_images --model_path main\model\mnist_cnn.pth
```

把 `path\to\digit_images` 换成你的测试图片文件夹路径。

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--test_data_dir` | 测试图片文件夹 |
| `--model_path` | 要加载的模型权重 |

## 测试 MNIST 数据集

也可以直接用 MNIST 测试集来测试模型，使用 `--use-mnist` 标志：

```powershell
python -m main.test --test_data_dir data --model_path main\model\mnist_cnn.pth --use-mnist --num-samples 20
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--test_data_dir` | 数据集保存位置 |
| `--model_path` | 要加载的模型权重 |
| `--use-mnist` | 使用 MNIST 数据集进行测试 |
| `--num-samples` | 测试样本数量（默认 10） |

## 常用完整命令

```
conda activate myenv
cd C:\Users\12445\Desktop\Mnist-CNN
```

**训练：**

```
python -m main.train --train_data_dir data --outputs_dir main\outputs --model_path main\model\mnist_cnn.pth
```

**测试（用 MNIST 数据集）：**

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
