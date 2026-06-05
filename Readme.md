# MNIST ConvNet Experiment

This project follows the PyTorch Tutorial ConvNet MNIST workflow:

https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/

It trains a CNN on MNIST and generates metrics, plots, model weights, and checkpoint files for an experiment report.

## Run Training

Use the `myenv` environment:

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe -m main.train
```

Training writes plots and metrics to `main/outputs/`, and trained model files to `main/model/`.

## Test One Image

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe -m main.test path\to\digit.png --model main\model\mnist_cnn.pth
```

The `--model` path is explicit so the command works from the project root.

## Files

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
    |   |-- mnist_cnn.pth
    |   |-- optimizer.pth
    |   `-- checkpoint.pth
    `-- outputs/
        |-- metrics.csv
        |-- loss_curve.png
        |-- accuracy_curve.png
        |-- sample_ground_truth.png
        |-- sample_predictions.png
        `-- report.md
```

## Model

`main/src/model.py` defines `ConvNet`:

| Layer | Definition | Output |
| --- | --- | --- |
| Input | MNIST grayscale image | 1 x 28 x 28 |
| Conv1 | `Conv2d(1, 10, kernel_size=5)` + ReLU + MaxPool2d(2, 2) | 10 x 12 x 12 |
| Conv2 | `Conv2d(10, 20, kernel_size=3)` + ReLU | 20 x 10 x 10 |
| Flatten | `view(batch_size, -1)` | 2000 |
| FC1 | `Linear(20 * 10 * 10, 500)` + ReLU | 500 |
| FC2 | `Linear(500, 10)` | 10 |
| Output | `log_softmax(dim=1)` | 10 log probabilities |

## Training

| Parameter | Value |
| --- | --- |
| batch size | 512 |
| epochs | 20 |
| optimizer | Adam |
| loss | `F.nll_loss` |
| seed | 1 |
| device | CUDA if available, otherwise CPU |

`main/outputs/report.md` is a manually written experiment report. Training does not overwrite it.
