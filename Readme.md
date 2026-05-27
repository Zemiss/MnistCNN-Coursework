# MNIST ConvNet Experiment

This project follows the PyTorch Tutorial ConvNet MNIST workflow:

https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/

It trains a CNN on MNIST and automatically generates the files needed for an experiment report.

## Run

Use the `myenv` environment:

```powershell
C:\Users\12445\miniconda3\envs\myenv\python.exe train.py
```

## Files

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
    ├── sample_ground_truth.png
    ├── sample_predictions.png
    ├── mnist_cnn.pth
    ├── optimizer.pth
    ├── checkpoint.pth
    └── report.md
```

## Model

`model.py` defines `ConvNet`:

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

After training, open `outputs/report.md` for the generated experiment report. It includes the dataset description, preprocessing, network structure, training method, loss function, metrics table, loss curve, accuracy curve, sample images, and result analysis.
