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
