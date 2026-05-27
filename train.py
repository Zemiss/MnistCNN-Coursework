from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN
from utils import (
    evaluate,
    generate_report,
    plot_curves,
    save_metrics,
    set_seed,
    train_one_epoch,
)


def main():
    seed = 42
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        metrics.append(row)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.4f}"
        )

    metrics_path = outputs_dir / "metrics.csv"
    loss_curve_path = outputs_dir / "loss_curve.png"
    accuracy_curve_path = outputs_dir / "accuracy_curve.png"
    model_path = outputs_dir / "mnist_cnn.pth"
    report_path = outputs_dir / "report.md"

    save_metrics(metrics, metrics_path)
    plot_curves(metrics, loss_curve_path, accuracy_curve_path)
    torch.save(model.state_dict(), model_path)
    generate_report(
        metrics_path,
        report_path,
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "device": str(device),
            "seed": seed,
        },
    )

    final_test_acc = metrics[-1]["test_acc"]
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print("Saved files:")
    for path in [
        metrics_path,
        loss_curve_path,
        accuracy_curve_path,
        model_path,
        report_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
