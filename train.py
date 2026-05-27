from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvNet
from utils import (
    evaluate,
    generate_report,
    plot_accuracy_curve,
    plot_loss_curve,
    save_ground_truth_examples,
    save_metrics,
    save_prediction_examples,
    set_seed,
    train_epoch,
)


def main():
    batch_size = 512
    n_epochs = 20
    log_interval = 30
    random_seed = 1

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)

    sample_ground_truth_path = outputs_dir / "sample_ground_truth.png"
    save_ground_truth_examples(example_data, example_targets, sample_ground_truth_path)

    network = ConvNet().to(device)
    optimizer = optim.Adam(network.parameters())

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    checkpoint_path = outputs_dir / "checkpoint.pth"
    model_path = outputs_dir / "mnist_cnn.pth"
    optimizer_path = outputs_dir / "optimizer.pth"
    metrics_path = outputs_dir / "metrics.csv"
    loss_curve_path = outputs_dir / "loss_curve.png"
    accuracy_curve_path = outputs_dir / "accuracy_curve.png"
    sample_predictions_path = outputs_dir / "sample_predictions.png"
    report_path = outputs_dir / "report.md"

    initial_test_loss, initial_test_acc = evaluate(
        network,
        test_loader,
        device,
        test_losses,
    )

    metrics = []
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(
            network,
            train_loader,
            optimizer,
            device,
            epoch,
            log_interval,
            train_losses,
            train_counter,
            checkpoint_path,
        )
        test_loss, test_acc = evaluate(network, test_loader, device, test_losses)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    torch.save(network.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    save_metrics(metrics, metrics_path)
    plot_loss_curve(train_counter, train_losses, test_counter, test_losses, loss_curve_path)
    plot_accuracy_curve(metrics, accuracy_curve_path)
    save_prediction_examples(network, example_data, device, sample_predictions_path)
    generate_report(
        metrics_path,
        report_path,
        {
            "batch_size": batch_size,
            "epochs": n_epochs,
            "log_interval": log_interval,
            "random_seed": random_seed,
            "device": str(device),
            "initial_test_loss": initial_test_loss,
            "initial_test_acc": initial_test_acc,
        },
    )

    final_test_acc = metrics[-1]["test_acc"]
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print("Saved files:")
    for path in [
        metrics_path,
        loss_curve_path,
        accuracy_curve_path,
        sample_ground_truth_path,
        sample_predictions_path,
        model_path,
        optimizer_path,
        checkpoint_path,
        report_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
