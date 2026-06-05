import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main.src.model import ConvNet
from main.src.utils import (
    evaluate,
    plot_accuracy_curve,
    plot_loss_curve,
    save_ground_truth_examples,
    save_metrics,
    save_prediction_examples,
    set_seed,
    train_epoch,
)


def save_best_model(network, test_acc: float, best_test_acc: float, model_path: Path):
    if test_acc > best_test_acc:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), model_path)
        return test_acc, True
    return best_test_acc, False


def parse_args(argv=None) -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument(
        "--train_data_dir",
        "--data-dir",
        dest="train_data_dir",
        type=Path,
        default=base_dir / "data",
        help="Directory for MNIST data.",
    )
    parser.add_argument(
        "--outputs_dir",
        "--outputs-dir",
        dest="outputs_dir",
        type=Path,
        default=base_dir / "outputs",
        help="Directory for metrics and plots.",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        dest="model_path",
        type=Path,
        default=base_dir / "model" / "mnist_cnn.pth",
        help="Path for the best model weights.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    batch_size = 512
    n_epochs = 20
    log_interval = 30
    random_seed = 1

    data_dir = args.train_data_dir
    outputs_dir = args.outputs_dir
    model_path = args.model_path
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 显示设备信息，明确告知是否在 GPU 上训练
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown GPU"
        print(f"Using device: {device} (GPU) — {gpu_name}, count={torch.cuda.device_count()}")
    else:
        print(f"Using device: {device} (CPU)")

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

    metrics_path = outputs_dir / "metrics.csv"
    loss_curve_path = outputs_dir / "loss_curve.png"
    accuracy_curve_path = outputs_dir / "accuracy_curve.png"
    sample_predictions_path = outputs_dir / "sample_predictions.png"

    evaluate(
        network,
        test_loader,
        device,
        test_losses,
    )

    metrics = []
    best_test_acc = -1.0
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
        )
        test_loss, test_acc = evaluate(network, test_loader, device, test_losses)
        best_test_acc, saved_best = save_best_model(
            network,
            test_acc,
            best_test_acc,
            model_path,
        )
        if saved_best:
            print(f"Saved best model: {model_path} (test_acc={test_acc:.4f})")
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    save_metrics(metrics, metrics_path)
    plot_loss_curve(train_counter, train_losses, test_counter, test_losses, loss_curve_path)
    plot_accuracy_curve(metrics, accuracy_curve_path)
    save_prediction_examples(network, example_data, device, sample_predictions_path)

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
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
