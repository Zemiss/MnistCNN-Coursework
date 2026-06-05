import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main.src.config import convert_section_paths, get_section, load_config, merge_config_with_args
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


ARG_TARGETS = {
    "train_data_dir": ("paths", "train_data_dir"),
    "outputs_dir": ("paths", "outputs_dir"),
    "model_path": ("paths", "model_path"),
    "batch_size": ("train", "batch_size"),
    "epochs": ("train", "epochs"),
    "log_interval": ("train", "log_interval"),
    "seed": ("train", "seed"),
    "optimizer": ("optimizer", "name"),
    "learning_rate": ("optimizer", "learning_rate"),
}


def save_best_model(network, test_acc: float, best_test_acc: float, model_path: Path):
    if test_acc > best_test_acc:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), model_path)
        return test_acc, True
    return best_test_acc, False


def parse_args(argv=None, config: dict = None) -> argparse.Namespace:
    """Parse training arguments with nested config defaults."""
    config = config or {"paths": {}, "train": {}, "optimizer": {}}
    base_dir = Path(__file__).resolve().parent
    paths_config = config.get("paths", {})
    train_config = config.get("train", {})
    optimizer_config = config.get("optimizer", {})

    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument(
        "--train-data-dir",
        dest="train_data_dir",
        type=Path,
        default=paths_config.get("train_data_dir", base_dir / "data"),
        help="Directory for MNIST data.",
    )
    parser.add_argument(
        "--outputs-dir",
        dest="outputs_dir",
        type=Path,
        default=paths_config.get("outputs_dir", base_dir / "outputs"),
        help="Directory for metrics and plots.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=Path,
        default=paths_config.get("model_path", base_dir / "model" / "mnist_cnn.pth"),
        help="Path for the best model weights.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=train_config.get("batch_size", 512),
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=train_config.get("epochs", 20),
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        type=int,
        default=train_config.get("log_interval", 30),
        help="Log interval.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=train_config.get("seed", 1),
        help="Random seed.",
    )
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        default=optimizer_config.get("name", "adam"),
        choices=["adam"],
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=optimizer_config.get("learning_rate", 0.001),
        help="Optimizer learning rate.",
    )
    return parser.parse_args(argv)


def main():
    config, main_dir = load_config()
    config = convert_section_paths(
        config,
        "paths",
        ["train_data_dir", "outputs_dir", "model_path"],
        main_dir,
    )

    args = parse_args(config=config)
    final_config = merge_config_with_args(config, args, ARG_TARGETS)

    paths_config = get_section(final_config, "paths")
    train_config = get_section(final_config, "train")
    optimizer_config = get_section(final_config, "optimizer")

    batch_size = train_config.get("batch_size", 512)
    epochs = train_config.get("epochs", 20)
    log_interval = train_config.get("log_interval", 30)
    seed = train_config.get("seed", 1)
    learning_rate = optimizer_config.get("learning_rate", 0.001)

    data_dir = paths_config.get("train_data_dir")
    outputs_dir = paths_config.get("outputs_dir")
    model_path = paths_config.get("model_path")

    data_dir = Path(data_dir) if data_dir and not isinstance(data_dir, Path) else data_dir
    outputs_dir = Path(outputs_dir) if outputs_dir and not isinstance(outputs_dir, Path) else outputs_dir
    model_path = Path(model_path) if model_path and not isinstance(model_path, Path) else model_path

    if not data_dir or not outputs_dir or not model_path:
        raise ValueError("Missing required configuration: paths.train_data_dir, paths.outputs_dir, or paths.model_path")

    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown GPU"
        print(f"Using device: {device} (GPU) - {gpu_name}, count={torch.cuda.device_count()}")
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
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

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
    for epoch in range(1, epochs + 1):
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
