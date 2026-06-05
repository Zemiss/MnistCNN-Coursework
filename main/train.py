import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main.src.config import convert_paths, load_config, merge_config_with_args
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


def parse_args(argv=None, config: dict = None) -> argparse.Namespace:
    """解析命令行参数，支持配置文件默认值"""
    if config is None:
        config = {}
    
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    
    # 从配置中获取默认值
    train_data_dir_default = config.get("train_data_dir", base_dir / "data")
    outputs_dir_default = config.get("outputs_dir", base_dir / "outputs")
    model_path_default = config.get("model_path", base_dir / "model" / "mnist_cnn.pth")
    batch_size_default = config.get("batch_size", 512)
    n_epochs_default = config.get("n_epochs", 20)
    log_interval_default = config.get("log_interval", 30)
    random_seed_default = config.get("random_seed", 1)
    
    parser.add_argument(
        "--train_data_dir",
        "--data-dir",
        dest="train_data_dir",
        type=Path,
        default=train_data_dir_default,
        help="Directory for MNIST data.",
    )
    parser.add_argument(
        "--outputs_dir",
        "--outputs-dir",
        dest="outputs_dir",
        type=Path,
        default=outputs_dir_default,
        help="Directory for metrics and plots.",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        dest="model_path",
        type=Path,
        default=model_path_default,
        help="Path for the best model weights.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=batch_size_default,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        dest="n_epochs",
        type=int,
        default=n_epochs_default,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        type=int,
        default=log_interval_default,
        help="Log interval.",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=int,
        default=random_seed_default,
        help="Random seed.",
    )
    return parser.parse_args(argv)


def main():
    # 加载 YAML 配置（返回配置字典和 main_dir）
    config, main_dir = load_config()
    
    # 将配置中的路径字符串转换为相对于 main_dir 的绝对路径
    path_keys = ["train_data_dir", "outputs_dir", "model_path"]
    config = convert_paths(config, path_keys, main_dir)
    
    # 解析命令行参数
    args = parse_args(config=config)
    
    # 合并配置和命令行参数（命令行参数优先）
    final_config = merge_config_with_args(config, args)
    
    # 从最终配置中提取所有参数
    batch_size = final_config.get("batch_size", 512)
    n_epochs = final_config.get("n_epochs", 20)
    log_interval = final_config.get("log_interval", 30)
    random_seed = final_config.get("random_seed", 1)

    data_dir = final_config.get("train_data_dir")
    outputs_dir = final_config.get("outputs_dir")
    model_path = final_config.get("model_path")
    
    # 确保都是 Path 对象
    data_dir = Path(data_dir) if data_dir and not isinstance(data_dir, Path) else data_dir
    outputs_dir = Path(outputs_dir) if outputs_dir and not isinstance(outputs_dir, Path) else outputs_dir
    model_path = Path(model_path) if model_path and not isinstance(model_path, Path) else model_path
    
    if not data_dir or not outputs_dir or not model_path:
        raise ValueError("Missing required configuration: train_data_dir, outputs_dir, or model_path")
    
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)

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
