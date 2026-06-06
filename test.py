import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from src.config import convert_section_paths, get_section, load_config, merge_config_with_args
from src.model import ConvNet


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
ARG_TARGETS = {
    "test_data_dir": ("paths", "test_data_dir"),
    "model_path": ("paths", "model_path"),
    "use_mnist": ("test", "use_mnist"),
    "num_samples": ("test", "num_samples"),
}


def load_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def load_mnist_image(image_tensor: torch.Tensor) -> torch.Tensor:
    transform = transforms.Normalize((0.1307,), (0.3081,))
    return transform(image_tensor).unsqueeze(0)


@torch.no_grad()
def predict(image_path: Path, model_path: Path, device: torch.device) -> tuple[int, list[float]]:
    network = ConvNet().to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    image = load_image(image_path).to(device)
    output = network(image)
    probabilities = output.exp().squeeze(0).cpu()
    prediction = int(probabilities.argmax().item())
    return prediction, probabilities.tolist()


@torch.no_grad()
def predict_mnist(
    image_tensor: torch.Tensor,
    label: int,
    model_path: Path,
    device: torch.device,
) -> tuple[int, list[float], int]:
    network = ConvNet().to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    image = load_mnist_image(image_tensor).to(device)
    output = network(image)
    probabilities = output.exp().squeeze(0).cpu()
    prediction = int(probabilities.argmax().item())
    return prediction, probabilities.tolist(), label


def list_image_paths(test_data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in test_data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args(argv=None, config: dict = None) -> argparse.Namespace:
    """Parse test arguments with nested config defaults."""
    config = config or {"paths": {}, "test": {}}
    base_dir = Path(__file__).resolve().parent
    paths_config = config.get("paths", {})
    test_config = config.get("test", {})

    test_data_dir_default = paths_config.get("test_data_dir")
    parser = argparse.ArgumentParser(description="Test a trained MNIST CNN on a folder of images or MNIST dataset.")
    parser.add_argument(
        "--test-data-dir",
        dest="test_data_dir",
        type=Path,
        default=test_data_dir_default,
        required=(test_data_dir_default is None),
        help="Directory of images to classify, or MNIST dataset directory when --use-mnist is set.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=Path,
        default=paths_config.get("model_path", base_dir / "model" / "mnist_cnn.pth"),
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--use-mnist",
        dest="use_mnist",
        action=argparse.BooleanOptionalAction,
        default=test_config.get("use_mnist", False),
        help="Load images from MNIST dataset instead of image files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=test_config.get("num_samples", 10),
        help="Number of samples to test, only with --use-mnist.",
    )
    return parser.parse_args(argv)


def main() -> None:
    config, main_dir = load_config()
    config = convert_section_paths(
        config,
        "paths",
        ["test_data_dir", "model_path"],
        main_dir,
    )

    args = parse_args(config=config)
    final_config = merge_config_with_args(config, args, ARG_TARGETS)

    paths_config = get_section(final_config, "paths")
    test_config = get_section(final_config, "test")

    test_data_dir = paths_config.get("test_data_dir")
    model_path = paths_config.get("model_path")
    use_mnist = test_config.get("use_mnist", False)
    num_samples = test_config.get("num_samples", 10)

    test_data_dir = Path(test_data_dir) if test_data_dir and not isinstance(test_data_dir, Path) else test_data_dir
    model_path = Path(model_path) if model_path and not isinstance(model_path, Path) else model_path

    if not test_data_dir:
        raise ValueError("--test-data-dir is required or must be set in paths.test_data_dir")
    if not test_data_dir.is_dir():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
    if not model_path or not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_mnist:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_dataset = datasets.MNIST(
            test_data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        num_samples_to_use = min(num_samples, len(test_dataset))
        correct = 0
        print(f"Testing on {num_samples_to_use} MNIST samples:\n")
        for i in range(num_samples_to_use):
            image_tensor, true_label = test_dataset[i]
            prediction, probabilities, label = predict_mnist(image_tensor, true_label, model_path, device)
            probability = probabilities[prediction]
            is_correct = "correct" if prediction == label else "wrong"
            print(f"Sample {i+1}: true_label={label}, predicted={prediction}, probability={probability:.4f} {is_correct}")
            if prediction == label:
                correct += 1
        accuracy = correct / num_samples_to_use
        print(f"\nAccuracy: {correct}/{num_samples_to_use} ({accuracy:.2%})")
    else:
        image_paths = list_image_paths(test_data_dir)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {test_data_dir}")

        print(f"Testing on {len(image_paths)} images:\n")
        for image_path in image_paths:
            prediction, probabilities = predict(image_path, model_path, device)
            probability = probabilities[prediction]
            print(f"{image_path.name}: predicted={prediction}, probability={probability:.4f}")


if __name__ == "__main__":
    main()
