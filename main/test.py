import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from main.src.model import ConvNet


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


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
    """将 MNIST 张量转换为模型输入格式"""
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
def predict_mnist(image_tensor: torch.Tensor, label: int, model_path: Path, device: torch.device) -> tuple[int, list[float], int]:
    """从 MNIST 数据集预测单张图片"""
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


def parse_args(argv=None) -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Test a trained MNIST CNN on a folder of images or MNIST dataset.")
    parser.add_argument(
        "--test_data_dir",
        type=Path,
        required=True,
        help="Directory of images to classify, or MNIST dataset directory (when --use-mnist is set).",
    )
    parser.add_argument(
        "--model_path",
        "--model",
        dest="model_path",
        type=Path,
        default=base_dir / "model" / "mnist_cnn.pth",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--use-mnist",
        action="store_true",
        help="Load images from MNIST dataset instead of image files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test (only with --use-mnist).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.test_data_dir.is_dir():
        raise FileNotFoundError(f"Test data directory not found: {args.test_data_dir}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.use_mnist:
        # Load from MNIST dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_dataset = datasets.MNIST(
            args.test_data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        num_samples = min(args.num_samples, len(test_dataset))
        correct = 0
        print(f"Testing on {num_samples} MNIST samples:\n")
        for i in range(num_samples):
            image_tensor, true_label = test_dataset[i]
            prediction, probabilities, label = predict_mnist(image_tensor, true_label, args.model_path, device)
            probability = probabilities[prediction]
            is_correct = "✓" if prediction == label else "✗"
            print(f"Sample {i+1}: true_label={label}, predicted={prediction}, probability={probability:.4f} {is_correct}")
            if prediction == label:
                correct += 1
        accuracy = correct / num_samples
        print(f"\nAccuracy: {correct}/{num_samples} ({accuracy:.2%})")
    else:
        # Load from image files
        image_paths = list_image_paths(args.test_data_dir)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {args.test_data_dir}")

        print(f"Testing on {len(image_paths)} images:\n")
        for image_path in image_paths:
            prediction, probabilities = predict(image_path, args.model_path, device)
            probability = probabilities[prediction]
            print(f"{image_path.name}: predicted={prediction}, probability={probability:.4f}")


if __name__ == "__main__":
    main()
