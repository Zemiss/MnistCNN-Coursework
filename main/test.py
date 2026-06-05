import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from main.src.model import ConvNet


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained MNIST CNN on one image.")
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model") / "mnist_cnn.pth",
        help="Path to the trained model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction, probabilities = predict(args.image, args.model, device)

    print(f"Predicted digit: {prediction}")
    print("Probabilities:")
    for digit, probability in enumerate(probabilities):
        print(f"{digit}: {probability:.4f}")


if __name__ == "__main__":
    main()
