import argparse
from pathlib import Path

import torch
from PIL import Image

from datasets.transforms import get_val_transforms
from models.flower_model import FlowerModel
from models.multi_modal import MultiModalFusionModel


def load_image(path: Path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, inputs, device, topk: int = 1):
    model.eval()
    with torch.no_grad():
        logits = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        probs = torch.softmax(logits, dim=1)
        topk_values, topk_indices = torch.topk(probs, topk, dim=1)
    return topk_values.squeeze(0), topk_indices.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with FlowerCNN models")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/checkpoints/best_visible.pth"))
    parser.add_argument("--mode", choices=["visible", "infrared", "multimodal"], default="visible")
    parser.add_argument("--image", type=Path, help="Image for visible/infrared prediction")
    parser.add_argument("--vis-image", type=Path, help="Visible image for multimodal prediction")
    parser.add_argument("--ir-image", type=Path, help="Infrared image for multimodal prediction")
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transforms()

    if args.mode == "multimodal":
        if not (args.vis_image and args.ir_image):
            parser.error("multimodal mode requires --vis-image and --ir-image")
        model = MultiModalFusionModel(num_classes=10).to(device)
        inputs = (
            load_image(args.vis_image, transform).to(device),
            load_image(args.ir_image, transform).to(device),
        )
    else:
        if not args.image:
            parser.error("visible/infrared modes require --image")
        model = FlowerModel(num_classes=10).to(device)
        inputs = load_image(args.image, transform).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    values, indices = predict(model, inputs, device, topk=args.topk)

    print("Predictions:")
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. class {idx} (confidence {score:.4f})")


if __name__ == "__main__":
    main()
