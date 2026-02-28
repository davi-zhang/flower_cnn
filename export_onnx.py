import argparse
import os

import torch

from models.flower_model import FlowerModel


def export(checkpoint_path: str, onnx_path: str, input_shape=(1, 3, 224, 224)) -> None:
    model = FlowerModel(num_classes=10)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    sample = torch.randn(input_shape)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        sample,
        onnx_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FlowerModel to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best_visible.pth",
        help="Path to a saved torch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/flower_model.onnx",
        help="Destination ONNX file",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
