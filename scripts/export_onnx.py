import argparse
import json
import os

import torch

from models.flower_model import FlowerModel


def resolve_num_classes(cli_num_classes: int | None) -> int:
    # 优先使用命令行参数，其次自动读取标签映射文件
    if cli_num_classes is not None:
        return cli_num_classes
    label2id_path = os.path.join("data", "annotations", "label2id.json")
    if not os.path.isfile(label2id_path):
        raise FileNotFoundError(f"未找到 {label2id_path}，请通过 --num-classes 指定类别数")
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
        num_classes = (max(label2id.values()) + 1) if label2id else 0
    print(f"自动检测类别数: {num_classes}")
    return num_classes


def export(checkpoint_path: str, onnx_path: str, num_classes: int, input_shape=(1, 3, 224, 224)) -> None:
    # 构建 FlowerModel 并准备加载训练好的参数
    model = FlowerModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 生成与输入形状一致的随机张量用于导出过程中的 tracing
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
    # 配置导出工具的命令行参数
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (must match checkpoint); default auto from label2id.json",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output, num_classes=resolve_num_classes(args.num_classes))


if __name__ == "__main__":
    main()
