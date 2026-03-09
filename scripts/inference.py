import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from datasets.transforms import get_val_transforms
from models.flower_model import FlowerModel
from models.multi_modal import MultiModalFusionModel

# 推理脚本：支持单模态（可见光/红外）和多模态模型，根据输入模式加载对应模型并输出 top-k 预测


def resolve_num_classes(cli_num_classes):
    # 优先使用命令行参数，其次自动读取标签映射文件
    if cli_num_classes is not None:
        return cli_num_classes
    label2id_path = Path("data/annotations/label2id.json")
    if not label2id_path.exists():
        raise FileNotFoundError(f"未找到 {label2id_path}，请通过 --num-classes 指定类别数")
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
        num_classes = (max(label2id.values()) + 1) if label2id else 0
    print(f"自动检测类别数: {num_classes}")
    return num_classes


def load_image(path: Path, transform):
    # 读取图片并应用验证阶段的预处理
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, inputs, device, topk: int = 1):
    # 封装推理流程，兼容多模态输入元组
    model.eval()
    with torch.no_grad():
        logits = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        probs = torch.softmax(logits, dim=1)
        topk_values, topk_indices = torch.topk(probs, topk, dim=1)
    return topk_values.squeeze(0), topk_indices.squeeze(0)


def main() -> None:
    # 命令行入口：选择模式、模型权重与输入图片
    parser = argparse.ArgumentParser(description="Run inference with FlowerCNN models")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/checkpoints/best_visible.pth"))
    parser.add_argument("--mode", choices=["visible", "infrared", "multimodal"], default="visible")
    parser.add_argument("--image", type=Path, help="Image for visible/infrared prediction")
    parser.add_argument("--vis-image", type=Path, help="Visible image for multimodal prediction")
    parser.add_argument("--ir-image", type=Path, help="Infrared image for multimodal prediction")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (must match checkpoint); default auto from label2id.json")
    args = parser.parse_args()
    num_classes = resolve_num_classes(args.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transforms()

    if args.mode == "multimodal":
        if not (args.vis_image and args.ir_image):
            parser.error("multimodal mode requires --vis-image and --ir-image")
        model = MultiModalFusionModel(num_classes=num_classes).to(device)
        inputs = (
            load_image(args.vis_image, transform).to(device),
            load_image(args.ir_image, transform).to(device),
        )
    else:
        if not args.image:
            parser.error("visible/infrared modes require --image")
        model = FlowerModel(num_classes=num_classes).to(device)
        inputs = load_image(args.image, transform).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    values, indices = predict(model, inputs, device, topk=args.topk)

    print("Predictions:")
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. class {idx} (confidence {score:.4f})")


if __name__ == "__main__":
    main()
