from pathlib import Path

import torch
from PIL import Image

from datasets.transforms import get_val_transforms
from models.flower_model import FlowerModel

# Web 推理工具：加载轻量模型并对上传图片进行分类，供 Flask 接口调用


_MODEL_PATH = Path(__file__).resolve().parents[1] / "model_web" / "model.pth"


def load_model(path: Path | str = _MODEL_PATH) -> FlowerModel:
    # 从指定路径加载模型权重并设置为 eval 模式
    model = FlowerModel(num_classes=10)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: FlowerModel, file) -> int:
    # 对上传文件执行推理，返回类别索引
    image = Image.open(file).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        return int(out.argmax(dim=1).item())
