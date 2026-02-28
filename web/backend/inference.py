from pathlib import Path

import torch
from PIL import Image

from datasets.transforms import get_val_transforms
from models.flower_model import FlowerModel


_MODEL_PATH = Path(__file__).resolve().parents[1] / "model_web" / "model.pth"


def load_model(path: Path | str = _MODEL_PATH) -> FlowerModel:
    model = FlowerModel(num_classes=10)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: FlowerModel, file) -> int:
    image = Image.open(file).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        return int(out.argmax(dim=1).item())
