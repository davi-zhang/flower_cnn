import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_mobilenetv3(num_classes: int = 10) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
