import torch
import torch.nn as nn

from .attention import SEBlock
from .mobilenetv3_base import build_mobilenetv3

# 花卉分类模型：在 MobileNetV3 特征后接入 SE 通道注意力，再使用原始分类头输出


class FlowerModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = build_mobilenetv3(num_classes=num_classes)
        self.attention = SEBlock(channels=576)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        features = self.attention(features)
        pooled = self.backbone.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        return self.backbone.classifier(flattened)
