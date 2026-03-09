import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 构建基础的 MobileNetV3-Small，并替换最后一层为指定类别数
# 优化：使用 ImageNet 预训练权重进行迁移学习，大幅提升收敛速度与精度


def build_mobilenetv3(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
