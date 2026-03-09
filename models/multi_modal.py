import torch.nn as nn

from .mobilenetv3_base import build_mobilenetv3

# 多模态融合模型：可见光与红外使用独立 MobileNetV3 并对输出 logits 求平均


class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.vis_model = build_mobilenetv3(num_classes=num_classes)
        self.ir_model = build_mobilenetv3(num_classes=num_classes)

    def forward(self, vis, ir):
        # 分别前向并融合两路 logits
        vis_logits = self.vis_model(vis)
        ir_logits = self.ir_model(ir)
        return (vis_logits + ir_logits) * 0.5
