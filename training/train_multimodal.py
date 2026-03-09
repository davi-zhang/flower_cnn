
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.transforms import get_train_transforms
from models.multi_modal import MultiModalFusionModel
from training.utils import accuracy
from training.trainer_base import TrainerBase

# 训练脚本（多模态）：使用可见光+红外双输入的数据集训练融合模型

class MultiModalDataset(Dataset):
    def __init__(
        self,
        root_vis: str,
        root_ir: str,
        annotations: str,
        transform=None,
    ):
        # 读取标注文件，每行包含可见光路径、红外路径与标签
        self.root_vis = root_vis
        self.root_ir = root_ir
        self.transform = transform
        self.samples = []
        with open(annotations, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vis_path, ir_path, label = line.split(",")
                self.samples.append((vis_path.strip(), ir_path.strip(), int(label.strip())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 返回对齐的可见光/红外图像及标签
        vis_rel, ir_rel, label = self.samples[index]
        vis_img = Image.open(os.path.join(self.root_vis, vis_rel)).convert("RGB")
        ir_img = Image.open(os.path.join(self.root_ir, ir_rel)).convert("RGB")
        if self.transform:
            vis_img = self.transform(vis_img)
            ir_img = self.transform(ir_img)
        return vis_img, ir_img, label

def train(
    vis_root: str = "data/processed/visible",
    ir_root: str = "data/processed/ir",
    annotations: str = "data/annotations/dual_visible_ir.csv",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    dataset = MultiModalDataset(
        root_vis=vis_root,
        root_ir=ir_root,
        annotations=annotations,
        transform=get_train_transforms(),
    )
    # 验证集可按需实现
    model = MultiModalFusionModel(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    log_dir = os.path.join("experiments", "logs", "multimodal")
    best_model_path = os.path.join("experiments", "checkpoints", "best_multimodal.pth")
    trainer = TrainerBase(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,  # 若有val集可替换
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log_dir=log_dir,
        best_model_path=best_model_path,
        batch_size=batch_size,
        patience=5,
    )
    trainer.train(epochs=epochs, accuracy_fn=accuracy)

if __name__ == "__main__":
    train()
