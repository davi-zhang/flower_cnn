
import os
from datasets.flower_dataset import FlowerDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.flower_model import FlowerModel
from training.utils import accuracy
from training.trainer_base import TrainerBase
import torch

# 训练脚本（红外数据）：构建数据集、模型与优化器，使用 TrainerBase 执行训练与保存最优权重

def train(
    root_dir: str = "data/processed/ir",
    train_ann: str = "data/annotations/train_ir.txt",
    val_ann: str = "data/annotations/val_ir.txt",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    log_dir: str = "experiments/logs/ir",
):
    # 构建训练/验证集与数据增强
    train_ds = FlowerDataset(root_dir=root_dir, annotations=train_ann, transform=get_train_transforms())
    val_ds = FlowerDataset(root_dir=root_dir, annotations=val_ann, transform=get_val_transforms())
    # 创建模型与优化组件
    model = FlowerModel(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_model_path = os.path.join("experiments", "checkpoints", "best_ir.pth")
    trainer = TrainerBase(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
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
