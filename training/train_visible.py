
import argparse
import json
import os
from datasets.flower_dataset import FlowerDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.flower_model import FlowerModel
from training.utils import accuracy
from training.trainer_base import TrainerBase
import torch

# 训练脚本（可见光）：加载 102 类花卉数据集并训练 FlowerModel，保存最佳可见光模型权重

def train(
    root_dir: str = "data/raw",
    train_ann: str = "data/annotations/train.txt",
    val_ann: str = "data/annotations/val.txt",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_classes: int | None = None,
    log_dir: str = "experiments/logs/tensorboard",
    best_model_path: str = "experiments/checkpoints/best_visible.pth",
):
    # 构建训练/验证集与数据增强
    train_ds = FlowerDataset(
        root_dir=root_dir,
        annotations=train_ann,
        transform=get_train_transforms(),
    )
    val_ds = FlowerDataset(
        root_dir=root_dir,
        annotations=val_ann,
        transform=get_val_transforms(),
    )
    # 自动推断类别数（优先使用传入参数）
    if num_classes is None:
        label2id_path = os.path.join("data", "annotations", "label2id.json")
        if not os.path.isfile(label2id_path):
            raise FileNotFoundError(f"未找到标签映射文件: {label2id_path}，请先执行数据准备脚本")
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
            num_classes = (max(label2id.values()) + 1) if label2id else 0
        print(f"自动检测类别数: {num_classes}")

    # 创建模型和优化组件
    model = FlowerModel(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        patience=999,  # 禁用早停，完整训练所有 epochs
    )
    trainer.train(epochs=epochs, accuracy_fn=accuracy)

if __name__ == "__main__":
    # 通过命令行参数灵活控制数据路径和超参数
    parser = argparse.ArgumentParser(description="Train the visible flower classification model")
    parser.add_argument("--root-dir", type=str, default="data/raw", help="Root folder where images are stored")
    parser.add_argument("--train-ann", type=str, default="data/annotations/train.txt", help="Training annotations file")
    parser.add_argument("--val-ann", type=str, default="data/annotations/val.txt", help="Validation annotations file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of target classes; default auto from data/annotations/label2id.json")
    parser.add_argument("--log-dir", type=str, default=os.path.join("experiments", "logs", "tensorboard"), help="TensorBoard log directory")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("experiments", "checkpoints", "best_visible.pth"), help="Best model checkpoint path")
    args = parser.parse_args()

    train(
        root_dir=args.root_dir,
        train_ann=args.train_ann,
        val_ann=args.val_ann,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        log_dir=args.log_dir,
        best_model_path=args.checkpoint,
    )
