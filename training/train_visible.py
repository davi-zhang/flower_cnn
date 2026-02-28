import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.flower_dataset import FlowerDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.flower_model import FlowerModel
from training.utils import accuracy, save_checkpoint


def train(
    root_dir: str = "data/processed/visible",
    train_ann: str = "data/annotations/train_visible.txt",
    val_ann: str = "data/annotations/val_visible.txt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    log_dir: str = "training/logs/visible",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FlowerDataset(root_dir=root_dir, annotations=train_ann, transform=get_train_transforms())
    val_ds = FlowerDataset(root_dir=root_dir, annotations=val_ann, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = FlowerModel(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d} | train loss {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_acc += accuracy(out, labels) * imgs.size(0)
        val_acc /= len(val_loader.dataset)
        print(f"Epoch {epoch:02d} | val acc {val_acc:.4f}")
        writer.add_scalar("val/accuracy", val_acc, epoch)

        save_checkpoint(model, f"experiments/logs/visible_epoch{epoch:02d}.pth")

    duration = time.time() - start_time
    print(f"Training finished in {duration/60:.2f} minutes")
    writer.close()


if __name__ == "__main__":
    train()
