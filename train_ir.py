import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.flower_dataset import FlowerDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.flower_model import FlowerModel
from training.utils import accuracy, save_checkpoint, EarlyStopping


def train(
    root_dir: str = "data/processed/ir",
    train_ann: str = "data/annotations/train_ir.txt",
    val_ann: str = "data/annotations/val_ir.txt",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    log_dir: str = "experiments/logs/ir",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FlowerDataset(root_dir=root_dir, annotations=train_ann, transform=get_train_transforms())
    val_ds = FlowerDataset(root_dir=root_dir, annotations=val_ann, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = FlowerModel(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    writer = SummaryWriter(log_dir)
    early_stopper = EarlyStopping(patience=5)
    best_acc = 0.0
    best_model_path = os.path.join("experiments", "checkpoints", "best_ir.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        accs = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss_total += loss.item()
                accs.append(accuracy(out, labels))

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_acc = sum(accs) / len(accs)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", avg_val_acc, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            save_checkpoint(model, best_model_path)
            print(f"Epoch {epoch}: New best IR model saved! Acc={best_acc:.4f}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"Ir early stopping at epoch {epoch}")
            break

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
        )

    duration = time.time() - start_time
    print(f"IR training finished in {duration/60:.2f} minutes")
    writer.close()


if __name__ == "__main__":
    train()
