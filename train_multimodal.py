import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets.transforms import get_train_transforms, get_val_transforms
from models.multi_modal import MultiModalFusionModel
from training.utils import accuracy, save_checkpoint, EarlyStopping


class MultiModalDataset(Dataset):
    def __init__(
        self,
        root_vis: str,
        root_ir: str,
        annotations: str,
        transform=None,
    ):
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
) -> None:
    dataset = MultiModalDataset(
        root_vis=vis_root,
        root_ir=ir_root,
        annotations=annotations,
        transform=get_train_transforms(),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalFusionModel(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    writer = SummaryWriter(os.path.join("experiments", "logs", "multimodal"))
    early_stopper = EarlyStopping(patience=5)

    best_acc = 0.0
    best_model_path = os.path.join("experiments", "checkpoints", "best_multimodal.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        accs = []

        for vis, ir, labels in loader:
            vis, ir, labels = vis.to(device), ir.to(device), labels.to(device)
            out = model(vis, ir)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            accs.append(accuracy(out, labels))

        avg_train_loss = train_loss / len(loader)
        avg_train_acc = sum(accs) / len(accs)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch)

        scheduler.step(avg_train_loss)

        if avg_train_acc > best_acc:
            best_acc = avg_train_acc
            save_checkpoint(model, best_model_path)
            print(f"Epoch {epoch}: saved best multimodal weights (Acc={best_acc:.4f})")

        early_stopper(avg_train_loss)
        if early_stopper.early_stop:
            print(f"Multimodal early stopping at epoch {epoch}")
            break

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    train()
