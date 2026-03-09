import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 通用训练基类：封装训练/验证循环、日志记录、早停与最佳模型保存

class TrainerBase:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, scheduler, log_dir, best_model_path, batch_size=32, patience=5, device=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.best_model_path = best_model_path
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.best_acc = 0.0
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

    def train(self, epochs, accuracy_fn=None):
        # 标准训练-验证循环，记录损失与精度并应用学习率调度
        for epoch in range(epochs):
            self.model.train()
            train_loss_total = 0.0
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

            for batch in train_loader:
                imgs, labels = self._unpack_batch(batch)
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.model(imgs)
                loss = self.criterion(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_total += loss.item()
            avg_train_loss = train_loss_total / len(train_loader)

            self.model.eval()
            val_loss_total = 0.0
            accs = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs, labels = self._unpack_batch(batch)
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    out = self.model(imgs)
                    loss = self.criterion(out, labels)
                    val_loss_total += loss.item()
                    if accuracy_fn:
                        accs.append(accuracy_fn(out, labels))
            avg_val_loss = val_loss_total / len(val_loader)
            avg_val_acc = sum(accs) / len(accs) if accs else 0.0

            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
            # 兼容 ReduceLROnPlateau 和 CosineAnnealingLR
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            if avg_val_acc > self.best_acc:
                self.best_acc = avg_val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Epoch {epoch}: New best model saved! Acc={self.best_acc:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        self.writer.close()

    def _unpack_batch(self, batch):
        # 支持多模态(batch为tuple)和单模态(batch为tensor)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            # 多模态：vis_img, ir_img, label
            imgs = torch.cat([batch[0].unsqueeze(1), batch[1].unsqueeze(1)], dim=1) if batch[0].dim() == 3 else batch[0]
            labels = batch[2]
            return imgs, labels
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        else:
            raise ValueError("Unknown batch format")
