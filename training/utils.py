from pathlib import Path

import torch

# 训练辅助函数：计算精度、保存检查点以及简单早停机制


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    # 计算分类准确率
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    # 保存模型权重到指定路径
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        # 监控验证损失，触发早停
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
