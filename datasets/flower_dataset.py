import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

# 简单的图像分类数据集读取器：从标注文件解析相对路径与标签

Transform = Callable[[Image.Image], Image.Image]


class FlowerDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotations: str,
        transform: Optional[Transform] = None,
    ):
        # 解析标注文件，每行格式：路径 标签
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(annotations, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid annotation line: {line}")
                rel_path, label = parts
                self.samples.append((rel_path.strip(), int(label.strip())))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        # 加载图片并应用预处理，返回图像与标签
        rel_path, label = self.samples[index]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
