import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

Transform = Callable[[Image.Image], Image.Image]


class FlowerDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotations: str,
        transform: Optional[Transform] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(annotations, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split(",")
                self.samples.append((rel_path.strip(), int(label.strip())))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        rel_path, label = self.samples[index]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
