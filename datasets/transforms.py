from torchvision import transforms
from torchvision.transforms import autoaugment

# 定义训练与验证阶段的图像预处理与增强
# 优化：使用 RandomResizedCrop 保留更多尺度信息，添加 AutoAugment 与 RandomErasing 提高泛化

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=0.2),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
