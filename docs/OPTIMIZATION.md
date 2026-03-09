# 花卉分类项目优化文档

## 一、问题分析

### 1.1 原始问题
- 训练 2 个 epoch 后验证准确率仅 ~1%，预测置信度极低
- 102 类分类任务，随机猜测准确率约 0.98%，说明模型几乎未学习

### 1.2 根因分析
| 问题类别 | 具体问题 | 影响 |
|---------|---------|------|
| 模型初始化 | MobileNetV3 使用随机权重 (`weights=None`) | 需要更长时间收敛，小数据集易过拟合 |
| 数据增强 | 仅使用简单翻转、旋转、色彩抖动 | 泛化能力不足 |
| 优化器 | Adam 无权重衰减 | 容易过拟合 |
| 学习率调度 | ReduceLROnPlateau 响应较慢 | 收敛效率低 |
| 损失函数 | 标准交叉熵 | 对过拟合敏感 |
| 数据加载 | 单线程加载，无 pin_memory | GPU 利用率低 |

---

## 二、优化措施

### 2.1 迁移学习（关键优化）
**文件**: `models/mobilenetv3_base.py`

```python
# 优化前
model = mobilenet_v3_small(weights=None)

# 优化后
from torchvision.models import MobileNet_V3_Small_Weights
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
model = mobilenet_v3_small(weights=weights)
```

**效果**: ImageNet 预训练权重提供良好的特征提取基础，预计准确率提升 30-50%

---

### 2.2 数据增强增强
**文件**: `datasets/transforms.py`

| 增强方法 | 原始 | 优化后 |
|---------|------|--------|
| 裁剪 | `Resize(224, 224)` | `RandomResizedCrop(224, scale=(0.8, 1.0))` |
| 色彩 | `ColorJitter(0.2, 0.2, 0.2)` | `ColorJitter(0.3, 0.3, 0.3, 0.1)` |
| 自动增强 | 无 | `AutoAugment(IMAGENET)` |
| 随机擦除 | 无 | `RandomErasing(p=0.2)` |

**效果**: 增加数据多样性，防止过拟合，预计泛化提升 5-10%

---

### 2.3 优化器与学习率调度
**文件**: `training/train_visible.py`

```python
# 优化前
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# 优化后
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**效果**:
- AdamW 解耦权重衰减，正则化效果更好
- CosineAnnealingLR 平滑调整学习率，收敛更稳定

---

### 2.4 标签平滑
**文件**: `training/train_visible.py`

```python
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

**效果**: 防止模型过度自信，提升泛化能力

---

### 2.5 数据加载优化
**文件**: `training/trainer_base.py`

```python
# 优化前
DataLoader(dataset, batch_size=32, shuffle=True)

# 优化后
DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
```

**效果**: 多线程预取数据，GPU 利用率提升 20-50%

---

## 三、预期效果

| 指标 | 优化前 (2 epoch) | 优化后 (50 epoch) |
|------|-----------------|-------------------|
| 验证准确率 | ~1% | **70-85%** |
| 训练时间/epoch | ~60s | ~45s (数据加载优化) |
| 收敛速度 | 慢 | 快 (迁移学习) |
| 泛化能力 | 差 | 好 (数据增强+正则化) |

---

## 四、运行命令

### 4.1 全量训练
```bash
cd /home/david/flower_cnn
source .venv/bin/activate
PYTHONPATH="." python training/train_visible.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --num-classes 102
```

### 4.2 查看训练日志
```bash
tensorboard --logdir experiments/logs/tensorboard
```

### 4.3 导出与推理
```bash
PYTHONPATH="." python scripts/export_onnx.py --num-classes 102
PYTHONPATH="." python scripts/inference.py --image <图片路径> --num-classes 102 --topk 5
```

---

## 五、后续优化建议

1. **混合精度训练**: 使用 `torch.cuda.amp` 加速训练
2. **学习率预热**: 前几个 epoch 使用较小学习率
3. **更大模型**: 尝试 MobileNetV3-Large 或 EfficientNet
4. **测试时增强 (TTA)**: 推理时使用多次增强取平均
5. **知识蒸馏**: 用大模型指导小模型训练

---

*文档生成时间: 2026-03-06*
