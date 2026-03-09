
# Flower CNN

## 概述

`flower_cnn` 是一个用于花卉物种数据集卷积模型实验的研究代码库。项目将数据处理、模型训练和部署解耦，便于实验和复现。

## 环境配置

1. 安装依赖：`pip install -r requirements.txt`
2. 数据集准备：将原始数据放入 `data/raw/`，每个数据集分割需遵循 `experiments/` 相关脚本要求的结构（如 `images/<label>/*.jpg`）。
3. 数据预处理：可使用 `datasets/` 或 `training/` 下的脚本，将数据处理后存入 `data/processed/`。

## 使用方法

- **训练**：运行 [training/train_visible.py](training/train_visible.py) 启动训练。可通过配置文件或命令行参数指定模型、数据集路径和超参数。
- **评估**：在 `experiments/` 下编写脚本，加载 `models/` 中的模型，对验证集进行推理和评分。
- **日志**：输出、模型权重和指标会保存在 `training/logs/` 和 `experiments/logs/`，每次运行建议用时间戳新建子目录。

## 监控

[training/train_visible.py](training/train_visible.py)、[training/train_ir.py](training/train_ir.py) 和 [training/train_multimodal.py](training/train_multimodal.py) 均会将 TensorBoard 日志写入 `experiments/logs/`。可通过如下命令启动 TensorBoard：

```
tensorboard --logdir experiments/logs --host 0.0.0.0 --port 6006
```

浏览器访问 http://localhost:6006 对比不同模态的实验结果。

## 实验流程

1. 每个实验在 `experiments/` 下新建目录（如 `experiments/ablation/`）。
2. 若需固定标签映射，可在 `data/annotations/` 版本化数据集分割文件。
3. 模型权重建议以 `flowercnn-{backbone}-{date}.pth` 命名，保存在 `models/`。
4. 实验元数据（如超参数、数据集校验和）建议记录在 `experiments/logs/metadata.json` 或类似文件，便于复现。

## 部署与服务

- `web/` 目录包含最小化的后端/前端演示。序列化代码放在 `web/backend/`，静态资源在 `web/frontend/`，模板在 `web/templates/`。
- `model_web/` 可用于存放 API 封装或集成训练模型的 Flask/Django 应用。

## 脚本说明

1. `training/train_ir.py` 类似于 `train_visible.py`，但针对红外数据（`data/processed/ir` 和 `data/annotations/*_ir.txt`），使用 `ReduceLROnPlateau`、TensorBoard（`experiments/logs/ir`），最佳权重保存在 `experiments/checkpoints/best_ir.pth`。
2. `training/train_multimodal.py` 用于可见光/红外配对数据（CSV 格式：`vis.jpg,ir.jpg,label`），融合模型见 `models/multi_modal.py`，日志写入 `experiments/logs/multimodal`，权重为 `best_multimodal.pth`。
3. `scripts/inference.py` 支持单张图片（可见光/红外）或配对图片（多模态）推理。通过 `--mode` 切换，指定 `--checkpoint`，并传入 `--image` 或 `--vis-image` + `--ir-image`。
4. `scripts/export_onnx.py` 将 `models/flower_model.py` 导出为 ONNX 格式（`models/flower_model.onnx`），支持动态 batch size。

## 手动演示

1. 将模型权重放在 [model_web/model.pth](model_web/model.pth)，供 [web/backend/inference.py](web/backend/inference.py) 加载。
2. 在项目根目录启动 Flask 服务：

	```
	FLASK_APP=web.backend.app flask run --host 0.0.0.0 --port 5000
	```

	此时 [web/backend/app.py](web/backend/app.py) 会提供 `/predict` 接口和前端页面。
3. 浏览器访问 http://localhost:5000，上传花卉图片，查看 `/predict` 返回的预测结果。

## 贡献指南

1. 修改请遵循目录结构（如 data、training、web 等）。
2. 每次实验建议在 `docs/thesis/` 或 `docs/diagrams/` 记录，并交叉引用相关代码配置。
3. 提交前请运行 `git status`，确保所有新文件已纳入版本控制。

欢迎提交 issue 或 pull request，贡献新实验或改进建议。
花卉图像来源
下载解压太慢，找国内镜像源，去GitHub仓库拉取，解压还是太慢，使用多线程分片下载优化解压速度。
预处理的步骤，做什么处理-灰度化，二值化。
模型训练基于什么原理方法
调用的什么模型
预期匹配率，实际匹配率，对比分析，原因以及改进
系统项目的可拓展性-添加动植物数据集进行识别
系统项目的创新性，局限性。

为什么要下载预训练权重？
因为：
• MobileNetV3 在 ImageNet 上训练过
• 预训练权重能让模型“带着知识”开始训练
• 在小数据集（比如 Oxford 102 Flowers）上效果会好很多
• 没有预训练，模型前几 epoch 会非常弱
所以这是 正常且必要的步骤。

你的整理脚本逻辑整体正确，但存在 两个关键错误，这两个错误会直接导致：
• 训练集标签与图片不匹配
• 模型完全学不到东西
• 验证准确率 ≈ 0.0098（随机猜测水平）

[优化报告](docs/OPTIMIZATION.md)

第一次训练：Epoch 0: New best model saved! Acc=0.1232
Epoch 0 | Train Loss: 4.1821 | Val Loss: 4.3195 | Val Acc: 0.1232
Epoch 1: New best model saved! Acc=0.2282
Epoch 1 | Train Loss: 2.6564 | Val Loss: 3.6790 | Val Acc: 0.2282
Epoch 2: New best model saved! Acc=0.4235
Epoch 2 | Train Loss: 1.8942 | Val Loss: 2.8076 | Val Acc: 0.4235
Epoch 3: New best model saved! Acc=0.5956
Epoch 3 | Train Loss: 1.5874 | Val Loss: 2.1594 | Val Acc: 0.5956
Epoch 4: New best model saved! Acc=0.7254
Epoch 4 | Train Loss: 1.4046 | Val Loss: 1.7986 | Val Acc: 0.7254
Epoch 5: New best model saved! Acc=0.7369
Epoch 5 | Train Loss: 1.2813 | Val Loss: 1.7701 | Val Acc: 0.7369
Epoch 6: New best model saved! Acc=0.7649
Epoch 6 | Train Loss: 1.2458 | Val Loss: 1.6716 | Val Acc: 0.7649
Epoch 7: New best model saved! Acc=0.7655
Epoch 7 | Train Loss: 1.1839 | Val Loss: 1.6533 | Val Acc: 0.7655
Epoch 8: New best model saved! Acc=0.7912
Epoch 8 | Train Loss: 1.1274 | Val Loss: 1.5811 | Val Acc: 0.7912
Epoch 9 | Train Loss: 1.0988 | Val Loss: 1.5655 | Val Acc: 0.7910
Epoch 10: New best model saved! Acc=0.7991
Epoch 10 | Train Loss: 1.0975 | Val Loss: 1.5267 | Val Acc: 0.7991
Epoch 11: New best model saved! Acc=0.8022
Epoch 11 | Train Loss: 1.0931 | Val Loss: 1.5157 | Val Acc: 0.8022
Epoch 12 | Train Loss: 1.0523 | Val Loss: 1.6538 | Val Acc: 0.7815
Epoch 13: New best model saved! Acc=0.8513
Epoch 13 | Train Loss: 1.0414 | Val Loss: 1.4385 | Val Acc: 0.8513
Epoch 14 | Train Loss: 1.0252 | Val Loss: 1.4777 | Val Acc: 0.8185
Epoch 15 | Train Loss: 0.9998 | Val Loss: 1.4644 | Val Acc: 0.8401
Epoch 16: New best model saved! Acc=0.8528
Epoch 16 | Train Loss: 0.9977 | Val Loss: 1.3890 | Val Acc: 0.8528
Epoch 17 | Train Loss: 1.0010 | Val Loss: 1.4346 | Val Acc: 0.8354
Epoch 18: New best model saved! Acc=0.8616
Epoch 18 | Train Loss: 0.9809 | Val Loss: 1.3849 | Val Acc: 0.8616
Epoch 19 | Train Loss: 0.9540 | Val Loss: 1.4465 | Val Acc: 0.8401
Epoch 20 | Train Loss: 0.9456 | Val Loss: 1.4341 | Val Acc: 0.8401
Epoch 21 | Train Loss: 0.9618 | Val Loss: 1.4006 | Val Acc: 0.8506
Epoch 22 | Train Loss: 0.9691 | Val Loss: 1.4319 | Val Acc: 0.8518
Early stopping at epoch 23
早停机制正常触发。训练在 epoch 23 停止是因为验证损失连续 5 个 epoch 没有改善（patience=5）。
