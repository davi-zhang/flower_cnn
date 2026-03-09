#!/usr/bin/env python3
"""
多数据源融合数据准备脚本

支持的数据源：
1. Oxford102 花卉数据集 (102类)
2. Kaggle Flowers 花卉数据集 (5类)

功能：
- 统一整理多个数据源到 data/raw/images 目录
- 生成统一的标签映射和划分文件
- 支持选择性加载数据源
- 支持跨数据源类别映射融合（Kaggle 五类映射到 Oxford 既有类）

使用方法：
    python scripts/prepare_multi_source.py --sources oxford102 kaggle_flowers
    python scripts/prepare_multi_source.py --sources oxford102  # 仅 Oxford102
    python scripts/prepare_multi_source.py --sources kaggle_flowers  # 仅 Kaggle Flowers
"""

import argparse
import json
import os
import random
import shutil
from typing import Dict, List, Tuple

from scipy.io import loadmat
from tqdm import tqdm

# =============================================================================
# 配置
# =============================================================================
DATA_SOURCE_ROOT = "data_source"
RAW_OUT = "data/raw/images"
ANNOT_OUT = "data/annotations"

# 数据源配置
SOURCES_CONFIG = {
    "oxford102": {
        "name": "Oxford 102 Flowers",
        "path": "oxford102",
        "num_classes": 102,
        "type": "mat_labels",  # 使用 .mat 文件标签
    },
    "kaggle_flowers": {
        "name": "Kaggle Flowers",
        "path": "Kaggle Flowers",
        "num_classes": 0,
        "type": "folder_structure",  # 使用文件夹结构作为标签
    },
}

# Kaggle Flowers -> Oxford102 类别映射（按用户指定）
# daisy -> class_0
# dandelion -> class_1
# rose -> class_52
# sunflower -> class_54
# tulip -> class_98
KAGGLE_TO_OXFORD_LABEL = {
    "daisy": 0,
    "dandelion": 1,
    "rose": 52,
    "sunflower": 54,
    "tulip": 98,
}


# =============================================================================
# 数据源处理函数
# =============================================================================

def prepare_oxford102(source_path: str, class_offset: int) -> Tuple[List[dict], Dict[str, int]]:
    """
    处理 Oxford102 数据集
    
    Args:
        source_path: 数据源根路径
        class_offset: 类别ID偏移量（用于多数据源融合）
    
    Returns:
        samples: 样本列表 [{"src": ..., "label": ..., "split": ...}, ...]
        label2id: 标签映射字典
    """
    print(f"\n[Oxford102] 开始处理...")
    
    # 查找图像目录
    jpg_dir = os.path.join(source_path, "jpg")
    images_dir = os.path.join(source_path, "images")
    
    if os.path.isdir(jpg_dir):
        img_dir = jpg_dir
    elif os.path.isdir(images_dir):
        img_dir = images_dir
    else:
        raise RuntimeError(f"[Oxford102] 未找到 jpg/ 或 images/ 文件夹: {source_path}")
    
    # 读取文件和标签
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    labels_mat = loadmat(os.path.join(source_path, "imagelabels.mat"))["labels"][0]
    labels_mat = labels_mat - 1  # 转为 0-based
    
    if len(labels_mat) != len(files):
        raise RuntimeError(f"[Oxford102] 标签数 {len(labels_mat)} != 图片数 {len(files)}")
    
    # 读取官方划分
    setid = loadmat(os.path.join(source_path, "setid.mat"))
    train_ids = set(setid["trnid"][0] - 1)
    val_ids = set(setid["valid"][0] - 1)
    test_ids = set(setid["tstid"][0] - 1)
    
    # 构建样本列表
    samples = []
    for i, fname in enumerate(files):
        label = int(labels_mat[i]) + class_offset
        if i in train_ids:
            split = "train"
        elif i in val_ids:
            split = "val"
        else:
            split = "test"
        
        samples.append({
            "src": os.path.join(img_dir, fname),
            "filename": fname,
            "label": label,
            "split": split,
            "source": "oxford102",
        })
    
    # 构建标签映射
    label2id = {f"oxford102_class_{i}": i + class_offset for i in range(102)}
    
    print(f"[Oxford102] 处理完成: {len(samples)} 张图片, 102 类 (offset={class_offset})")
    return samples, label2id


def prepare_kaggle_flowers(source_path: str, class_offset: int,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> Tuple[List[dict], Dict[str, int]]:
    """
    处理 Kaggle Flowers 数据集
    
    Args:
        source_path: 数据源根路径（包含 daisy/dandelion/rose/sunflower/tulip）
        class_offset: 类别ID偏移量（此数据源映射到 Oxford 固定类别，实际不使用）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        samples: 样本列表
        label2id: 标签映射字典
    """
    print(f"\n[KaggleFlowers] 开始处理...")
    
    if not os.path.isdir(source_path):
        raise RuntimeError(f"[KaggleFlowers] 路径不存在: {source_path}")
    
    # 获取所有类别文件夹
    class_dirs = sorted([d for d in os.listdir(source_path) 
                         if os.path.isdir(os.path.join(source_path, d))])
    
    print(f"[KaggleFlowers] 检测到 {len(class_dirs)} 个类别")

    if len(class_dirs) == 0:
        raise RuntimeError("[KaggleFlowers] 未检测到任何类别目录")
    
    samples = []
    label2id = {}
    
    for class_idx, class_name in enumerate(class_dirs):
        if class_name not in KAGGLE_TO_OXFORD_LABEL:
            raise RuntimeError(
                f"[KaggleFlowers] 检测到未映射类别: {class_name}，"
                f"请先在 KAGGLE_TO_OXFORD_LABEL 中配置映射"
            )
        class_path = os.path.join(source_path, class_name)
        label = KAGGLE_TO_OXFORD_LABEL[class_name]
        label2id[f"kaggle_{class_name}"] = label
        
        # 获取该类别所有图片
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 随机划分
        random.seed(42 + class_idx)  # 固定随机种子保证可复现
        random.shuffle(images)
        
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        for i, fname in enumerate(images):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            
            samples.append({
                "src": os.path.join(class_path, fname),
                "filename": fname,
                "label": label,
                "split": split,
                "source": "kaggleflowers",
                "class_name": class_name,
            })
    
    print(f"[KaggleFlowers] 处理完成: {len(samples)} 张图片, {len(class_dirs)} 类 (映射到 Oxford 类别)")
    return samples, label2id


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="多数据源融合数据准备脚本")
    parser.add_argument(
        "--sources", 
        nargs="+", 
        choices=list(SOURCES_CONFIG.keys()),
        default=["oxford102"],
        help="要处理的数据源列表"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="清空输出目录后再处理"
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="图片处理方式: copy(复制) 或 symlink(符号链接)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("多数据源融合数据准备脚本")
    print("=" * 60)
    print(f"选择的数据源: {args.sources}")
    
    # 清空输出目录
    if args.clean:
        print("\n清空输出目录...")
        if os.path.exists(RAW_OUT):
            shutil.rmtree(RAW_OUT)
        if os.path.exists(ANNOT_OUT):
            shutil.rmtree(ANNOT_OUT)
    
    os.makedirs(RAW_OUT, exist_ok=True)
    os.makedirs(ANNOT_OUT, exist_ok=True)
    
    # 处理各数据源
    all_samples = []
    all_label2id = {}
    class_offset = 0
    source_stats = {}
    
    for source_name in args.sources:
        config = SOURCES_CONFIG[source_name]
        source_path = os.path.join(DATA_SOURCE_ROOT, config["path"])
        
        if source_name == "oxford102":
            samples, label2id = prepare_oxford102(source_path, class_offset)
        elif source_name == "kaggle_flowers":
            samples, label2id = prepare_kaggle_flowers(source_path, class_offset)
        else:
            raise ValueError(f"未知数据源: {source_name}")
        
        all_samples.extend(samples)
        all_label2id.update(label2id)
        source_labels = sorted({s["label"] for s in samples}) if samples else []
        source_stats[source_name] = {
            "num_samples": len(samples),
            "num_classes": len(source_labels),
            "class_range": (f"{source_labels[0]} - {source_labels[-1]}" if source_labels else "N/A"),
        }
        class_offset += config["num_classes"]
    
    # 以“最大标签 + 1”作为模型输出维度，兼容跨源标签映射
    total_classes = (max(sample["label"] for sample in all_samples) + 1) if all_samples else 0
    print(f"\n总计: {len(all_samples)} 张图片, {total_classes} 个类别")
    
    # 创建类别文件夹并复制/链接图片
    print("\n整理图片...")
    used_labels = sorted({sample["label"] for sample in all_samples})
    for label in used_labels:
        os.makedirs(os.path.join(RAW_OUT, f"class_{label}"), exist_ok=True)
    
    for sample in tqdm(all_samples, desc="处理图片"):
        dst_dir = os.path.join(RAW_OUT, f"class_{sample['label']}")
        # 添加数据源前缀避免文件名冲突
        dst_filename = f"{sample['source']}_{sample['filename']}"
        dst = os.path.join(dst_dir, dst_filename)
        
        if args.copy_mode == "copy":
            shutil.copy(sample["src"], dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(sample["src"]), dst)
        
        sample["dst_path"] = f"images/class_{sample['label']}/{dst_filename}"
    
    # 生成标签映射文件
    label2id_path = os.path.join(ANNOT_OUT, "label2id.json")
    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(all_label2id, f, indent=4, ensure_ascii=False)
    print(f"\nlabel2id.json 已生成 ({len(all_label2id)} 类)")
    
    # 生成数据源统计文件
    stats = {
        "total_samples": len(all_samples),
        "total_classes": total_classes,
        "sources": source_stats,
    }
    stats_path = os.path.join(ANNOT_OUT, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"dataset_stats.json 已生成")
    
    # 生成划分文件
    splits = {"train": [], "val": [], "test": []}
    for sample in all_samples:
        splits[sample["split"]].append(sample)
    
    for split_name, split_samples in splits.items():
        split_path = os.path.join(ANNOT_OUT, f"{split_name}.txt")
        with open(split_path, "w") as f:
            for sample in split_samples:
                f.write(f"{sample['dst_path']},{sample['label']}\n")
        print(f"{split_name}.txt 已生成 ({len(split_samples)} 样本)")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    for source_name, stat in source_stats.items():
        print(f"  {source_name}:")
        print(f"    - 样本数: {stat['num_samples']}")
        print(f"    - 类别数: {stat['num_classes']}")
        print(f"    - 类别范围: {stat['class_range']}")
    print(f"\n  总计:")
    print(f"    - 样本数: {len(all_samples)}")
    print(f"    - 类别数: {total_classes}")
    print(f"    - 训练集: {len(splits['train'])}")
    print(f"    - 验证集: {len(splits['val'])}")
    print(f"    - 测试集: {len(splits['test'])}")
    print("\n🎉 多数据源融合完成！")


if __name__ == "__main__":
    main()
