---
date: 2025-04-14T04:00:59-07:00
description: ""
featured_image: "/images/pytorch/pia.jpg"
tags: ["data processing"]
title: "PyTorch"
---

### **1. 基础概念**

**DataLoader**：包装 `Dataset`，提供批量迭代、多线程加载等功能。

- **Dataset** (必需)：存储样本及其标签，需实现 `__len__` 和 `__getitem__` 方法。
- **`batch_size`**：每批的样本数。默认值为1。
- **`shuffle`**：是否在每个epoch开始时打乱数据（训练集通常设为True，验证集False）。
- **`num_workers`**：加载数据的子进程数。默认为0，意味着数据将在主进程中加载。
- **`drop_last`**：当样本数不能被batch_size整除时，是否丢弃最后一个不完整的批次。
- `sampler` (可选)：定义从数据集中抽取样本的策略。如果指定，则忽略shuffle参数。
- `batch_sampler` (可选)：与 sampler 类似，但一次返回一个批次的索引。不能与 batch_size、shuffle 和 sampler 同时使用。
- `collate_fn` (可选)：如何将多个数据样本整合成一个批次。通常不需要指定。
- `drop_last` (可选)：如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次。默认为False。

---

### **2. 创建自定义Dataset**
假设数据是随机生成的张量，实际场景中可能需要从文件读取。

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data      # 样本数据（例如图像）
        self.labels = labels  # 样本标签
    
    def __len__(self):
        return len(self.data)  # 返回数据总数量
    
    def __getitem__(self, idx):
        # 返回单个样本及其标签（自动组成批次由DataLoader处理）
        return self.data[idx], self.labels[idx]
```

<!--more-->

---

### **3. 生成示例数据**
```python
# 假设有100个样本，每个样本是3通道的32x32图像
data = torch.randn(100, 3, 32, 32)
# 对应的标签是0-9的整数
labels = torch.randint(0, 10, (100,))
```

---

### **4. 实例化Dataset和DataLoader**
```python
dataset = CustomDataset(data, labels)
dataloader = DataLoader(
    dataset,
    batch_size=4,    # 每批4个样本
    shuffle=True,    # 每个epoch打乱数据
    num_workers=2,   # 使用2个子进程加载数据
)
```

---

### **5. 遍历DataLoader**
```python
for batch_data, batch_labels in dataloader:
    # batch_data形状: [4, 3, 32, 32]
    # batch_labels形状: [4]
    print("Batch data shape:", batch_data.shape)
    print("Batch labels shape:", batch_labels.shape)
    # 在此处执行模型训练/验证...
```

---

### **6. 处理真实数据（示例：图像分类）**
若处理图像文件，需在 `Dataset` 中读取图片并预处理：

```python
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform  # 数据增强（如归一化、裁剪等）
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 示例：定义数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # 将PIL图像转为Tensor，并归一化到[0,1]
])

# 假设file_paths和labels是预先准备好的列表
dataset = ImageDataset(file_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

