---
date: 2025-02-21T11:00:59-04:00
description: ""
featured_image: "/images/lss/lucky.jpg"
tags: ["RL"]
title: "Lift-splat-shoot"
---

### 1. 语义分割

**目标**：将图像中的每个像素分配一个语义类别标签。

- **输入**：一张RGB图像（或其他类型的图像，如深度图、红外图等）。
- **输出**：像素级标签图，标注出道路、车辆、行人、交通标志等类别。

LSS生成的BEV特征可以直接用于 **BEV空间的语义分割**，例如对BEV栅格中的每个位置进行类别预测。

&nbsp;

### 2. 论文阅读

 [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)

&nbsp;

### 3. 基础知识学习

通过**显式**估计图像的深度信息，对采集到的环视图像进行特征提取，并根据估计出来的离散深度信息，实现**图像特征向BEV特征的转换**，进而完成自动驾驶中的语义分割任务。

+ 核心流程：

  - **Lift**：将2D图像特征显式提升到3D空间（通过深度估计生成视锥特征）。

  - **Splat**：将3D特征“展开”到BEV空间，构建鸟瞰图特征。

  - **Shoot**：基于BEV特征进行运动规划或轨迹预测。