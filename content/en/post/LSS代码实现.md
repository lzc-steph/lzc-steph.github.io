---
date: 2025-04-14T04:00:59-07:00
description: ""
featured_image: "/images/lssCode/pia.jpg"
tags: ["CV"]
title: "LSS代码"
---

文章搬运，自用。

论文：[Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)

官方源码：[lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)

&nbsp;

## [Lift, Splat, Shoot图像BEV安装与模型代码详解](https://blog.csdn.net/qq_41366026/article/details/133840315)

对于任意数量不同相机帧的图像直接提取场景的BEV表达；主要由三部分实现：

1. **Lift**：将每一个相机的图像帧根据相机的内参转换提升到 frustum（锥形）形状的点云空间中。
2. **splate**：将所有相机转换到锥形点云空间中的特征根据相机的内参 K 与相机相对于 ego 的外参T映射到栅格化的 3D 空间（BEV）中来融合多帧信息。
3. **shoot**：根据上述 BEV 的检测或分割结果来生成规划的路径 proposal；从而实现可解释的端到端的路径规划任务。

注：LSS在训练的过程中并不需要激光雷达的点云来进行监督。

<!--more-->

