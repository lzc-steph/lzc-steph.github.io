---
date: 2024-12-27T11:00:59-04:00
description: "计算机视觉旨在使计算机能够从图像或视频中“看”并理解视觉信息。它结合了图像处理、模式识别、机器学习和深度学习等技术，模拟人类视觉系统的功能，从而实现对视觉数据的分析和理解。"
featured_image: "/images/chapter1/taytay.HEIC"
tags: ["deep learning"]
title: "CV"
---

问题: 处理高分辨率图像时，原始图像的像素数量通常非常庞大。

类型：root node, decision node, leaf node

### 解决方案：CNN

卷积神经网络通过引入卷积操作，有效地解决了大图片参数过大的问题。

1. **Automatic Feature Extraction**

   CNNs automatically learn hierarchical features from raw data (like images) without needing manual feature engineering. 

2. **Parameter Sharing**

   In a CNN, the same filter (or kernel) is applied across the entire image. This concept of **weight sharing** significantly reduces the number of parameters compared to fully connected networks, making CNNs more computationally efficient. 

3. **分层特征学习**
4. **Translation Invariance**（pooling layers）
5. **Reduced Computational Complexity**
6. **Robustness to Small Variations in Input**
7. **Transfer Learning**



## 边缘检测

1. **垂直边缘检测滤波**

   ![6](/images/CV/6.png)

2. **变权**

   ![7](/images/CV/7.png)

3. **利用反向传播学习**

### Padding

1. Padding：外层填充像素

   - 存在问题：
     - throw away information from edge
     - shranky output

   解决：**外层填充像素**

2. 填充多少像素？

   + **Valid:** (n-f+1) * (n-f+1)

   + **Same**: Pad so that output size is the same as the input size.

### 三维卷积

![8](/images/CV/8.png)

第一个可能是垂直边缘检测器，第二个可能是水平边缘检测器，将结果叠加。

卷积网络层的类型: convolution  + pooling + fully connected

## **池化层（Pooling Layer）**

池化层是深度学习中卷积神经网络（CNN）的一个重要组成部分，主要用于**降低特征图的尺寸加速计算，同时保留主要特征。**

例如：Max pooling, Average pooling...



