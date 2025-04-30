---
date: 2024-12-27T11:00:59-04:00
description: "计算机视觉旨在使计算机能够从图像或视频中“看”并理解视觉信息。它结合了图像处理、模式识别、机器学习和深度学习等技术，模拟人类视觉系统的功能，从而实现对视觉数据的分析和理解。"
featured_image: "/images/CV/taytay.HEIC"
tags: ["CV"]
title: "CV"
---

问题: 处理高分辨率图像时，原始图像的像素数量通常非常庞大。

### 边缘检测

1. **垂直边缘检测滤波**

   ![6](/images/CV/6.png)

2. **变权**

   ![7](/images/CV/7.png)

3. **利用反向传播学习**

&nbsp; 

### Padding

1. Padding：外层填充像素

   - 存在问题：
     - throw away information from edge
     - shranky output

   解决：**外层填充像素**

2. 填充多少像素？

   + **Valid:** (n-f+1) * (n-f+1)

   + **Same**: Pad so that output size is the same as the input size.

&nbsp; 
<!--more-->

### 三维卷积

![8](/images/CV/8.png)

第一个可能是垂直边缘检测器，第二个可能是水平边缘检测器，将结果叠加。

卷积网络层的类型: convolution  + pooling + fully connected

&nbsp; 

### **池化层（Pooling Layer）**

池化层是深度学习中卷积神经网络（CNN）的一个重要组成部分，主要用于**降低特征图的尺寸加速计算，同时保留主要特征。**

例如：Max pooling, Average pooling...

&nbsp; 

### 完整卷积神经网络

![10](/images/CV/10.png)

- 常见结构

  CONV-POOL-CONV-POOL-FC-FC-FC-softmax

  使用梯度下降来减少代价函数进行训练

- 卷积优势

  - **Parameter sharing**:

    A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.

  - **Sparsity of connections**:

    In each layer, each output value depends only on a small number of inputs.
  
- **平移不变性**

  含义：即使目标的外观发生了某种变化，但是你依然可以把它识别出来。

  - **卷积**：在神经网络中，卷积被定义为不同位置的特征检测器，也就意味着，**无论目标出现在图像中的哪个位置，它都会检测到同样的这些特征，输出同样的响应。**
  - **池化**：比如最大池化，它返回感受野中的最大值，**如果最大值被移动了，但是仍然在这个感受野中，那么池化层也仍然会输出相同的最大值。**这就有点平移不变的意思了。

  即使图像被平移，卷积保证仍然能检测到它的特征，池化则尽可能地保持一致的表达。



&nbsp; 
&nbsp; 

---

## 2. 经典卷积神经网络架构

### LeNet - 5

![11](/images/CV/11.png)

### AlexNet

![12](/images/CV/12.png)

### VGG - 16

![13](/images/CV/13.png)

### 残差网络

- #### 残差块 (Residual block)

  ![14](/images/CV/14.png)

  使用残差可以训练更深的网络

- #### 残差网络（Residual Network）

  ![16](/images/CV/16.png)

  ![15](/images/CV/15.png)

- 增加残差不会影响原始新性能，还可能提升性能

  ![17](/images/CV/17.png)

### 1*1 卷积

![18](/images/CV/18.png)
