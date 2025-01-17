---
date: 2024-12-27T11:00:59-04:00
description: "计算机视觉旨在使计算机能够从图像或视频中“看”并理解视觉信息。它结合了图像处理、模式识别、机器学习和深度学习等技术，模拟人类视觉系统的功能，从而实现对视觉数据的分析和理解。"
featured_image: "/images/CV/taytay.HEIC"
tags: ["deep learning"]
title: "CV"
---

问题: 处理高分辨率图像时，原始图像的像素数量通常非常庞大。

类型：root node, decision node, leaf node
&nbsp; 

## 1. 解决方案：CNN

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

&nbsp; 

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


&nbsp; 
&nbsp; 

---

## 3. 目标检测

![19](/images/CV/19.png)

- 技巧

  **Ensembling**：Train several networks independently and average their outputs **Multi-crop at test time**：Run classifier on multiple versions of test images and average results


&nbsp; 

### 定位

![20](/images/CV/20.png)

Need to output bx, by, bn, bw, class label (1-4)

![21](/images/CV/21.png)

需人工标注**特征点的坐标**

&nbsp; 

### 基于滑动窗口的目标检测算法

1. 先训练卷积网络识别物体
2. 滑动+放大窗口+再次滑动

**问题**：计算效率大，慢

&nbsp; 

### 在卷积层上的滑动窗口目标检测算法

一次得到所有的预测值。**问题**：可能不准确

&nbsp; 

### yolo **边界框**

**图像划分、边界框预测、类预测、最终预测**

![22](/images/CV/22.png)

物体只会被分配到其中一个格子（观察对象的中点）

单次卷积实现，共享计算

**优点**：快

**评估指标**：交并比IoU函数（"Correct" if loU ≥ 0.5）

&nbsp; 

### 非极大值抑制（NMS）

确保只检测出一次

1. 当p小于阈值，去除重复框。
2. 选择最高概率的
3. 去除和高亮标记边界框重叠面积高的边框
4. 循环

若n个对象，则做n次非极大值抑制

&nbsp; 

### Anchor box

解决两个对象同时出现在同一个单位中

![23](/images/CV/23.png)

观察哪个Anchor box的IoU更高

可以手工决定Anchor box形状大小，也可以通过**K-means算法**重新调整Anchor Boxes，保证最优的边界框预测。

&nbsp; 

### YOLO

![24](/images/CV/24.png)

![25](/images/CV/25.png)

最后，跑非极大值抑制

![26](/images/CV/26.png)

&nbsp; 

### 候选区域

&nbsp; 

---

## 4. 人脸识别







