---
date: 2025-01-31T11:00:59-04:00
description: ""
featured_image: "/images/CV/lucky.jpg"
tags: ["deep learning", "CV"]
title: "目标检测"
---

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

&nbsp; <!--more-->

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





