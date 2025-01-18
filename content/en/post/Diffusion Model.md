---
date: 2025-01-17T10:00:59-04:00
description: "扩散模型是一种生成模型，主要用于生成与给定数据集相似的数据。它的核心思想是通过逐步添加噪声（扩散过程）和逐步去噪（逆扩散过程）来学习数据的分布，从而生成高质量的数据样本。"
featured_image: "/images/DM/taytay1.HEIC"
tags: ["Generative AI"]
title: "Diffusion Model"
---

### Advantages:

- **High-Quality Outputs:** Diffusion models are known for producing high-quality, realistic samples.
- **Flexibility:** They can be applied to various types of data, including images, audio, and even text.
- **Theoretical Foundation:** The process is grounded in well-understood principles of probability and statistics.

**Challenges**: Computational Cost、Training Complexity.



# 原理

+ **Reverse Process**（多次Denoise）

  **reconstructing meaningful data from noise** by iteratively removing noise that was added during the **forward process**.

+ **Forward Process**:

  Gradually adds noise to data over multiple steps until the data becomes pure noise.

![1](/images/DM/1.png)

+ Denoise输入：图片 + 噪音程度

  ![2](/images/DM/2.png)

### Denoise Model内部

![3](/images/DM/3.png)

**Noise Predicter**: 预测noise长什么样

+ 如何训练Noise Predicter？

  + 人为创造（加杂讯/Forward Process）

    ![4](/images/DM/4.png)

### **Text-to-Image**

Laion拥有5.85B图片，可进行搜索

输入：图片+噪声程度+**文字叙述**

![5](/images/DM/5.png)

### 算法

![6](/images/DM/6.png)





# Stable Diffusion

+ **框架组成**：

  +  **Text Encoder**：输入为文字，输出为向量
  + **Deneration Model**：输入杂讯和向量，输出中间产物（图片压缩结果）
  + **Decoder**：将中间产物还原回原来的图片

  通常，3个Model分开训练，最后组合在一起

![7](/images/DM/7.png)

+ 经典结构：

  1. Stable Diffusion

     ![8](/images/DM/8.png)



























