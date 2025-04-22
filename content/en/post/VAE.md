---
date: 2024-12-29T11:00:59-04:00
description: "变分自编码器（VAE） 是一种生成模型，通过学习数据的潜在表示来生成新样本。它结合了深度学习和概率建模，通过编码器将输入数据映射到潜在空间，再通过解码器从潜在空间重建数据。VAE的训练目标包括重建误差和潜在空间的正则化（KL散度），确保潜在空间连续且结构化。"
featured_image: "/images/vae/taytay.HEIC"
tags: ["Generative AI"]
title: "VAE"
---



## 普通自动编码器

目标：将高维度数据压缩成较小的表示

![截屏2024-12-30 19.27.19.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/ad01491f-ff96-44a7-9a2f-606c66461df8/%E6%88%AA%E5%B1%8F2024-12-30_19.27.19.png)

![截屏2024-12-30 19.28.31.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/40f99add-baee-40ad-ab9b-45dc5f312af9/%E6%88%AA%E5%B1%8F2024-12-30_19.28.31.png)

![截屏2024-12-30 19.29.08.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/99bd3fd9-fabb-421d-bcdb-0e6d0c5be310/%E6%88%AA%E5%B1%8F2024-12-30_19.29.08.png)

应用：

![截屏2024-12-30 19.29.47.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/1b594f9d-d71f-4af3-b7c9-285e1dfffec6/%E6%88%AA%E5%B1%8F2024-12-30_19.29.47.png)

## 去噪自动编码器

![截屏2024-12-30 19.35.42.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/a8e8568e-18b2-4281-a7af-05a627e6f5f0/%E6%88%AA%E5%B1%8F2024-12-30_19.35.42.png)

![截屏2024-12-30 19.36.58.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/ce2c3930-ae83-40a9-b9d3-aa3f4f0246bd/%E6%88%AA%E5%B1%8F2024-12-30_19.36.58.png)

![截屏2024-12-30 19.37.09.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/fdd726fa-7eb9-4b24-af63-f917ee593b1c/%E6%88%AA%E5%B1%8F2024-12-30_19.37.09.png)

## 变分自动编码器

![截屏2024-12-30 19.39.12.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/f4509f09-0920-4a7e-b5ba-74cbec294241/%E6%88%AA%E5%B1%8F2024-12-30_19.39.12.png)

- 损失函数

  ![截屏2024-12-30 19.40.36.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/add2cf71-1094-4639-b2d5-699f551beb48/%E6%88%AA%E5%B1%8F2024-12-30_19.40.36.png)

### Reparameterization Trick(重参数化)

如何实现反向传播？

![截屏2024-12-30 19.42.30.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/db9e763f-81d3-460f-81dc-b81242f41cdb/%E6%88%AA%E5%B1%8F2024-12-30_19.42.30.png)

![截屏2024-12-30 19.44.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/e3fb22c3-4a30-4511-9457-4dc08d40b210/%E6%88%AA%E5%B1%8F2024-12-30_19.44.43.png)

- 核心代码

  ![截屏2024-12-30 19.45.33.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/9c033e5a-8e28-4e82-ab6e-80d29dd08a20/%E6%88%AA%E5%B1%8F2024-12-30_19.45.33.png)

  ![截屏2024-12-30 19.45.42.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/64564c69-b355-4559-95b2-2186d3cf9e08/%E6%88%AA%E5%B1%8F2024-12-30_19.45.42.png)

  ![截屏2024-12-30 19.46.02.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/df9a2aa4-f700-4975-a5ca-f315eab2bcab/%E6%88%AA%E5%B1%8F2024-12-30_19.46.02.png)

### **解耦变分自编码器**

目的：确保潜在分布中的不同神经元互不相干，都在尝试学习输入数据中的不同内容。

- 解决方式：增加超参数$\beta$，衡量损失函数中的KL散度。

  ![截屏2024-12-30 22.11.49.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/9501fd11-c948-42ff-ba18-807ca26a11eb/%E6%88%AA%E5%B1%8F2024-12-30_22.11.49.png)

![截屏2024-12-30 22.18.10.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/d64275f8-60ea-4cd6-862a-d8d7821c8b3d/%E6%88%AA%E5%B1%8F2024-12-30_22.18.10.png)

$\beta$过小：过拟合

$\beta$过大：失去输入中的大量细节

**在强化学习中的作用**：使Agent可以在压缩后的输入T空间上运行。
