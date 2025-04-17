---
date: 2025-04-11T04:00:59-07:00
description: ""
featured_image: "/images/nuSences/pia.jpg"
tags: ["RL", "AD", "data processing"]
title: "数据集-NuSences"
---

1. #### 内容

   nuScenes 包含 1000 个场景，大约 1.4M 的相机图像、390k LIDAR 扫描、1.4M 雷达扫描和 40k 关键帧中的 1.4M 对象边界框。

   nuScenes-lidarseg 包含 40000 个点云和 1000 个场景（850 个用于训练和验证的场景，以及 150 个用于测试的场景）中的 14 亿个注释点。

   &nbsp;

2. #### 数据采集

   1. **车辆设置**

      ![1](/images/nuSences/1.png)

      1. 1 个旋转激光雷达 （Velodyne HDL32E）
      2. 5 个远程雷达传感器 （Continental ARS 408-21）
      3. 6 个相机 （Basler acA1600-60gc）
      4. 1个 IMU & GPS （高级导航空间版）

   2. **Sensor(传感器)校准 - 内外参**

      1. LIDAR extrinsics
      2. 相机 extrinsics
      3. RADAR extrinsics
      4. 相机 intrinsic 校准

   3. **Sensor(传感器)同步**

      实现跨模态数据对齐：当顶部 LIDAR 扫描相机 FOV 的中心时，会触发相机的曝光

   <!--more-->

   &nbsp;

3. #### 数据标准

   + nuScenes 数据集中的所有对象都带有一个**语义类别**，以及一个 **3D 边界框**和它们所在的**每一帧的属性**。：[类的定义](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md)

   + nuScenes-lidarseg 使用**语义标签**注释激光雷达点云中的每个点：[类的定义](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md)

     