---
date: 2025-04-12T04:00:59-07:00
description: ""
featured_image: "/images/script/pia.jpg"
tags: ["data processing"]
title: "脚本使用"
---

#### 脚本学习

1. python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0 tensorboard --logdir=./runs --bind_all

   #### 第一部分：`python main.py train mini/trainval ...`

   ```bash
   python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
   ```

   **参数解析**：

   - `train`：表示执行训练模式。
   - `mini/trainval`：数据集的子集（如训练集+验证集的迷你版本）。
   - `--dataroot=NUSCENES_ROOT`：指定数据集路径，需替换`NUSCENES_ROOT`为实际路径。
   - `--logdir=./runs`：训练日志和模型检查点会保存在`./runs`目录。
   - `--gpuid=0`：使用第0号GPU（如果有多块GPU）。

   &nbsp;

   **第二部分：`tensorboard --logdir=./runs --bind_all`**

   ```bash
   tensorboard --logdir=./runs --bind_all
   ```

   1. **作用**：

      启动TensorBoard服务，用于可视化训练过程中的指标（如损失曲线、准确率等）。

   2. **参数解析**：

      - `--logdir=./runs`：指定TensorBoard读取的日志目录（与训练脚本的`--logdir`一致）。
      - `--bind_all`：允许通过网络访问TensorBoard（默认地址通常是`localhost:6006`）。

   <!--more-->

&nbsp;

####  `Fire` 库

作用：将Python函数**自动转换为命令行接口**

+ **命令结构**

  ```bash
  python main.py <函数名> [函数参数] [--选项=值]
  ```

  - <函数名>：必须是 `Fire` 字典中的键（如 `train`、`lidar_check`）。
  - [函数参数]：传递给对应函数的位置参数。
  - [--选项=值]：对应函数的关键字参数（如 `--dataroot`、`--logdir`）。

+ **举例**

  ```bash
  python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
  ```

  + **`train`**：选择执行 `src.train.train` 函数。
  + **`mini/trainval`**：作为位置参数传递给 `src.train.train` 的第一个参数。
  + **`--dataroot`、`--logdir`、`--gpuid`**：作为关键字参数传递给 `src.train.train`。