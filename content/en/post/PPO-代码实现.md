---
date: 2025-03-29T11:00:59-04:00
description: ""
featured_image: "/images/PPOcode/meovv.jpg"
tags: ["RL"]
title: "PPO 代码实现"
---

## 1. 论文详读

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

&nbsp;

## 2. Policy Gradient

### 1. 完整过程

1. 随机初始化 actor 参数 theta

2. 玩 n 次游戏，收集 n 个 trajectory（state、action），算出 reward

3. 用得到的 data 去更新参数 theta

   ![1](/images/PPOcode/1.png)

   ![2](/images/PPOcode/2.png)

   如果 R(τⁿ) 为正，梯度更新会提升该轨迹中所有动作的概率；若为负，则降低概率。

   <!--more-->

4. 得到新的 actor 后，再去玩新的 n 次游戏

   ![3](/images/PPOcode/3.png)

5. 循环往复上述过程

&nbsp;

### 2. 如何更新参数

+ 以分类问题为例：

  



