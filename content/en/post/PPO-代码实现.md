---
date: 2025-03-29T11:00:59-04:00
description: ""
featured_image: "/images/PPOcode/pia.jpg"
tags: ["RL"]
title: "PPO 代码实现"
---

## 1. 论文详读

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

&nbsp;

## 2. Policy Gradient

1. 随机初始化 actor 参数 theta

2. 玩 n 次游戏，收集 n 个 trajectory（state、action），算出 reward

3. 用得到的 data 去更新参数 theta

   ![1](/Users/aijunyang/DearAJ.github.io/static/images/PPOcode/1.png)

   <!--more-->