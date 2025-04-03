---
date: 2025-04-02T12:00:59-05:00
description: ""
featured_image: "/images/trpo/pia.jpg"
tags: ["RL"]
title: "TRPO"
---

基于策略的方法的缺点：当策略网络是深度模型时，沿着策略梯度更新参数，很有可能由于步长太长，策略突然显著变差，进而影响训练效果。

&nbsp;

+ **信任区域策略优化**（trust region policy optimization，TRPO）算法的核心思想：

  **信任区域**（trust region）：在这个区域上更新策略时能够得到某种策略性能的安全性保证。

&nbsp;

## 1. 策略目标

![1](/images/trpo/1.png)

![2](/images/trpo/2.png)

