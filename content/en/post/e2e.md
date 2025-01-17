---
date: 2024-12-26T11:00:59-04:00
description: "端到端指的是从输入到输出的整个过程由一个统一的模型或系统完成，而不需要中间的手动处理或多个模块的拼接。这种方法强调简化流程，减少人为干预，直接通过数据驱动的方式完成任务。"
featured_image: "/images/e2e/taytay.HEIC"
tags: ["RL"]
title: "End2End"
---

对于由多个阶段组成的学习系统，端到端学习捕获所有阶段，将其替代为单个神经网络。

- 优点：
  - Let the data speak
  - Less hand-designing of components needed
- 缺点：
  - May need large amount of data
  - Excludes potentially useful hand-designed components

**关键**：是否有足够的数据

![1](/images/e2e/1.png)
