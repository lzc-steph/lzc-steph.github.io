---
date: 2025-01-04T11:00:59-04:00
description: ""
featured_image: "/images/GA/taytay.jpeg"
tags: ["Generative AI"]
title: "Generative AI"
---

# ChatGPT

![截屏2025-01-05 23.37.32.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/ba53c8d8-3a8d-4bb9-a20f-5cd7d295a29f/%E6%88%AA%E5%B1%8F2025-01-05_23.37.32.png)

![截屏2025-01-04 00.25.09.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/740b097c-4c79-42af-bc9b-5817abdc3521/%E6%88%AA%E5%B1%8F2025-01-04_00.25.09.png)

1. ChatGPT 真正做的事：文字接龙

   **Autoregressive Generation**：逐个生成

   ![截屏2025-01-04 00.18.34.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/a140588c-153b-4bf3-812b-794d09b256c2/%E6%88%AA%E5%B1%8F2025-01-04_00.18.34.png)

   ![截屏2025-01-04 00.27.44.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/10249f01-191e-4450-9387-201ee36c281d/247f57f5-a4f8-4859-8ce8-9664487924cd.png)

2. **token**

   文字接龙时可以选择的符号

   ![截屏2025-01-04 00.29.01.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/e0bc078c-68ba-4504-b3dc-2bee0739d6fa/%E6%88%AA%E5%B1%8F2025-01-04_00.29.01.png)

3. 每次回答都随机（掷骰子）

   ![截屏2025-01-04 11.07.26.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/b30b483b-47dd-4cfd-9a2d-bcd030b1b169/%E6%88%AA%E5%B1%8F2025-01-04_11.07.26.png)

4. 进化关键：**自督导式学习(预训练) ➡️ 督导式学习(微调) ➡️ 强化学习**

   ![截屏2025-01-04 23.52.07.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/b2ca8f91-7703-4339-ae03-089a67c3519f/%E6%88%AA%E5%B1%8F2025-01-04_23.52.07.png)

   有预训练，督导式学习不用大量资料。

   强化学习提供回馈。督导式学习提供完整资料，强化学习给反馈（如两次答案，有没有比上次更好）

   【注】：模型要有一定程度的能力才适合进入强化学习。

   **Alignment(对齐)**：督导式学习 + 强化学习

5. **强化学习**

   1. 学习reward model

      reward model：模仿人类的偏好

   2. 用reward model进行学习

      模型只需要向reward model学习

6. **GPT-4: 可以看图+引导**

7. **如何激发gpt的能力？**

   1. 把需求说清楚
   2. 提供咨询
   3. 提供范例
   4. 鼓励gpt想一想
   5. 训练generator
   6. 上传资料
   7. 使用其它工具
   8. 大任务拆解成小任务
   9. gpt会反省

8. 可以做什么？

   1. prompt engineering
   2. 训练自己的模型（如调整LLaMA参数），困难

# 大型语言模型训练过程

![截屏2025-01-06 00.37.19.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/7e6521a3-d2cf-42bc-8550-aacc3027b83f/%E6%88%AA%E5%B1%8F2025-01-06_00.37.19.png)

1. 自我学习阶段

   - 调整超参数

   - 训练成果，但测试失败：找到多样数据

   - 找到合适的初始参数：随机/ 先验知识

     先验知识：爬网络资料+资料清理(训练资料品质分类器/除重)

2. 人类指导阶段
