---
date: 2025-01-04T11:00:59-04:00
description: ""
featured_image: "/images/GA/taytay.jpeg"
tags: ["Generative AI"]
title: "Generative AI"
---

# ChatGPT

![1](/images/GA/1.png)

+ G：generative

+ P：pre-trained

+ T：transformer

&nbsp;

1. ChatGPT 真正做的事：文字接龙

   **Autoregressive Generation**：逐个生成

   ![3](/images/GA/3.png)

   ![4](/images/GA/4.png)

2. **token**

   文字接龙时可以选择的符号

   ![5](/images/GA/5.png)

3. 每次回答都随机（掷骰子）

   ![6](/images/GA/6.png)

   <!--more-->

4. 进化关键：**自督导式学习(预训练) ➡️ 督导式学习(微调) ➡️ 强化学习**

   ![7](/images/GA/7.png)

   有预训练，督导式学习不用大量资料。

   强化学习提供回馈。督导式学习提供完整资料，强化学习给反馈（如两次答案，有没有比上次更好）

   *【注】：模型要有一定程度的能力才适合进入强化学习。*

   **Alignment(对齐)**：督导式学习 + 强化学习

5. **强化学习**

   1. 学习reward model

      reward model：模仿人类的偏好

   2. 用reward model进行学习

      模型只需要向reward model学习

6. **GPT-4: 可以看图+引导**

7. **如何激发gpt的能力？**

   把需求说清楚；提供咨询；提供范例；鼓励gpt想一想；训练generator；上传资料；使用其它工具；大任务拆解成小任务；gpt会反省...
   
8. 可以做什么？

   1. prompt engineering
   2. 训练自己的模型（如调整LLaMA参数），困难

 &nbsp;

&nbsp;

# 大型语言模型训练过程

![8](/images/GA/8.png)

1. 自我学习阶段

   - 调整超参数

   - 训练成果，但测试失败：找到多样数据

   - 找到合适的初始参数：随机/ 先验知识

     先验知识：爬网络资料+资料清理(训练资料品质分类器/除重)

2. 人类指导阶段
