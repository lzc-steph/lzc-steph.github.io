---
date: 2026-05-11T11:00:59-04:00
description: "该论文提出了GoG，用于在不完整知识图谱（IKGQA）上进行问答任务。GoG将LLM同时作为代理和知识图谱，通过“思考-搜索-生成”框架动态整合外部知识图谱和LLM的内部知识。该方法有效缓解了LLM的知识不足和幻觉问题，为复杂问答任务提供了新思路。"
featured_image: "/images/paper-GoG/tomo.jpg"
tags: ["paper"]
title: "「论文阅读」Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering"
---

论文链接：https://arxiv.org/abs/2404.14741

提出了一种称为 Generate-on-Graph(GoG) 的免训练方法，它可以在探索 KG 时，生成新的事实三元组。

具体来说，在不完全知识图谱(IKGQA) 中，GoG 通过 Thinking-Searching-Generating 框架进行推理，它将 LLM 同时视为 Agent 和 KG。

&nbsp;

<!--more-->

### 1 Introduction

![1](/Users/aijunyang/DearAJ.github.io/static/images/paper-GoG/1.png)

+ #### KG + LLM 结合的方法：

  + **语义解析(SP)方法**：使用 LLMs 将 nlp 问题转换为逻辑查询，然后通过在 KG 上执行这些逻辑查询来获得答案。

    ![2](/Users/aijunyang/DearAJ.github.io/static/images/paper-GoG/2.png)

  + **检索增强(RA)方法**：从 KG 检索与问题相关的信息，作为外部知识以指导 LLMs 生成答案。

    ![3](/Users/aijunyang/DearAJ.github.io/static/images/paper-GoG/3.png)

  + **Generate-on-Graph**

    ![4](/Users/aijunyang/DearAJ.github.io/static/images/paper-GoG/4.png)





1. **研究背景**：
   - **问题**：这篇文章旨在解决大型语言模型（LLMs）在处理复杂推理任务时存在的知识不足和幻觉问题。
   - **难点**：该问题的研究难点在于现有的方法通常在完整的知识图谱上进行评估，而现实世界中的知识图谱往往是部分不完整的，这使得LLMs难以有效整合内部和外部的知识源。
   - **相关工作**：现有工作主要集中在将LLMs与知识图谱结合，但这些方法大多在完整的知识图谱上进行评估，未能充分模拟真实世界的场景。
2. **研究方法**：
   - 提出了一个新的基准任务，称为不完整知识图谱问答（IKGQA），用于模拟现实世界中知识图谱不完整的情况，并构建了相应的IKGQA数据集。
   - 提出了一个名为Generate-on-Graph（GoG）的训练方法，该方法通过Thinking-Searching-Generating框架，使LLMs能够生成新的三元组以回答不完整知识图谱中的问题。
   - 具体来说，GoG包括三个主要步骤：Thinking（思考）、Searching（搜索）和Generating（生成）。在Thinking阶段，LLMs分解问题并决定是否需要进一步搜索或生成相关三元组；在Searching阶段，LLMs使用预定义的工具（如SPARQL查询）探索知识图谱并过滤无关三元组；在Generating阶段，LLMs利用其内部知识和推理能力生成所需的新三元组并进行验证。
3. **实验设计**：
   - 在两个广泛使用的知识图谱问答数据集（WebQuestionSP和Complex WebQuestion）上进行了实验，通过随机删除关键三元组来模拟不完整的知识图谱。
   - 使用四个LLMs（GPT-3.5、GPT-4、Qwen-1.5-72B-Chat和LLaMA3-70B-Instruct）作为基础模型，并设置了不同的实验参数，如最大生成长度为256，温度参数为0.7，每个数据集使用3个提示进行实验。
4. **结果与分析**：
   - 实验结果表明，GoG在两个数据集上的表现优于所有基线方法，特别是在不完整的知识图谱设置下，GoG的平均Hits@1得分提高了5.0%。
   - 具体而言，在WebQuestionSP数据集上，GoG在使用GPT-4作为基础模型时的Hits@1得分为84.4%，显著高于其他方法。在Complex WebQuestion数据集上，GoG在使用GPT-4时的Hits@1得分为80.3%，同样表现出色。
   - 进一步的分析表明，GoG在处理复合值类型（CVTs）时表现尤为出色，能够有效利用邻居信息预测尾实体。
5. **总体结论**：
   - 提出了利用LLMs在不完整的知识图谱中进行问答的方法，并构建了相应的基准数据集。
   - 实验结果表明，GoG能够有效整合LLMs的内部和外部知识，显著提升在不完整知识图谱中的问答性能。
   - 未来的研究可以进一步探索如何更好地利用不同LLMs的优势，以应对更复杂的问答任务。
