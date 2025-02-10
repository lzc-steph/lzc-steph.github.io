---
date: 2025-02-08T11:00:59-04:00
description: "Transformer 的核心思想是自注意力机制（Self-Attention），它能够捕捉序列中不同位置之间的依赖关系。通过并行处理序列数据，Transformer 显著提高了训练效率。"
featured_image: "/images/trans/lucky.jpg"
tags: ["LLM"]
title: "transformer"
---

## 一、Transformer架构

- ﻿基于编码器-解码器架构来处理序列对

- ﻿跟使用注意力的seq2seq不同，Transformer是纯基于注意力

  - seq2seq

    ![1](/images/trans/1.png)

  - transformer

    ![1](/images/trans/2.jpg)

&nbsp;

### 1. 多头注意力(Muti-head attention)

+ 对同一key，value，query，希望抽取不同的信息*（类似卷积的多通道）*
  + 例如短距离关系和长距离关系

+ 多头注意力使用h个独立的注意力池化
  + 合并各个头（head） 输出得到最终输出

![3](/images/trans/3.png)

1. 通过全连阶层，映射到一个较低的维度
2. 进行多个attention
3. 对每一个attention的输出，进行concat
4. 再通过一个全连接，得到输出的维度

#### 数学原理

![4](/images/trans/4.jpg)

<!--more-->



&nbsp;

### 2. 有掩码的多头注意力(Masked muti-head attention)

- 解码器对序列中一个元素输出时，不应该考虑该元素之后的元素
- ﻿可以通过掩码来实现
  - ﻿也就是计算x，输出时，假装当前序列长度为i
