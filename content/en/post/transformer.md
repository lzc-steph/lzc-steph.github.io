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

<!--more-->

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

&nbsp;

### 2. 有掩码的多头注意力(Masked muti-head attention)

- 解码器对序列中一个元素输出时，不应该考虑该元素之后的元素
- ﻿可以通过掩码来实现
  - ﻿也就是计算$x_i$输出时，假装当前序列长度为i；掩掉了后面的key value。

&nbsp;

### 3. 基于位置的前馈网络（全连接层）

- 将输入形状由（b, n, d）变换成（bn, d）
- ﻿﻿作用两个全连接层
- ﻿﻿输出形状由（bn, d）变化回（b,n, d）
- ﻿等价于两层核窗口为1的一维卷积层

&nbsp;

### 4. add&norm

+ add是一个resnet block

- 不能用批量归一化
  - 对每个特征/通道里元素进行归一化，不稳定，不适合序列长度会变的NLP应用
- ﻿norm: 层归一化对**每个样本里的元素**进行归一化

![5](/images/trans/5.png)

&nbsp;

### 5. 信息传递

- ﻿编码器中的输出了$y_1$, ..., $y_n$
- ﻿﻿将其作为解码中第i个Transformer块中多头注意力的key和value
  - ﻿它的query来自目标序列
- ﻿意味着编码器和解码器中块的个数和输出维度都是一样

![6](/images/trans/6.png)

&nbsp;

### 6. 预测

- 预测第t+1个输出时，解码器中输入前t个预测值
- ﻿在自注意力中，前t个预测值作为key和value，第t个预测值还作为query

![7](/images/trans/7.jpg)

&nbsp;

&nbsp;



## 二、Transformer总结

- Transformer是一个纯使用注意力的编码-解码器
- ﻿编码器和解码器都有n个 transformer块
- ﻿﻿每个块里使用多头自注意力，基于位置的前馈网络，和层归一化





