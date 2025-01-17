---
date: 2024-12-25T11:00:59-04:00
description: ""
featured_image: "/images/trans/taytay.HEIC"
tags: ["deep learning"]
title: "transformer"
---

The **Transformer** is a deep learning architecture introduced in the 2017 paper *"Attention is All You Need"* by Vaswani et al. It revolutionized natural language processing (NLP) and has since become the foundation for many state-of-the-art models, such as BERT, GPT, and T5.

### Seq2seq

Input a sequence, output a sequence The output length is determined by model.

![截屏2024-12-31 15.12.59.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/e22b161f-7704-4d51-8881-70ee7e4c6dcb/%E6%88%AA%E5%B1%8F2024-12-31_15.12.59.png)

![截屏2024-12-31 19.34.56.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/180291f8-9817-42b6-87cb-39bf0734b9ef/%E6%88%AA%E5%B1%8F2024-12-31_19.34.56.png)

![截屏2024-12-31 19.35.51.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/31c00a57-de1e-43a9-b92a-13e3a050be48/%E6%88%AA%E5%B1%8F2024-12-31_19.35.51.png)

## Encoder

![截屏2024-12-31 15.15.42.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/901263ca-4e06-4f07-a672-aab03a6a86d4/cb6d7470-a2be-4ea5-9611-1cf0859d782f.png)

![截屏2024-12-31 15.29.37.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/3f6e614d-e0be-49d2-935e-642ebc3eaa82/%E6%88%AA%E5%B1%8F2024-12-31_15.29.37.png)

- **细节**：

  ![截屏2024-12-31 15.18.01.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/641849c0-19ea-464e-9b96-2e7b93b1d3d5/c8331594-f0cc-4348-967f-b026b8e2b68f.png)

## Decoder

### Autoregressive Decoder

![截屏2024-12-31 19.38.05.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/d3f0993c-63a9-475f-8dd1-44557690a9c6/%E6%88%AA%E5%B1%8F2024-12-31_19.38.05.png)

![截屏2024-12-31 19.38.57.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/5d00d852-7b10-4489-9d0f-edecaa391e13/%E6%88%AA%E5%B1%8F2024-12-31_19.38.57.png)

![截屏2024-12-31 19.39.11.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/56d6bc5d-c2d4-4cc5-9149-5d6d1201d02d/%E6%88%AA%E5%B1%8F2024-12-31_19.39.11.png)

**Error Propogtion**：一步错步步错

- 细节：

  ![截屏2024-12-31 19.39.53.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/96ce4537-c8d6-4822-bd51-1869260f3fb1/%E6%88%AA%E5%B1%8F2024-12-31_19.39.53.png)

### Non-autoregressive(NAT)

![截屏2024-12-31 19.42.34.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/9ea4d1bd-66ed-46d6-8fa8-9a2335c07586/%E6%88%AA%E5%B1%8F2024-12-31_19.42.34.png)

- How to decide the output length for NAT decoder?
  - Another predictor for output length
  - Output a very long sequence, ignore tokens after END
- **Advantage**: parallel, controllable output length
- NAT is usually worse than AT (why? Multi-modality)

## Cross attention

1. 第一步

   ![截屏2024-12-31 19.52.18.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/1a737598-e158-4b82-ba06-5c3c2ae44553/%E6%88%AA%E5%B1%8F2024-12-31_19.52.18.png)

2. 第二步…

   ![截屏2024-12-31 19.53.04.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/2bb2d505-9ef0-4c7d-9f74-c84e33a80e3e/%E6%88%AA%E5%B1%8F2024-12-31_19.53.04.png)

encoder和decoder可以有多种连接方式

![截屏2024-12-31 19.55.35.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/fdc49317-b8fd-481d-8e87-2cf61ff9d3e4/%E6%88%AA%E5%B1%8F2024-12-31_19.55.35.png)

## Training

**Teacher Forcing**: 训练时，使用正确的答案作为decoder输入。

![截屏2024-12-31 19.59.09.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/e5b4e7f2-3305-4ff3-976c-391aa2757792/%E6%88%AA%E5%B1%8F2024-12-31_19.59.09.png)

### 训练技巧

1. **Copy Mechanism**

   ![截屏2024-12-31 20.01.44.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/9bd3c2d1-86c1-4934-bff6-c2e1d3a1115a/%E6%88%AA%E5%B1%8F2024-12-31_20.01.44.png)

   ![截屏2024-12-31 20.02.03.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/bc92bc0c-f2d3-4ed9-aad7-ecad4959c340/af272953-461a-4ea2-b35b-2b05f82bed81.png)

2. **Guided Attention**

   ![截屏2024-12-31 20.05.33.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/dde2d640-2507-4d44-b02f-60ff2c38052e/%E6%88%AA%E5%B1%8F2024-12-31_20.05.33.png)

   **解决方式**：Monotonic Attention；Location-aware attention

3. **Beam Search**

   Assume there are only two tokens (V=2).

   ![截屏2024-12-31 20.08.21.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2da5ebdb-aecd-4b19-910d-af09587de5f1/c74673ff-15a6-4d94-a56d-2d0cfacf5105/%E6%88%AA%E5%B1%8F2024-12-31_20.08.21.png)

### 优势

- **并行化能力强：**

  Transformer 摒弃了 RNN 的顺序处理限制，使用自注意力机制并行处理整个序列，从而显著提高了训练效率。

- **捕获长程依赖关系：**

  自注意力机制允许模型在一次计算中关注整个序列，因此相比于 RNN 和 CNN，Transformer 更擅长处理长距离依赖。

- **更高的表示能力：**

  多头注意力机制能够让模型同时关注不同的语义和特征，使模型具有更丰富的表达能力。

- **灵活性：**

  Transformer 能够适用于各种任务（例如自然语言处理、计算机视觉、时间序列分析），只需要对架构进行微调。

**(5) 易于扩展：**

•	由于 Transformer 不依赖序列处理，其结构可以轻松扩展到更深或更宽的模型，例如 GPT、BERT、ViT 等。

**(6) 研究生态成熟**
