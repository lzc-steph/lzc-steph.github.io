---
date: 2025-01-20T11:00:59-04:00
description: ""
featured_image: "/images/Quanti/taytay.HEIC"
tags: ["deep learning"]
title: "Quantization"
---

# 概述

+ **量化**：Store the parameters of the model in lower precision

- **Knowledge distillation**: Train a smaller model (student) using the original model (instructor).
- **Pruning**: Remove connections (weights) from the model
- **Application**: Perform linear quantization on any model using Quanto in two lines

## 内容

1. **Linear Quantization theory** in detail and how to code it (*per channel, per tensor, per group quantization*)

2. Build your own 8-bit linear quantizer and apply it on real models.

   量化与模型无关（通用）

3. Learn about **quantization challenges** (weight packing, lims quantization)





# 线性量化

## 量化和去量化

量化指将一个大集合映射到一个较小值集合的过程。

![1](/images/Quanti/1.png)

### 可以量化的内容

- **The weights**: Neural network parameters
- ﻿﻿**The activations**: Values that propagate through the layers of the neural network

If you quantize the NN after it has been trained, you are doing **post training quantization (PTQ)**

### 优点

1. 更小的模型

2. 速度更快

   + 内存带宽

   - ﻿﻿Faster operations

     - ﻿﻿GEMM: General Matrix Multiply (matrix to matrix multiplication)

     - ﻿﻿GEMV: General Matrix Vector
        Multiplication (matrix to vector multiplication)



### 线性量化理论

+ 参数

  Scale：s（e.g. in FP32）

  Zero point：z（e.g. INT8）

![2](/images/Quanti/2.png)





## 非对称性变体



## 缩放因子



## 零点







# 量化的挑战

1. Quantization error
2. Retraining (Quantization Aware Training)
3. Limited Hardware support
4. Calibration dataset needed
5. ﻿﻿packing/unpacking



