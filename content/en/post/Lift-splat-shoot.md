---
date: 2025-03-27T11:00:59-04:00
description: ""
featured_image: "/images/lss/deva.jpg"
tags: ["RL"]
title: "Lift-splat-shoot"
---

 [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)

### 1. 关键：Lift

![1](/images/lss/1.png)

1. **特征提取&深度估计**

   多视角相机输入后，进行特征提取与深度估计

2. **外积（Outer product）**

3. **Grid Sampling**

   

<!--more-->

### 2. LSS完整流程

1. 生成视锥，并根据相机内外参将视锥中的点投影到 ego 坐标系

   + 生成视锥

     其位置是基于图像坐标系的，同时锥点是图像特征上每个单元格映射回原始图像的位置

   + 锥点由图像坐标系向 ego 坐标系进行坐标转化

     主要涉及到相机的内外参数

2. 对环视图像完成特征的提取，并构建图像特征点云

   + 利用 [Efficientnet-B0](https://zhida.zhihu.com/search?content_id=219055188&content_type=Article&match_order=1&q=Efficientnet-B0&zhida_source=entity) 主干网络对环视图像进行特征提取。

     输入的环视图像 (bs, N, 3, H, W)，在进行特征提取之前，会将前两个维度进行合并，一起提取特征，对应维度变换为 (bs, N, 3, H, W) -> (bs * N, 3, H, W)

   + 对其中的后两层特征进行融合，丰富特征的语义信息，融合后的特征尺寸大小为 (bs * N, 512, H / 16, W / 16）

   + 估计深度方向的概率分布，并输出特征图每个位置的语义特征 (用64维的特征表示）

     整个过程用1x1卷积层实现

   + 对上一步骤估计出来的离散深度，利用softmax()函数计算深度方向的概率密度

   + 利用得到的深度方向的概率密度和语义特征，通过外积运算构建图像特征点云

   ![2](/images/lss/2.png)

3. 利用变换后的 ego 坐标系的点与图像特征点云利用[Voxel Pooling](https://zhida.zhihu.com/search?content_id=219055188&content_type=Article&match_order=1&q=Voxel+Pooling&zhida_source=entity)构建BEV特征

4. 对生成的 BEV 特征利用 BEV Encoder 做进一步的特征融合

5. 利用特征融合后的 BEV 特征完成语义分割任务

&nbsp;

### 2. 论文阅读



本文提出了一种架构，旨在从任意摄像机装备推断鸟瞰图表示。

1. #### Introduction

   目标：从任意数量的摄像机中直接提取给定图像数据的场景的鸟瞰图表示。

   + 单视图扩展成多视图的对称性：

     1. **平移等方差**： 如果图像中的像素坐标全部偏移，则输出将偏移相同的量。
     2. **Permutation invariance**： 最终输出不取决于 n 相机的特定顺序。
     3. **自我框架等距等方差**： 无论捕获图像的相机相对于自我汽车的位置如何，都会在给定图像中检测到相同的对象。

     缺点：反向传播不能用于使用来自下游规划器的反馈来自动改进感知系统。

   传统在与输入图像相同的坐标系中进行预测，我们的模型遵循上述对称性，直接在给定的鸟瞰图框架中进行预测，以便从多视图图像进行端到端规划。

   <!--more-->

   ![1](/images/lss/1.png)

2. #### Related Work

   1. **单目物体检测**
      1. 在图像平面中应用一个成熟的 2D 对象检测器，然后训练第二个网络将 2D 框回归到 3D 框。
      2. 伪激光雷达：训练一个网络进行单目深度预测，另一个网络分别进行鸟瞰检测。
      3. 使用 3 维对象基元，
   2. **BEV 框架中的推理**：使用 extrinsics 和 intrinsics 直接在鸟瞰框架中执行推理
      1. MonoLayout：从单个图像执行鸟瞰图推理，并使用对抗性损失来鼓励模型对合理的隐藏对象进行修复。
      2. Pyramid Occupancy Networks：提出了一种 transformer 架构，将图像表示转换为鸟瞰图表示。
      3. FISHING Net：提出了一种多视图架构，既可以分割当前时间步中的对象，也可以执行未来预测。

3. #### Method

   对每个图像，都有一个 extrinsic matrix 和 intrinic matrix，它们共同定义每个相机从参考坐标 (x,y,z) 到局部像素坐标 (h,w,d) 的映射。

   + 核心流程：

     - **Lift**：将2D图像特征显式提升到3D空间（通过深度估计生成视锥特征）。

     - **Splat**：将3D特征“展开”到BEV空间，构建鸟瞰图特征。

     - **Shoot**：基于BEV特征进行运动规划或轨迹预测。

   1. **Lift：潜在深度分布**

      目的：将每个图像从本地 2 维坐标系 “提升” 到在所有摄像机之间共享的 3 维帧。

      ![2](/images/lss/2.png)

   2. **Splat：支柱池**

      + lift输出：大点云

      + 将每个点分配给最近的 pillar，并执行总和池化，以创建一个可由标准 CNN 处理以进行鸟瞰推理的 C×H×W 张量。（pillars 是具有无限高度的体素）

      ![3](/images/lss/3.png)

      + 加速：不是填充每个 pillar 然后执行 sum pooling，而是通过使用 packing 和利用 “cumsum 技巧” 进行 sum pooling 来避免填充。

   3. **Shoot: 运动规划**

      + 定义 planning：预测自我车辆在模板轨迹上的 K 分布。

        ![4](/images/lss/4.png)

      + 在测试时，实现使用 inferred cost map 的 planning：

        通过“射击”不同的轨迹，对它们的成本进行评分，然后根据最低成本轨迹采取行动。

      + 在实践中，我们通过在大量 template trajectories 上运行 K-Means 来确定模板轨迹集。

      ![5](/images/lss/5.png)

4. #### Implementation

   模型有两个大型网络主干，由 lift-splat 层连接起来。

   1. 其中一个主干 对每个图像单独进行操作，以便对每个图像生成的点云进行特征化。

      *利用了在 Imagenet 上预训练的 EfficientNet-B0 中的层。*

   2. 另一个主干 在点云被展开到参考系中的pillars后，对点云进行操作。

      *使用类似于 PointPillars 的 ResNet 块组合。*

   + 技巧：
     + 选择了跨 pillar 的 sum pooling，而不是 max pooling ：免于因填充而导致的过多内存使用。
     + Frustum Pooling：将 n 图像产生的视锥转换为固定维度 C×H×W 的张量，而与相机 n 的数量无关。



&nbsp;

### 3. 算法实现过程梳理

1. **相关参数设置**

   在LSS源码中，其感知范围，BEV单元格大小，BEV下的网格尺寸如下：

   - 感知范围
     x轴方向的感知范围 -50m ~ 50m；y轴方向的感知范围 -50m ~ 50m；z轴方向的感知范围 -10m ~ 10m；
   - BEV单元格大小
     x轴方向的单位长度 0.5m；y轴方向的单位长度 0.5m；z轴方向的单位长度 20m；
   - BEV的网格尺寸
     200 x 200 x 1；
   - 深度估计范围
     由于LSS需要显式估计像素的离散深度，论文给出的范围是 4m ~ 45m，间隔为1m，也就是算法会估计41个离散深度；

   &nbsp;

2. **模型相关参数**

   - imgs：输入的环视相机图片，imgs = (bs, N, 3, H, W)，N代表环视相机个数；
   - rots：由相机坐标系->车身坐标系的旋转矩阵，rots = (bs, N, 3, 3)；
   - trans：由相机坐标系->车身坐标系的平移矩阵，trans=(bs, N, 3)；
   - intrinsic：相机内参，intrinsic = (bs, N, 3, 3)；
   - post_rots：由图像增强引起的旋转矩阵，post_rots = (bs, N, 3, 3)；
   - post_trans：由图像增强引起的平移矩阵，post_trans = (bs, N, 3)；
   - binimgs：由于LSS做的是语义分割任务，所以会将真值目标投影到BEV坐标系，将预测结果与真值计算损失；具体而言，在binimgs中对应物体的bbox内的位置为1，其他位置为0；

   &nbsp;

3. **算法前向过程**

   1）生成视锥，并根据相机内外参将视锥中的点投影到ego坐标系；

   2）对环视图像完成特征的提取，并构建图像特征点云；

   3）利用变换后的ego坐标系的点与图像特征点云利用Voxel Pooling 构建BEV特征；

   4）对生成的BEV特征利用BEV Encoder做进一步的特征融合；

   5）利用特征融合后的BEV特征完成语义分割任务；

   &nbsp;

   &nbsp;

   1. **生成视锥，并完成视锥锥点由图像坐标系->ego坐标系的空间位置转换**

      **a）生成视锥**

      需要注意的是，生成的锥点，其位置是基于图像坐标系的，同时锥点是图像特征上每个单元格映射回原始图像的位置。生成方式如下：

      ```python3
      def create_frustum():
          # 原始图片大小  ogfH:128  ogfW:352
          ogfH, ogfW = self.data_aug_conf['final_dim']
          
          # 下采样16倍后图像大小  fH: 8  fW: 22
          fH, fW = ogfH // self.downsample, ogfW // self.downsample 
           
          # self.grid_conf['dbound'] = [4, 45, 1]
          # 在深度方向上划分网格 ds: DxfHxfW (41x8x22)
          ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
          
          D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
          """
          1. torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
          tensor([0.0000, 16.7143, 33.4286, 50.1429, 66.8571, 83.5714, 100.2857,
                  117.0000, 133.7143, 150.4286, 167.1429, 183.8571, 200.5714, 217.2857,
                  234.0000, 250.7143, 267.4286, 284.1429, 300.8571, 317.5714, 334.2857,
                  351.0000])
                  
          2. torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
          tensor([0.0000, 18.1429, 36.2857, 54.4286, 72.5714, 90.7143, 108.8571,
                  127.0000])
          """
          
          # 在0到351上划分22个格子 xs: DxfHxfW(41x8x22)
          xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  
          
          # 在0到127上划分8个格子 ys: DxfHxfW(41x8x22)
          ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  
          
          # D x H x W x 3
          # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
          frustum = torch.stack((xs, ys, ds), -1)  
          return nn.Parameter(frustum, requires_grad=False)
      ```

      **b）锥点由图像坐标系向ego坐标系进行坐标转化**

      这一过程主要涉及到相机的内外参数，对应代码中的函数为get_geometry()；

      ```python
      def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
          B, N, _ = trans.shape  # B: batch size N：环视相机个数
      
          # undo post-transformation
          # B x N x D x H x W x 3
          # 抵消数据增强及预处理对像素的变化
          points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
          points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
      
          # 图像坐标系 -> 归一化相机坐标系 -> 相机坐标系 -> 车身坐标系
          # 但是自认为由于转换过程是线性的，所以反归一化是在图像坐标系完成的，然后再利用
          # 求完逆的内参投影回相机坐标系
          points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                              points[:, :, :, :, :, 2:3]
                              ), 5)  # 反归一化
                              
          combine = rots.matmul(torch.inverse(intrins))
          points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
          points += trans.view(B, N, 1, 1, 1, 3)
          
          # (bs, N, depth, H, W, 3)：其物理含义
          # 每个batch中的每个环视相机图像特征点，其在不同深度下位置对应
          # 在ego坐标系下的坐标
          return points
      ```

   &nbsp;

   2. **对环视图像进行特征提取，并构建图像特征点云**

      **a）利用 Efficientnet-B0 主干网络对环视图像进行特征提取**
      输入的环视图像 (bs, N, 3, H, W)，在进行特征提取之前，会将前两个维度进行合并，一起提取特征，对应维度变换为 (bs, N, 3, H, W) -> (bs * N, 3, H, W)；其输出的多尺度特征尺寸大小如下：

      ```python3
      level0 = (bs * N, 16, H / 2, W / 2)
      level1 = (bs * N, 24, H / 4, W / 4)
      level2 = (bs * N, 40, H / 8, W / 8)
      level3 = (bs * N, 112, H / 16, W / 16)
      level4 =  (bs * N, 320, H / 32, W / 32)
      ```

      **b）对其中的后两层特征进行融合**

      丰富特征的语义信息，融合后的特征尺寸大小为 (bs * N, 512, H / 16, W / 16）

      ```python3
      Step1: 对最后一层特征升采样到倒数第二层大小；
      level4 -> Up -> level4' = (bs * N, 320, H / 16, W / 16)
      
      Step2：对主干网络输出的后两层特征进行concat；
      cat(level4', level3) -> output = (bs * N, 432, H / 16, W / 16)
      
      Step3：对concat后的特征，利用ConvLayer卷积层做进一步特征拟合；
             
      ConvLayer(output) -> output' = (bs * N, 512, H / 16, W / 16)
      
      其中ConvLayer层构造如下：
      """Sequential(
        (0): Conv2d(432, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )"""
      ```

      **c）估计深度方向的概率分布并输出特征图每个位置的语义特征** (用64维的特征表示），整个过程用1x1卷积层实现

      ```python3
      c)步骤整体pipeline
      output' -> Conv1x1 -> x = (bs * N, 105, H / 16, W / 16)
      
      b)步骤输出的特征：
      output = Tensor[(bs * N, 512, H / 16, W / 16)]
      
      c)步骤使用的1x1卷积层：
      Conv1x1 = Conv2d(512, 105, kernel_size=(1, 1), stride=(1, 1))
      
      c)步骤输出的特征以及对应的物理含义：
      x = Tensor[(bs * N, 105, H / 16, W / 16)] 
      第二维的105个通道分成两部分；第一部分：前41个维度代表不同深度上41个离散深度；
                               第二部分：后64个维度代表特征图上的不同位置对应的语义特征；
      ```

      **d）对c)步骤估计出来的离散深度利用softmax()函数计算深度方向的概率密度**

      **e）利用得到的深度方向的概率密度和语义特征通过外积运算构建图像特征点云**

      代码实现：

      ```python3
      # d)步骤得到的深度方向的概率密度
      depth = (bs * N, 41, H / 16, W / 16) -> unsqueeze -> (bs * N, 1, 41, H / 16, W / 16)
      
      # c)步骤得到的特征，选择后64维是预测出来的语义特征
      x[:, self.D:(self.D + self.C)] = (bs * N, 64, H / 16, W / 16) -> unsqueeze(2) -> (bs * N, 64, 1 , H / 16, W / 16)
      
      # 概率密度和语义特征做外积，构建图像特征点云
      new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # (bs * N, 64, 41, H / 16, W / 16)
      ```

      论文中表示构建图像特征点云的实现过程插图：

      ![2](/images/lss/2.png)

   &nbsp;

   3. 利用ego坐标系下的坐标点与图像特征点云，利用Voxel Pooling构建BEV特征

      **a）Voxel Pooling前的准备工作**

      ```python
      def voxel_pooling(self, geom_feats, x):
          # geom_feats；(B x N x D x H x W x 3)：在ego坐标系下的坐标点；
          # x；(B x N x D x fH x fW x C)：图像点云特征
      
          B, N, D, H, W, C = x.shape
          Nprime = B*N*D*H*W 
      
          # 将特征点云展平，一共有 B*N*D*H*W 个点
          x = x.reshape(Nprime, C) 
      
          # flatten indices
          geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long() # ego下的空间坐标转换到体素坐标（计算栅格坐标并取整）
          geom_feats = geom_feats.view(Nprime, 3)  # 将体素坐标同样展平，geom_feats: (B*N*D*H*W, 3)
          batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                                   device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
          geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: (B*N*D*H*W, 4)
      
          # filter out points that are outside box
          # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
          kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
              & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
              & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
          x = x[kept]
          geom_feats = geom_feats[kept]
      
          # get tensors from the same voxel next to each other
          ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
               + geom_feats[:, 1] * (self.nx[2] * B)\
               + geom_feats[:, 2] * B\
               + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
          sorts = ranks.argsort()
          x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
         
          # cumsum trick
          if not self.use_quickcumsum:
              x, geom_feats = cumsum_trick(x, geom_feats, ranks)
          else:
              x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
      
          # griddify (B x C x Z x X x Y)
          final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: bs x 64 x 1 x 200 x 200
          final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中
      
          # collapse Z
          final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维
      
          return final  # final: bs x 64 x 200 x 200
      ```

      **b）采用cumsum_trick完成Voxel Pooling运算**

      + 该技巧是基于本文方法用图像产生的点云形状是固定的，因此每个点可以预先分配一个区间（即BEV网格）索引，用于指示其属于哪一个区间。按照索引排序后，按下列方法操作：

        ![6](/images/lss/6.png)

        需要注意的，图中的`区间索引`代表下面代码中的ranks，`点的特征`代表的是x；

      + 

      ```python
      class QuickCumsum(torch.autograd.Function):
          @staticmethod
          def forward(ctx, x, geom_feats, ranks):
              x = x.cumsum(0) # 求前缀和
              kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)  
              kept[:-1] = (ranks[1:] != ranks[:-1])  # 筛选出ranks中前后rank值不相等的位置
      
              x, geom_feats = x[kept], geom_feats[kept]  # rank值相等的点只留下最后一个，即一个batch中的一个格子里只留最后一个点
              x = torch.cat((x[:1], x[1:] - x[:-1]))  # x后一个减前一个，还原到cumsum之前的x，此时的一个点是之前与其rank相等的点的feature的和，相当于把同一个格子的点特征进行了sum
      
              # save kept for backward
              ctx.save_for_backward(kept)
      
              # no gradient for geom_feats
              ctx.mark_non_differentiable(geom_feats)
      
              return x, geom_feats
      ```

   &nbsp;

   4. (+5) **对生成的BEV特征利用BEV Encoder做进一步的特征融合** + **语义分割结果预测**

      **a）对BEV特征先利用 ResNet-18 进行多尺度特征提取**，输出的多尺度特征尺寸如下：

      ```python3
      level0：(bs, 64, 100, 100)
      level1: (bs, 128, 50, 50)
      level2: (bs, 256, 25, 25)
      ```

      **b）对输出的多尺度特征进行特征融合 + 对融合后的特征实现BEV网格上的语义分割**

      ```python3
      Step1: level2 -> Up (4x) -> level2' = (bs, 256, 100, 100)
      
      Step2: concat(level2', level0) -> output = (bs, 320, 100, 100)
      
      Step3: ConvLayer(output) -> output' = (bs, 256, 100, 100)
      
      '''ConvLayer的配置如下
      Sequential(
        (0): Conv2d(320, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )'''
      
      Step4: Up2(output') -> final = (bs, 1, 200, 200) # 第二个维度的1就代表BEV每个网格下的二分类结果
      '''Up2的配置如下
      Sequential(
        (0): Upsample(scale_factor=2.0, mode=bilinear)
        (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
      )'''
      ```

   **最后就是将输出的语义分割结果与binimgs的真值标注做基于像素的交叉熵损失**，从而指导模型的学习过程。