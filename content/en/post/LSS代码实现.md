---
date: 2025-04-14T04:00:59-07:00
description: ""
featured_image: "/images/lssCode/pia.jpg"
tags: ["CV", "automatic driving"]
title: "LSS代码"
---

文章搬运，自用。

论文：[Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)

官方源码：[lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)

&nbsp;

#### [Lift, Splat, Shoot图像BEV安装与模型代码详解](https://blog.csdn.net/qq_41366026/article/details/133840315)

对于任意数量不同相机帧的图像直接提取场景的BEV表达；主要由三部分实现：

1. **Lift**：将每一个相机的图像帧根据相机的内参转换提升到 frustum（锥形）形状的点云空间中。
2. **splate**：将所有相机转换到锥形点云空间中的特征根据相机的内参 K 与相机相对于 ego 的外参T映射到栅格化的 3D 空间（BEV）中来融合多帧信息。
3. **shoot**：根据上述 BEV 的检测或分割结果来生成规划的路径 proposal；从而实现可解释的端到端的路径规划任务。

注：LSS在训练的过程中并不需要激光雷达的点云来进行监督。

<!--more-->



### 代码

#### LiftSplatShoot

```python
class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        """BEV语义分割核心模型，实现Lift-Splat-Shoot三阶段
        Args:
            grid_conf (dict): BEV网格配置，包含xbound/ybound/zbound/dbound
            data_aug_conf (dict): 数据增强参数，包含final_dim等预处理信息
            outC (int): 输出通道数（分割类别数，如车辆检测为1）
        """
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # 生成BEV网格参数：dx(网格分辨率), bx(起点坐标), nx(网格数量)
        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        # 注册为不可训练参数（固定BEV几何结构）
        self.dx = nn.Parameter(dx, requires_grad=False)  # [x_res, y_res, z_res] 例如[0.5, 0.5, 20.0]
        self.bx = nn.Parameter(bx, requires_grad=False)  # [x_min, y_min, z_min] 例如[-49.75, -49.75, -10.0]
        self.nx = nn.Parameter(nx, requires_grad=False)  # [x_dim, y_dim, z_dim] 例如[200, 200, 1]

        # 网络结构参数
        self.downsample = 16  # 图像下采样倍数（相对于输入图像）
        self.camC = 64        # 图像特征通道数
        
        # 创建视锥体（定义图像平面到3D空间的采样点）
        self.frustum = self.create_frustum()  # 形状[D, H_down, W_down, 3]
        self.D, _, _, _ = self.frustum.shape   # D为深度采样数（来自dbound配置）
        
        # 编码模块
        self.camencode = CamEncode(self.D, self.camC, self.downsample)  # 图像特征编码器
        self.bevencode = BevEncode(inC=self.camC, outC=outC)            # BEV特征解码器
        
        # 加速选项：使用自定义QuickCumsum或自动微分实现体素池化
        self.use_quickcumsum = True  

    def create_frustum(self):
        """创建视锥体网格：在图像平面上生成不同深度的采样点
        返回：形状[D, fH, fW, 3]的张量，每个点包含(x, y, depth)
        """
        # 获取降采样后的特征图尺寸
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图像尺寸（如128x352）
        fH = ogfH // self.downsample  # 降采样后的高度（如128/16=8）
        fW = ogfW // self.downsample  # 降采样后的宽度（如352/16=22）

        # 生成深度采样点（dbound格式：[起始, 结束, 步长]）
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float)  # 形状[D]
        ds = ds.view(-1, 1, 1).expand(-1, fH, fW)  # 扩展为[D, fH, fW]

        # 生成图像平面x坐标（0到原图宽度-1，共fW个点）
        xs = torch.linspace(0, ogfW-1, fW, dtype=torch.float)
        xs = xs.view(1, 1, fW).expand(ds.shape[0], fH, fW)  # 扩展为[D, fH, fW]

        # 生成图像平面y坐标（0到原图高度-1，共fH个点）
        ys = torch.linspace(0, ogfH-1, fH, dtype=torch.float)
        ys = ys.view(1, fH, 1).expand(ds.shape[0], fH, fW)  # 扩展为[D, fH, fW]

        # 组合为3D坐标 (D x H x W x 3)
        frustum = torch.stack((xs, ys, ds), dim=-1)
        return nn.Parameter(frustum, requires_grad=False)  # 固定视锥体

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """计算3D几何位置（Lift阶段的核心操作）
        Args:
            rots (Tensor): 相机旋转矩阵 [B, N_cams, 3, 3]
            trans (Tensor): 相机平移向量 [B, N_cams, 3]
            intrins (Tensor): 相机内参矩阵 [B, N_cams, 3, 3]
            post_rots (Tensor): 数据增强旋转补偿 [B, N_cams, 3, 3]
            post_trans (Tensor): 数据增强平移补偿 [B, N_cams, 3]
        Returns:
            Tensor: 3D点坐标 [B, N, D, fH, fW, 3]（车辆坐标系）
        """
        B, N, _ = trans.shape  # B: batch_size, N: 摄像头数量

        # 步骤1：消除数据增强的影响（逆变换）
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # 平移补偿
        # 旋转补偿（应用逆矩阵）
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
        points = points.squeeze(-1)  # 结果形状 [B, N, D, fH, fW, 3]

        # 步骤2：相机坐标系 -> 车辆坐标系
        # 将图像坐标(u, v, depth)转换为相机坐标(x, y, z)
        points = torch.cat((
            points[..., :2] * points[..., 2:3],  # (u*depth, v*depth)
            points[..., 2:3]                     # depth
        ), dim=-1)

        # 计算组合变换矩阵：R @ K^-1
        combine = rots @ torch.inverse(intrins)  # [B, N, 3, 3]
        # 应用变换矩阵
        points = combine.view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
        points = points.squeeze(-1)  # [B, N, D, fH, fW, 3]
        
        # 添加平移量
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        """提取图像特征（每个摄像头独立处理）
        Args:
            x (Tensor): 输入图像 [B, N, C, H, W]
        Returns:
            Tensor: 图像特征 [B, N, D, fH, fW, C]
        """
        B, N, C, H, W = x.shape
        
        # 合并批次和摄像头维度
        x = x.view(B * N, C, H, W)
        # 通过图像编码器（CNN + 深度方向处理）
        x = self.camencode(x)  # 输出形状 [B*N, camC, D, fH, fW]
        
        # 恢复原始维度并调整顺序
        x = x.view(B, N, self.camC, self.D, H//self.downsample, W//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, fH, fW, camC]
        return x

    def voxel_pooling(self, geom_feats, x):
        """体素池化操作（Splat阶段的核心）
        Args:
            geom_feats (Tensor): 3D坐标 [B, N, D, fH, fW, 3]
            x (Tensor): 图像特征 [B, N, D, fH, fW, C]
        Returns:
            Tensor: BEV特征图 [B, C, H_bev, W_bev]
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W  # 总点数

        # 展平特征和坐标
        x = x.reshape(Nprime, C)  # [Nprime, C]
        geom_feats = geom_feats.reshape(Nprime, 3)  # [Nprime, 3]

        # 计算每个点对应的BEV网格索引
        geom_feats = ((geom_feats - (self.bx - self.dx/2)) / self.dx).long()
        
        # 添加批次索引
        batch_ix = torch.cat([
            torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) 
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # [Nprime, 4]（x,y,z,batch）

        # 过滤超出边界的点
        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &
            (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &
            (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 按体素索引排序（相同体素的特征连续存放）
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) +
            geom_feats[:, 1] * (self.nx[2] * B) +
            geom_feats[:, 2] * B +
            geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 累积求和：将同一体素内的特征相加
        if self.use_quickcumsum:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        else:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # 构建BEV特征图
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
        # 合并Z维度（因zbound通常设置为单层）
        final = torch.cat(final.unbind(dim=2), 1)  # [B, C*Z, X, Y]
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """整合Lift+Splat阶段
        Returns:
            Tensor: BEV特征图 [B, C, H_bev, W_bev]
        """
        # Lift：获取3D几何坐标
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # 提取图像特征
        x = self.get_cam_feats(x)
        # Splat：体素池化
        x = self.voxel_pooling(geom, x)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """前向传播
        Args:
            x (Tensor): 输入图像 [B, N, C, H, W]
            其他参数同get_geometry
        Returns:
            Tensor: BEV分割图 [B, outC, H_bev, W_bev]
        """
        # 步骤1-2：Lift + Splat
        bev_feats = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        # 步骤3：Shoot（BEV解码）
        return self.bevencode(bev_feats)
```





语义分割部分（BevEncode）：

```python
class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        """BEV特征解码器，生成语义分割结果
        Args:
            inC (int): 输入特征通道数（来自Lift-Splat阶段的BEV特征维度）
            outC (int): 输出通道数（分割类别数，如车辆检测为1）
        """
        super(BevEncode, self).__init__()
        
        # 骨干网络：基于ResNet-18结构（移除预训练，适配BEV特征）
        trunk = resnet18(pretrained=False, zero_init_residual=True)  # 从零开始训练
        
        # --- 特征提取部分（适配ResNet输入）---
        # 修改第一层卷积，适配输入通道数（原始ResNet输入为3通道，此处输入为inC通道）
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 下采样2倍
        self.bn1 = trunk.bn1    # 批归一化层
        self.relu = trunk.relu  # 激活函数
        
        # --- 复用ResNet中间层 ---
        self.layer1 = trunk.layer1  # ResNet第一个残差块组 [64 → 64]
        self.layer2 = trunk.layer2  # 第二个残差块组 [64 → 128]
        self.layer3 = trunk.layer3  # 第三个残差块组 [128 → 256] （原始ResNet-18有4个layer，此处只用到layer3）

        # --- 上采样解码部分 ---
        # 第一次上采样：融合深层特征（layer3输出）和浅层特征（layer1输出）
        self.up1 = Up(64+256, 256, scale_factor=4)  # 自定义上采样模块（通道拼接+卷积）
        
        # 最终上采样：生成分割结果
        self.up2 = nn.Sequential(
            # 上采样2倍（与layer2的下采样抵消，最终恢复1/2分辨率）
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 形状扩大2倍
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),        # 通道压缩
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),  # 1x1卷积输出分割结果
            # 注：此处未加Sigmoid，因训练代码中可能使用带Sigmoid的损失函数（如BCEWithLogitsLoss）
        )

    def forward(self, x):
        """前向传播
        Args:
            x (Tensor): BEV特征图，形状 [B, inC, H, W]
        Returns:
            Tensor: 语义分割结果，形状 [B, outC, H_out, W_out]
        """
        # 初始下采样
        x = self.conv1(x)  # [B, inC, H, W] → [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块组特征提取
        x1 = self.layer1(x)  # [B, 64, H/2, W/2] → [B, 64, H/2, W/2]
        x = self.layer2(x1)  # → [B, 128, H/4, W/4]
        x = self.layer3(x)   # → [B, 256, H/8, W/8] （假设layer3有下采样）

        # 第一次上采样：融合深层特征(x)和浅层特征(x1)
        x = self.up1(x, x1)  # [B, 256, H/8, W/8] + [B, 64, H/2, W/2] → [B, 256, H/2, W/2]
        
        # 最终上采样：恢复至接近输入分辨率（假设原始BEV特征图是[H, W]）
        x = self.up2(x)      # [B, 256, H/2, W/2] → [B, outC, H, W]
        
        return x  # 输出语义分割logits（需配合Sigmoid/Softmax使用）
```



#### 
