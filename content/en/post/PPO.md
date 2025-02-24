---
date: 2025-02-23T11:00:59-04:00
description: ""
featured_image: "/images/PPO/lucky.jpg"
tags: ["RL"]
title: "PPO"
---

### 1. 基础概念

![1](/images/PPO/1.png)

1. **enviroment**：看到的画面+看不到的后台画面，不了解细节
2. **agent(智能体)**：根据策略得到尽可能多的奖励
3. **state**：当前状态
4. **observation**：state的一部分（有时候agent无法看全）
5. **action**：agent做出的动作

6. **reward**：agent做出一个动作后环境给予的奖励

7. **action space**：可以选择的动作，如上下左右

8. **policy**：策略函数，输入state，输出Action的**概率分布**。一般用π表示。

   1. 训练时应尝试各种action
   2. 输出应具有多样性

9. **Trajectory/Episode/Rollout**：轨迹，用t表示，一连串状态和动作的序列。

   有的状态转移是确定的，也有的是不确定的。

10. **Return**：回报，从当前时间点到游戏结束的 Reward 的累积和。

强化学习目标：训练一个Policy神经网络π，在所有状态S下，给出相应的Action，得到Return的期望最大。

&nbsp;

### 2. Policy gradient

目标：求return期望的最大值

1. #### **计算过程**

   ![2](/images/PPO/2.png)

   ![3](/images/PPO/3.png)

   + 直观理解：

     对所有可能的 trajectory 期望最大的梯度。可以用这个梯度乘学习率去更新神经网络里的参数。

     <!--more-->

   + 若去掉梯度，则表达式的意义：若一个**trajectory 得到的 return 大于零**，则**增大**这个trajectory里所有状态下，采取当前action的概率。

2. #### **训练policy神经网络**

   + **输入**：当前画面

   + **输出**：action 的概率

   ![4](/images/PPO/4.png)

   玩n场游戏后，得到n个trajectory的最后的return值

   此时可以得到loss里的所有值，可以进行一个batch训练，来更新policy神经网络

   ![5](/images/PPO/5.png)

   存在问题：大部分时间在采集数据，很慢

&nbsp;

### 3. actor-critic 算法

+ #### 可改进：

  1. 是否增大某动作概率，应该看做了该动作之后，到游戏结束累积的 reward；而不是整个 trajectory 的 reward。因为一个动作只能影响它之后的 reward，无法影响它之前的 reward。
  2. 某动作可能只影响接下来的几步，且影响逐步衰减。后面的reward更多是被当时的action影响。

+ #### 修改公式：actor-critic

  ![6](/images/PPO/6.png)

  + 对当前动作到结束的reward进行求和

    ![7](/images/PPO/7.png)

  + 引入 gamma

  + 为了避免训练慢：对所有的动作减去一个baseline，得到的就是该动作相对于其他动作的好坏

  actor：用来做动作的神经网络

  critic：评估动作好坏的网络

![8](/images/PPO/8.png)

&nbsp;

### 4. 计算优势函数(Advantage Function)

1. **动作价值函数**

    ![9](/images/PPO/9.png)

2. **优势函数计算公式**

   ![10](/images/PPO/10.png)

之前需要训练两个神经网络（动作+状态），变成只需要训练一个代表动作价值的函数。

![11](/images/PPO/11.png)

+ #### Generalized Advantage Estimation (GAE)

  全部考虑，权重随着步数增加而降低

  ![12](/images/PPO/12.png)

&nbsp;

### 5. 基础回顾

![13](/images/PPO/13.png)

目标：策略梯度优化目标函数值(即第三个式子)越大越好

+ 状态价值函数用神经网络拟合，它可以和策略函数公用网络参数，只是最后一层不同：状态价值函数在最后一层输出一个单一值代表当前价值即可：

  统计当前步到 trajectory 结束，所有 reward的加减加和，衰减系数用gamma控制。用网络拟合retuen值即可。

  ![14](/images/PPO/14.png)

&nbsp;

&nbsp;

## 6. 邻近策略优化算法(Proximal Policy Optimization, PPO)









π