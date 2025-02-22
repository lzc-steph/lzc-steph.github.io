---
date: 2025-02-20T11:00:59-04:00
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



### 2. Policy gradient

目标：求return期望的最大值

1. #### **计算过程**

   ![2](/images/PPO/2.png)

   ![3](/images/PPO/3.png)

   + 直观理解：

     对所有可能的 trajectory 期望最大的梯度。可以用这个梯度乘学习率去更新神经网络里的参数。

   + 若去掉梯度，则表达式的意义：

      若一个**trajectory 得到的 return 大于零**，则**增大**这个trajectory里所有状态下，采取当前action的概率。

2. #### **训练policy神经网络**

   + **输入**：当前画面

   + **输出**：action 的概率

   ![4](/images/PPO/4.png)

   玩n场游戏后，得到n个trajectory的最后的return值

   此时可以得到loss里的所有值，可以进行一个batch训练，来更新policy神经网络

   ![5](/images/PPO/5.png)

   存在问题：大部分时间在采集数据，很慢











π