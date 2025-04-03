---
date: 2025-04-02T01:00:59-05:00
description: ""
featured_image: "/images/ac/pia.jpg"
tags: ["RL"]
title: "Actor-Critic"
---

Actor-Critic 算法本质上是基于策略的算法，因为这一系列算法的**目标都是优化一个带参数的策略，只是会额外学习价值函数**，从而帮助策略函数更好地学习。

Actor-Critic 算法则可以在每一步之后都进行更新，并且不对任务的步数做限制。

+ 更一般形式的策略梯度

  ![1](/images/ac/1.png)

&nbsp;

### 1. Actor（策略网络）

Actor 要做的是与环境交互，并在 Critic 价值函数的指导下用策略梯度学习一个更好的策略。

Actor 的更新采用策略梯度的原则。

### 2.  Critic（价值网络）

Critic 要做的是通过 Actor 与环境交互收集的数据学习一个价值函数，这个价值函数会用于判断在当前状态什么动作是好的，什么动作不是好的，进而帮助 Actor 进行策略更新。

<!--more-->

+ 价值函数的损失函数：

  ![2](/images/ac/2.png)

#### Actor-Critic 算法的具体流程:

![3](/images/ac/3.png)

![4](/images/ac/4.png)

+ **策略网络（Actor）**：参数为 θ，定义策略 πθ(a∣s)，输出动作的概率分布。

+ **价值网络（Critic）**：参数为 ω，输出状态价值 Vω(s)，评估当前状态的好坏。

&nbsp;

```python
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
```

1. #### 定义策略网络 `PolicyNet`

   ```python
   class PolicyNet(torch.nn.Module):
       def __init__(self, state_dim, hidden_dim, action_dim):
           super(PolicyNet, self).__init__()
           self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
           self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
   
       def forward(self, x):
           x = F.relu(self.fc1(x))
           return F.softmax(self.fc2(x), dim=1)
   ```

   &nbsp;

2. #### 定义价值网络 `ValueNet`

   输入是某个状态，输出则是状态的价值。

   ```python
   class ValueNet(torch.nn.Module):
       def __init__(self, state_dim, hidden_dim):
           super(ValueNet, self).__init__()
           self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
           self.fc2 = torch.nn.Linear(hidden_dim, 1)
   
       def forward(self, x):
           x = F.relu(self.fc1(x))
           return self.fc2(x)
   ```

   &nbsp;

3. #### 定义`ActorCritic`算法

   包含采取动作（`take_action()`）和更新网络参数（`update()`）两个函数。

   ```python
   class ActorCritic:
       def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device):
           
           self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
           self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
           
           self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                   lr=actor_lr)    # 策略网络优化器
           self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=critic_lr)  # 价值网络优化器
           self.gamma = gamma
           self.device = device
   
       def take_action(self, state):
           state = torch.tensor([state], dtype=torch.float).to(self.device)
           probs = self.actor(state)
           action_dist = torch.distributions.Categorical(probs)
           action = action_dist.sample()
           return action.item()
   
       def update(self, transition_dict):
           states = torch.tensor(transition_dict['states'],
                                 dtype=torch.float).to(self.device)
           actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
               self.device)
           rewards = torch.tensor(transition_dict['rewards'],
                                  dtype=torch.float).view(-1, 1).to(self.device)
           next_states = torch.tensor(transition_dict['next_states'],
                                      dtype=torch.float).to(self.device)
           dones = torch.tensor(transition_dict['dones'],
                                dtype=torch.float).view(-1, 1).to(self.device)
   
           # 时序差分目标
           td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                          dones)
           td_delta = td_target - self.critic(states)  # 时序差分误差
           log_probs = torch.log(self.actor(states).gather(1, actions))
           actor_loss = torch.mean(-log_probs * td_delta.detach())
           # 均方误差损失函数
           critic_loss = torch.mean(
               F.mse_loss(self.critic(states), td_target.detach()))
           self.actor_optimizer.zero_grad()
           self.critic_optimizer.zero_grad()
           actor_loss.backward()  # 计算策略网络的梯度
           critic_loss.backward()  # 计算价值网络的梯度
           self.actor_optimizer.step()  # 更新策略网络的参数
           self.critic_optimizer.step()  # 更新价值网络的参数
   ```

   &nbsp;

4. #### 在车杆环境上训练

   ```python
   actor_lr = 1e-3
   critic_lr = 1e-2
   num_episodes = 1000
   hidden_dim = 128
   gamma = 0.98
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
       "cpu")
   
   env_name = 'CartPole-v0'
   env = gym.make(env_name)
   env.seed(0)
   torch.manual_seed(0)
   state_dim = env.observation_space.shape[0]
   action_dim = env.action_space.n
   agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                       gamma, device)
   
   return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
   ```

   

