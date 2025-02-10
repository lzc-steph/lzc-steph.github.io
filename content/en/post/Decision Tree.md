---
date: 2025-01-11T10:58:08-04:00
description: "决策树是一种用于分类和回归的机器学习算法，通过树状结构进行决策。它由节点、分支和叶子节点组成，每个内部节点表示一个特征测试，分支代表测试结果，叶子节点则输出类别或数值。"
featured_image: "/images/chapter1/taytay.HEIC"
tags: ["machine learning"]
title: "Decision Tree"
---

![1](/images/chapter1/1.png)

## 熵和信息增益

**Measuring purity**

![2](/images/chapter1/2.png)

![3](/images/chapter1/3.png)

选择信息增益更大的分裂特征

![4](/images/chapter1/4.png)

## 决策树训练(递归)

1. Start with all examples at the root node
2. Calculate information gain for all possible features, and pick the one with the highest information gain
3. Split dataset according to selected feature, and create left and right branches of the tree
4. Keep repeating splitting process until stopping criteria is met:
   - When a node is 100% one class
   - When splitting a node will result in the tree exceeding a maximum depth
   - Information gain from additional splits is less than threshold
   - When number of examples in a node is below a threshold

<!--more-->

+ **取值为多个离散值**

  创造独热编码

+ **取值为连续值**

  ![5](/images/chapter1/5.png)



---



# 随机森林

### 使用多个决策树

单个决策树的缺点：对数据太敏感

构造多个树，最后投票

### 替换取样（Sampling with replacement）

构造一个新的数据集

![6](/images/chapter1/6.png)

## 随机森林算法

```python
Given training set of size m
	For b = 1 to B:
	Use sampling with replacement to create a new training set of size m
	Train a decision tree on the new
```

![7](/images/chapter1/7.png)

- 改进：**Randomizing the feature choice**

  At each node, when choosing a feature to use to split, if n features are available, pick a random subset of k <n features and allow the algorithm to only choose from that subset of features.

  更加健壮



---



# XGBoost

### Boosted trees intuition

重点训练之前学得不好的部分

```python
Given training set of size m
	For b = 1 to B:
	Use sampling with replacement to create a new training set of size m
		But instead of picking from all examples with equal (1/m) probability, make it more likely to pick examples that the previously trained trees misclassify
	Train a decision tree on the new dataset
```

![8](/images/chapter1/8.png)

### XBoost (eXtreme Gradient Boosting)

- Open source implementation of boosted trees
- Fast efficient implementation
- Good choice of default splitting criteria and criteria for when to stop splitting
- Built in regularization to prevent overfitting
- Highly competitive algorithm for machine learning competitions (eg: Kaggle competitions)
- 给不同训练例子分配了不同的权重

```python
from xgboost import XGBClassifier
model = XGBClassifier ()
model. fit(X_train, Y_train)
y_pred = model predict (X_test)
```

```python
from xgboost import XGBRegressor
model = XGBRegressor ()
model.fit(X_train, y_train)
y_pred = model predict (X_test)
```

# Decision Trees vs Neural Networks

- **Decision Trees and Tree ensembles**

  一次训练一棵树

  - Works well on tabular (structured) data
  - Not recommended for unstructured data (images, audio, text)
  - Fast
  - Small decision trees may be human interpretable

- **Neural Networks**

  用梯度下降训练

  - Works well on all types of data, including tabular (structured) and unstructured data
  - May be slower than a decision tree
  - Works with transfer learning
  - When building a system of multiple models working together, it might be easier to string together multiple neural networks
