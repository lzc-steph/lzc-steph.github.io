---
date: 2025-02-12T11:00:59-04:00
description: ""
featured_image: "/images/algorithm/lucky.jpg"
tags: ["algorithm"]
title: "「持续更新」算法题笔记"
---

1. ### 两数之和 - hash

   **题目描述**：给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

   **方法**：找数 `x`，寻找数组中是否存在 `target - x`。

   使用哈希表，可以将寻找 `target - x`的时间复杂度降低到从 O(N) 降低到 O(1) —— 创建一个哈希表，对于每一个 `x`，我们首先查询哈希表中是否存在 `target - x`，然后将 `x`插入到哈希表中，即可保证不会让 `x`和自己匹配。

   ```c++
   class Solution {
   public:
       vector<int> twoSum(vector<int>& nums, int target) {
           unordered_map<int, int> hashtable;
           for (int i = 0; i < nums.size(); ++i) {
               auto it = hashtable.find(target - nums[i]);
               if (it != hashtable.end()) {
                   return {it->second, i};
               }
               hashtable[nums[i]] = i;
           }
           return {};
       }
   };
   ```
   
   1. **`find` 方法**
   
      `find` 是哈希表的一个成员函数，用于查找指定的键（key）。它的作用是：
   
      - 如果哈希表中存在该键，则返回一个指向该键值对的迭代器。
      - 如果哈希表中不存在该键，则返回 `hashtable.end()`，表示未找到。
   
   2. **`auto it`**
   
      - `auto` 是 C++11 引入的关键字，用于自动推导变量的类型。
   
        在这里，`it` 的类型会被推导为哈希表迭代器的类型（例如 `std::unordered_map<int, int>::iterator`）。
   
      - `it` 是一个**迭代器**，指向哈希表中找到的键值对（如果找到的话）。
   
      <!--more-->
   
   3.  **`return {it->second, i};`** 
   
      C++11 引入的一种语法特性，称为**初始化列表**（initializer list）。它用于**返回一个包含多个值的对象**。
   
      1. **`it->second`** 表示迭代器指向的键值对中的值（value）。
   
         在哈希表中，键值对的形式是 `{key, value}`，`it->second` 就是 `value`。
   
      2. **`{it->second, i}`**是一个初始化列表，用于构造一个包含两个值的对象。
   
      3. **`return`**
         - 返回单个值时，直接写值。
         - 返回容器或对象时，可以使用 `{}` 构造返回值。
         - 返回空容器时，使用 `{}`。
   
   



&nbsp;

2. ### 二叉树的最小深度 - D/BFS

   最小深度是从根节点到最近叶子节点的最短路径上的节点数量 —— 本质上是求最短距离。

   **说明：**叶子节点是指没有子节点的节点。
   
   1. **if-else大法**
   
      ```c++
      class Solution {
      public:
      	int minDepth(TreeNode root) {
      		if (root == null) return 0;
      		else if (root.left == null) return minDepth(root.right) + 1;
      		else if (root.right == null) return minDepth(root.left) + 1;
      		else return min(minDepth(root.left), minDepth(root.right)) + 1;
      	}
      }
      ```
   
      *min* 函数比较了两个整数*a*和*b*，并返回较小的那个值。
   
   2. **DFS** 深度遍历的解法
   
      每当遍历到一条树枝的叶子节点，就会更新最小深度，**当遍历完整棵树后**，就能算出整棵树的最小深度。
   
      ```c++
      class Solution {
      private:
          // 记录最小深度（根节点到最近的叶子节点的距离）
          int minDepthValue = INT_MAX;
          // 记录当前遍历到的节点深度
          int currentDepth = 0;
      
      public:
          int minDepth(TreeNode* root) {
              if (root == nullptr) {
                  return 0;
              }
      
              // 从根节点开始 DFS 遍历
              traverse(root);
              return minDepthValue;
          }
      
          void traverse(TreeNode* root) {
              if (root == nullptr) {
                  return;
              }
      
              // 前序位置进入节点时增加当前深度
              currentDepth++;
      
              // 如果当前节点是叶子节点，更新最小深度
              if (root->left == nullptr && root->right == nullptr) {
                  minDepthValue = min(minDepthValue, currentDepth);
              }
      
              traverse(root->left);
              traverse(root->right);
      
              // 后序位置离开节点时减少当前深度
              currentDepth--;
          }
      };
      ```
   
   3. **BFS** 层序遍历的解法
   
      按照 BFS 从上到下逐层遍历二叉树的特点，当遍历到第一个叶子节点时，就能得到最小深度。
   
      ```c++
      class Solution {
      public:
          int minDepth(TreeNode* root) {
              if (root == nullptr) return 0;
              queue<TreeNode*> q;
              q.push(root);
              // root 本身就是一层，depth 初始化为 1
              int depth = 1;
      
              while (!q.empty()) {
                  int sz = q.size();
                  // 遍历当前层的节点
                  for (int i = 0; i < sz; i++) {
                      TreeNode* cur = q.front();
                      q.pop();
                      // 判断是否到达叶子结点
                      if (cur->left == nullptr && cur->right == nullptr)
                          return depth;
                      // 将下一层节点加入队列
                      if (cur->left != nullptr)
                          q.push(cur->left);
                      if (cur->right != nullptr)
                          q.push(cur->right);
                  }
                  // 这里增加步数
                  depth++;
              }
              return depth;
          }
      };
      ```
   
      1. **`pop()` 函数**
   
         该函数没有参数，也没有返回值。它仅用于删除栈顶元素。
   
      2. **`front()` 函数**
   
         返回对容器中第一个元素的引用。



&nbsp;

3. 

















