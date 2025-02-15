---
date: 2025-02-12T11:00:59-04:00
description: ""
featured_image: "/images/btree/lucky.jpg"
tags: ["algorithm"]
title: "二叉树"
---

# 二叉树

## 二叉树的实现方式

1. 最常见的二叉树就是类似链表那样的链式存储结构，每个二叉树节点有指向左右子节点的指针

   ```c++
   class TreeNode {
   public:
       int val;
       TreeNode* left;
       TreeNode* right;
     
       // 构造函数，参数是int x， :后面的部分是初始化列表，{} 是构造函数的函数体为空，即不需要额外的操作。
       TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
   };
   
   // 你可以这样构建一棵二叉树：
   TreeNode* root = new TreeNode(1);
   root->left = new TreeNode(2);
   root->right = new TreeNode(3);
   root->left->left = new TreeNode(4);
   root->right->left = new TreeNode(5);
   root->right->right = new TreeNode(6);
   
   // 构建出来的二叉树是这样的：
   //     1
   //    / \
   //   2   3
   //  /   / \
   // 4   5   6
   ```

   ##### `public`用法总结：

   1. `public` 关键字用于指定类成员的访问权限，允许外部代码直接访问这些成员。
   2. 如果不加 `public`，这些成员默认是 `private` 的，外部代码无法直接访问它们。

   

   在 `TreeNode` 的构造函数中，初始化列表的作用是：

   - `val(x)`：将成员变量 `val` 初始化为参数 `x` 的值。
   - `left(nullptr)`：将成员变量 `left` 初始化为 `nullptr`（表示空指针）。
   - `right(nullptr)`：将成员变量 `right` 初始化为 `nullptr`。



2. 可以用哈希表，其中的键是父节点 id，值是子节点 id 的列表（每个节点的 id 是唯一的），那么一个键值对就是一个多叉树节点：

   ```c++
       1
      / \
     2   3
    /   / \
   4   5   6
   ```

   ```c++
   // 1 -> {2, 3}
   // 2 -> {4}
   // 3 -> {5, 6}
   
   unordered_map<int, vector<int>> tree;
   tree[1] = {2, 3};
   tree[2] = {4};
   tree[3] = {5, 6};
   ```

   这样就可以模拟和操作二叉树/多叉树结构。图论中叫做 [邻接表](https://labuladong.online/algo/data-structure-basic/graph-basic/)。



&nbsp;

&nbsp;

## 二叉树的递归

### 递归遍历（DFS）

```c++
// 二叉树的遍历框架
void traverse(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    // 前序位置
    traverse(root->left);
    // 中序位置
    traverse(root->right);
    // 后序位置
}
```





&nbsp;

### 层序遍历（BFS）

1. **最简单的写法**

   ```c++
   void levelOrderTraverse(TreeNode* root) {
       if (root == nullptr) {
           return;
       }
     
       std::queue<TreeNode*> q;
       q.push(root);
       while (!q.empty()) {
           TreeNode* cur = q.front();
           q.pop();
           // 访问 cur 节点
           std::cout << cur->val << std::endl;
           
           // 把 cur 的左右子节点加入队列
           if (cur->left != nullptr) {
               q.push(cur->left);
           }
           if (cur->right != nullptr) {
               q.push(cur->right);
           }
       }
   }
   ```

   > [!NOTE]
   >
   > 缺点：无法知道当前节点在第几层。
   >
   > 知道节点的层数是个常见的需求（收集每一层的节点，或者计算二叉树的最小深度）

2. **改进版**

   每向下遍历一层，就给 `depth` 加 1。**depth**记录当前遍历到的层数；**sz**记录队列的长度；i记录的是节点 **cur**是当前层的第几个。

   ```c++
   void levelOrderTraverse(TreeNode* root) {
       if (root == nullptr) {
           return;
       }
       queue<TreeNode*> q;
       q.push(root);
     
       // 记录当前遍历到的层数（根节点视为第 1 层）
       int depth = 1;
   
       while (!q.empty()) {
           int sz = q.size();											//变量 sz 记录队列的长度
         
           for (int i = 0; i < sz; i++) {          //变量 i 记录的是节点 cur 是当前层的第几个
               TreeNode* cur = q.front();
               q.pop();
               // 访问 cur 节点，同时知道它所在的层数
               cout << "depth = " << depth << ", val = " << cur->val << endl;
   
               // 把 cur 的左右子节点加入队列
               if (cur->left != nullptr) {
                   q.push(cur->left);
               }
               if (cur->right != nullptr) {
                   q.push(cur->right);
               }
           }
         
           depth++;
       }
   }
   ```

3. **进阶版** —— 多叉树的层序遍历、图的 BFS 遍历、经典的BFS 暴力穷举算法框架

   方法二可以理解为每条树枝的权重是 1，二叉树中每个节点的深度，是从根节点到这个节点的路径权重和，且同一层的所有节点，路径权重和都是相同的。

   如果每条树枝的权重和可以是任意值，同一层节点的路径权重和就不一定相同了：**写法三在写法一的基础上添加一个 `State` 类，让每个节点自己负责维护自己的路径权重和，代码如下：**

   ```c++
   class State {
   public:
       TreeNode* node;
       int depth;
   
       State(TreeNode* node, int depth) : node(node), depth(depth) {}
   };
   
   void levelOrderTraverse(TreeNode* root) {
       if (root == nullptr) {
           return;
       }
       queue<State> q;
       // 根节点的路径权重和是 1
       q.push(State(root, 1));
   
       while (!q.empty()) {
           State cur = q.front();
           q.pop();
           // 访问 cur 节点，同时知道它的路径权重和
           cout << "depth = " << cur.depth << ", val = " << cur.node->val << endl;
   
           // 把 cur 的左右子节点加入队列
           if (cur.node->left != nullptr) {
               q.push(State(cur.node->left, cur.depth + 1));
           }
           if (cur.node->right != nullptr) {
               q.push(State(cur.node->right, cur.depth + 1));
           }
       }
   }
   ```

   这样每个节点都有了自己的 `depth` 变量，是最灵活的，可以满足所有 BFS 算法的需求.

&nbsp;

&nbsp;

&nbsp;

# 多叉树

## 多叉树的节点实现

```c++
class Node {
public:
    int val;
    vector<Node*> children;
};
```

&nbsp;

## **递归遍历**

```c++
// N 叉树的遍历框架
void traverse(Node* root) {
    if (root == nullptr) {
        return;
    }
    // 前序位置
    for (Node* child : root->children) {
        traverse(child);
    }
    // 后序位置
}
```

唯一的区别是，多叉树没有了中序位置。

&nbsp;

## 层序遍历

```c++
void levelOrderTraverse(Node* root) {
    if (root == nullptr) {
        return;
    }
    std::queue<Node*> q;
    q.push(root);
  
    while (!q.empty()) {
        Node* cur = q.front();
        q.pop();
        // 访问 cur 节点
        std::cout << cur->val << std::endl;

        // 把 cur 的所有子节点加入队列
        for (Node* child : cur->children) {
            q.push(child);
        }
    }
}
```
