---
date: 2025-02-05T11:00:59-04:00
description: ""
featured_image: "/images/c++STL/lucky.jpg"
tags: ["algorithm"]
title: "算法 - C++STL常用容器"
---

### **1. string 字符串**
```cpp
#include <string>
using namespace std;

string s = "Hello";
s += " World";          // 拼接

char first_char = str[0];  // 访问第1个字符（索引为0）'H'
string sub = s.substr(6, 5); // substr(起始位置, 截取长度) → "World"
size_t pos = s.find("World");// 查找子串，返回位置，未找到返回 string::npos

s.length();              // 字符串长度
s.empty();               // 判断是否为空
s.starts_with("He");     // C++20 判断前缀
s.contains("or");        // C++20 判断是否包含子串
```

---

### **2. vector 动态数组**
```cpp
#include <vector>
vector<int> v;

// 核心操作
v.push_back(10);         // 尾部插入
v.pop_back();            // 尾部删除
v.size();                // 元素数量
v[0] = 5;               // 随机访问（不检查越界）
v.at(0);                // 带越界检查的访问

// 遍历方式
for (auto num : v) { /* 遍历 */ }
for (int i=0; i<v.size(); ++i) { v[i]; }

// 特性
vector<int> v2(100, 0);  // 初始化 100 个 0
vector<vector<int>> mat; // 二维数组（矩阵）
```

---

### **3. map 有序键值对**
```cpp
#include <map>
map<string, int> m;

m["apple"] = 5;         // 插入/修改键值
m.erase("apple");       // 删除键
m.size();               // 键值对数量
m.contains("apple");    // C++20 判断键是否存在

// 遍历（按键升序）
for (auto& [key, val] : m) { 
    cout << key << ": " << val << endl; 
}

// 注意：访问 m["key"] 时若 key 不存在，会自动插入默认值！
```

---

### **4. set 有序唯一集合**
```cpp
#include <set>
set<int> s;

s.insert(3);            // 插入元素（自动去重）
s.erase(3);             // 删除元素
s.contains(3);          // 判断存在性
s.size();               // 元素数量

// 遍历（按升序）
for (auto num : s) { cout << num << " "; }

// 典型用途：排序+去重
vector<int> nums = {2,1,2,3};
set<int> tmp(nums.begin(), nums.end()); // → {1,2,3}
```

---

### 5. queue 队列

```c++
queue<int> q;		// 初始化

q.push(10);		// 添加元素

cout << q.front() 	// 访问队首
cout << q.back()   // 访问队尾

q.pop();		// 移除队首元素，
q.empty();		// 检查队列是否为空
std::cout << q.size();  // 获取队列大小
```
#### 5.5 priority_queue 优先队列

`priority_queue` 默认是一个 **最大堆**（Max Heap），即优先级最高的元素（默认是最大的元素）会最先出队。

插入操作是通过 `push()` 或 `emplace()` 来完成的，而不是 `push_back`。

```c++
priority_queue<int> pq; // 定义：默认是最大堆

pq.push(10); // 插入元素 10
pq.push(5);  // 插入元素 5
int topElement = pq.top(); // 获取堆顶元素（最大值）
pq.pop(); // 删除堆顶元素
```

定义最小堆：

```c++
// greater<int> 是一个比较函数，表示元素越小优先级越高
priority_queue<int, vector<int>, greater<int>> minHeap;
```

对于 `priority_queue<pair<int, int>>`，默认情况下，它会按照 `pair` 的**第一个元素**（即 `first`）进行排序。如果第一个元素相同，则会比较第二个元素（即 `second`）

&nbsp;

### 6 pair<int, int>

用于存储两个值（可以是相同类型或不同类型)

### 7 tuple<int, int, int>

用于存储三个值（可以是相同类型或不同类型)

&nbsp;

1. **map 遍历有序性**：题目若要求无序输出需改用 `unordered_map`（哈希表，O(1)访问）
2. **set 去重原理**：插入时若值已存在则忽略，需去重时优先考虑
3. **复杂度**：
   - vector 尾部操作 O(1)，中间插入 O(n)
   - map/set 增删查 O(log n)

