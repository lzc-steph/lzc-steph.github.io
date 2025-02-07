---
date: 2025-02-05T11:00:59-04:00
description: ""
featured_image: "/images/c++/lucky.jpg"
tags: ["algorithm"]
title: "c++语法"
---

在 C++ 中，`using namespace std;`指令允许用户使用 std 命名空间中的所有标识符，而无需在它们前面加上` std::`。



### 标准输出

C++ 的标准输出是 `cout`，用 `<<` 运算符把需要打印的内容传递给 `cout`，`endl` 是换行符。

```c++
#include <iostream>
int a = 10;

// 输出：10
std::cout << a << std::endl;

// 可以串联输出
// 输出：Hello, World!
std::cout << "Hello" << ", " << "World!" << std::endl;

string s = "abc";
// 输出：abc 10
std::cout << s << " " << a << std::endl;
```





### 动态数组 `vector`

1. `vector` 的初始化方法如下：

   ```c++
   #include <vector>
   using namespace std;
   
   int n = 7, m = 8;
   
   // 初始化一个 int 型的空数组 nums
   vector<int> nums;
   
   // 初始化一个大小为 n 的数组 nums，数组中的值默认都为 0
   vector<int> nums(n);
   
   // 初始化一个元素为 1, 3, 5 的数组 nums
   vector<int> nums{1, 3, 5};
   
   // 初始化一个大小为 n 的数组 nums，其值全都为 2
   vector<int> nums(n, 2);
   
   // 初始化一个二维 int 数组 dp
   vector<vector<int>> dp;
   
   // 初始化一个大小为 m * n 的布尔数组 dp，
   // 其中的值都初始化为 true
   vector<vector<bool>> dp(m, vector<bool>(n, true));
   ```

2. `vector` 的常用操作：

   ```c++
   #include <iostream>
   #include <vector>
   using namespace std;
   
   int main() {
       int n = 10;
       // 数组大小为 10，元素值都为 0
       vector<int> nums(n);
       // 输出 0 (false)
       cout << nums.empty() << endl;
       // 输出：10
       cout << nums.size() << endl;
   
       // 在数组尾部插入一个元素 20
       nums.push_back(20);
       // 输出：11
       cout << nums.size() << endl;
   
       // 得到数组最后一个元素的引用
       // 输出：20
       cout << nums.back() << endl;
   
       // 删除数组的最后一个元素（无返回值）
       nums.pop_back();
       // 输出：10
       cout << nums.size() << endl;
   
       // 可以通过方括号直接取值或修改
       nums[0] = 11;
       // 输出：11
       cout << nums[0] << endl;
   
       // 在索引 3 处插入一个元素 99
       nums.insert(nums.begin() + 3, 99);
   
       // 删除索引 2 处的元素
       nums.erase(nums.begin() + 2);
   
       // 交换 nums[0] 和 nums[1]
       swap(nums[0], nums[1]);
   
       // 遍历数组
       // 0 11 99 0 0 0 0 0 0 0
       for (int i = 0; i < nums.size(); i++) {
           cout << nums[i] << " ";
       }
       cout << endl;
   }
   ```





### 双链表 `list`

`list` 是 C++ 标准库中的双向链表容器。

1. 初始化方法：

   ```c++
   #include <list>
   
   int n = 7;
   
   // 初始化一个空的双向链表 lst
   std::list<int> lst;
   
   // 初始化一个大小为 n 的链表 lst，链表中的值默认都为 0
   std::list<int> lst(n);
   
   // 初始化一个包含元素 1, 3, 5 的链表 lst
   std::list<int> lst{1, 3, 5};
   
   // 初始化一个大小为 n 的链表 lst，其中值都为 2
   std::list<int> lst(n, 2);
   ```

2. `list` 的常用方法：

   ```c++
   #include <iostream>
   #include <list>
   using namespace std;
   
   int main() {
       // 初始化链表
       list<int> lst{1, 2, 3, 4, 5};
   
       // 检查链表是否为空，输出：false
       cout << lst.empty() << endl;
   
       // 获取链表的大小，输出：5
       cout << lst.size() << endl;
   
       // 在链表头部插入元素 0
       lst.push_front(0);
       // 在链表尾部插入元素 6
       lst.push_back(6);
   
       // 获取链表头部和尾部元素，输出：0 6
       cout << lst.front() << " " << lst.back() << endl;
   
       // 删除链表头部元素
       lst.pop_front();
       // 删除链表尾部元素
       lst.pop_back();
   
       // 在链表中插入元素
       auto it = lst.begin();	//begin()函数用于返回指向容器或数组第一个元素的迭代器。
       // 移动到第三个位置
       advance(it, 2);		//advance()函数是用于将迭代器前进或后退指定长度的距离
       // 在第三个位置插入 99
       lst.insert(it, 99);
   
       // 删除链表中某个元素
       it = lst.begin();
       // 移动到第二个位置
       advance(it, 1);
       // 删除第二个位置的元素
       lst.erase(it);
   
       // 遍历链表
       // 输出：1 99 3 4 5
       for (int val : lst) {
           cout << val << " ";
       }
       cout << endl;
   
       return 0;
   }
   ```

   一般来说，当我们想在**头部增删元素**时会使用**双链表**，它在头部增删元素的效率比 `vector` 高。

   通过**索引访问元素**，这种场景下我们会使用 `vector`。





### 队列 `queue`

`queue` 基于先进先出（FIFO）的原则。队列适用于只允许从一端（队尾）添加元素、从另一端（队头）移除元素的场景。

```c++
#include <iostream>
#include <queue>
using namespace std;

int main() {
    // 初始化一个空的整型队列 q
    queue<int> q;

    // 在队尾添加元素
    q.push(10);
    q.push(20);
    q.push(30);

    // 检查队列是否为空，输出：false
    cout << q.empty() << endl;

    // 获取队列的大小，输出：3
    cout << q.size() << endl;

    // 获取队列的队头和队尾元素，输出：10 和 30
    cout << q.front() << " " << q.back() << endl;

    // 删除队头元素
    q.pop();

    // 输出新的队头元素：20
    cout << q.front() << endl;

    return 0;
}
```





###  栈 `stack`

栈是一种后进先出（LIFO）的数据结构，栈适用于只允许在一端（栈顶）添加或移除元素的场景。

```c++
#include <iostream>
#include <stack>
using namespace std;

int main() {

    // 初始化一个空的整型栈 s
    stack<int> s;

    // 向栈顶添加元素
    s.push(10);
    s.push(20);
    s.push(30);

    // 检查栈是否为空，输出：false
    cout << s.empty() << endl;

    // 获取栈的大小，输出：3
    cout << s.size() << endl;

    // 获取栈顶元素，输出：30
    cout << s.top() << endl;

    // 删除栈顶元素
    s.pop();

    // 输出新的栈顶元素：20
    cout << s.top() << endl;

    return 0;
}
```





### 哈希表 `unordered_map`

`unordered_map` 提供了基于键值对（key-value）的存储，提供了常数时间复杂度的查找、插入和删除键值对的操作。

1. 初始化方法：

   ```c++
   #include <unordered_map>
   using namespace std;
   
   // 初始化一个空的哈希表 map
   unordered_map<int, string> hashmap;
   
   // 初始化一个包含一些键值对的哈希表 map
   unordered_map<int, string> hashmap{{1, "one"}, {2, "two"}, {3, "three"}};
   ```

2. `unordered_map` 的常用方法

   > [!CAUTION]
   >
   > **在 C++ 的哈希表中，如果你访问一个不存在的键，它会自动创建这个键，对应的值是默认构造的值**。
   >
   > 记住访问值之前要先判断键是否存在，否则可能会意外地创建新键，导致算法出错。详见下面的示例。

   ```c++
   #include <iostream>
   #include <unordered_map>
   using namespace std;
   
   int main() {
       // 初始化哈希表
       unordered_map<int, string> hashmap{{1, "one"}, {2, "two"}, {3, "three"}};
   
       // 检查哈希表是否为空，输出：0 (false)
       cout << hashmap.empty() << endl;
   
       // 获取哈希表的大小，输出：3
       cout << hashmap.size() << endl;
   
       // 查找指定键是否存在
       // 注意 contains 方法是 C++20 新增的
       // 输出：Key 2 -> two
       if (hashmap.contains(2)) {
           cout << "Key 2 -> " << hashmap[2] << endl;
       } else {
           cout << "Key 2 not found." << endl;
       }
   
       // 获取指定键对应的值，若不存在会返回默认构造的值
       // 输出空字符串
       cout << hashmap[4] << endl;
   
       // 插入一个新的键值对
       hashmap[4] = "four";
   
       // 获取新插入的值，输出：four
       cout << hashmap[4] << endl;
   
       // 删除键值对
       hashmap.erase(3);
   
       // 检查删除后键 3 是否存在
       // 输出：Key 3 not found.
       if (hashmap.contains(3)) {
           cout << "Key 3 -> " << hashmap[3] << endl;
       } else {
           cout << "Key 3 not found." << endl;
       }
   
       // 遍历哈希表
       // 输出（顺序可能不同）：
       // 4 -> four
       // 2 -> two
       // 1 -> one
       for (const auto &pair: hashmap) {
           cout << pair.first << " -> " << pair.second << endl;
       }
   
       // 特别注意，访问不存在的键会自动创建这个键
       unordered_map<int, string> hashmap2;
   
       // 键值对的数量是 0
       cout << hashmap2.size() << endl; // 0
   
       // 访问不存在的键，会自动创建这个键，对应的值是默认构造的值
       cout << hashmap2[1] << endl; // empty string
       cout << hashmap2[2] << endl; // empty string
   
       // 现在键值对的数量是 2
       cout << hashmap2.size() << endl; // 2
   
       return 0;
   }
   ```

   



### 哈希集合 `unordered_set`

`unordered_set` 用于存储不重复的元素，常见使用场景是**对元素进行去重**。

1. 初始化方法：

   ```c++
   #include <unordered_set>
   using namespace std;
   
   // 初始化一个空的哈希集合 set
   unordered_set<int> uset;
   
   // 初始化一个包含一些元素的哈希集合 set
   unordered_set<int> uset{1, 2, 3, 4};`
   ```

2. `unordered_set` 的常用方法：

   ```c++
   #include <iostream>
   #include <unordered_set>
   using namespace std;
   
   int main() {
       // 初始化哈希集合
       unordered_set<int> hashset{1, 2, 3, 4};
   
       // 检查哈希集合是否为空，输出：0 (false)
       cout << hashset.empty() << endl;
   
       // 获取哈希集合的大小，输出：4
       cout << hashset.size() << endl;
   
       // 查找指定元素是否存在
       // 输出：Element 3 found.
       if (hashset.contains(3)) {
           cout << "Element 3 found." << endl;
       } else {
           cout << "Element 3 not found." << endl;
       }
   
       // 插入一个新的元素
       hashset.insert(5);
   
       // 删除一个元素
       hashset.erase(2);
       // 输出：Element 2 not found.
       if (hashset.contains(2)) {
           cout << "Element 2 found." << endl;
       } else {
           cout << "Element 2 not found." << endl;
       }
   
       // 遍历哈希集合
       // 输出（顺序可能不同）：
       // 1
       // 3
       // 4
       // 5
       for (const auto &element : hashset) {
           cout << element << endl;
       }
   
       return 0;
   }
   ```






### 传值和传引用

函数参数的传递方式主要有两种：**传值**和**传引用**。

#### 传值（Pass by Value）

**传值**是指将函数参数的一个副本传递给函数，在函数内部对该副本的修改**不会影响**到原始数据。

下面是一个例子：

```c++
#include <iostream>
using namespace std;

void modifyValue(int x) {
    x = 10;  // 只修改副本，不会影响原始数据
}

int main() {
    int num = 5;
    modifyValue(num);
    // 输出：5
    cout << "After modifyValue, num = " << num << endl;
    return 0;
}
```

在上述代码中，`num` 的值在调用 `modifyValue` 后并未改变，因为传入的是 `num` 的副本，函数内的修改仅影响副本。



#### 传引用（Pass by Reference）

**传引用**是指将实参的地址传递给函数，函数可以直接操作原始数据。这意味着**对参数的修改会直接影响原始数据**。

```c++
#include <iostream>
using namespace std;

void modifyReference(int &x) {
    x = 10;  // 修改原始数据
}

int main() {
    int num = 5;
    modifyReference(num);
    // 输出：10
    cout << "After modifyReference, num = " << num << endl;
    return 0;
}
```

在上述代码中，`num` 的值被修改为 10，因为我们传递的是 `num` 的引用，函数内对 `x` 的修改直接影响了 `num`。



#### 做算法题时的选择

如果是传递基本类型，比如 `int`、`bool` 等，用传值比较多，因为这类数据一般不需要在函数内部修改，而且复制得开销很小。

如果是传递容器，比如 `vector`、`unordered_map` 等，用传引用比较多，因为可以避免复制数据副本的开销，而且容器一般需要在函数内部修改。

特别注意: **递归函数的参数千万别使用传值的方式，否则每次递归都会创建一个数据副本，消耗大量的内存和时间**。