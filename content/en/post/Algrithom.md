---
date: 2025-01-22T11:00:59-04:00
description: ""
featured_image: "/images/algorithm/lucky.jpg"
tags: ["algorithm"]
title: "「持续更新」算法题笔记"
---



# 哈希表

`UT_hash_handle hh` 是 `uthash` 库中的一个关键结构体，用于实现哈希表。

### 1. `UT_hash_handle hh` 的作用
`UT_hash_handle hh` 是哈希表中每个元素必须包含的结构体，用于存储哈希表的内部信息（如哈希值、链表指针等）。`uthash` 通过这个结构体来管理哈希表中的元素。

### 2. 使用方法
在使用 `uthash` 时，通常会在自定义的结构体中包含 `UT_hash_handle hh`，并通过宏操作来管理哈希表。

#### 示例代码
```c
#include "uthash.h"

// 自定义结构体
struct my_struct {
    int id;                    // 键值
    char name[10];             // 数据
    UT_hash_handle hh;         // 必须包含的哈希句柄
};

// 全局哈希表指针
struct my_struct *users = NULL;
```

```c
// 添加用户到哈希表
void add_user(int user_id, const char *name) {
    struct my_struct *s;

    // 检查是否已存在相同键值的元素
    HASH_FIND_INT(users, &user_id, s);
    if (s == NULL) {
        // 不存在则创建新元素
        s = (struct my_struct *)malloc(sizeof(struct my_struct));
        s->id = user_id;
        HASH_ADD_INT(users, id, s);  // 添加到哈希表
    }
    strcpy(s->name, name);
}
```

```c
// 根据用户 ID 查找用户
struct my_struct *find_user(int user_id) {
    struct my_struct *s;

    // 查找元素
    HASH_FIND_INT(users, &user_id, s);
    return s;
}
```

```c
// 删除用户
void delete_user(struct my_struct *user) {
    // 删除元素
    HASH_DEL(users, user);
    free(user);  // 释放内存
}
```

```c
// 打印哈希表中的所有用户
void print_users() {
    struct my_struct *s;

    // 遍历哈希表
    for (s = users; s != NULL; s = (struct my_struct *)(s->hh.next)) {
        printf("user id %d: name %s\n", s->id, s->name);
    }
}
```

### 3. 关键点

- **`UT_hash_handle hh`**：必须包含在自定义结构体中，用于哈希表管理。
- **`HASH_ADD_INT`**：将元素添加到哈希表，`INT` 表示键值为整数类型。
- **`HASH_FIND_INT`**：根据键值查找元素。
- **`HASH_DEL`**：从哈希表中删除元素。
- **遍历哈希表**：通过 `hh.next` 指针遍历哈希表中的元素。





- ### 例题两数之和

  **题目描述**：给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

  **方法**：找数 `x`，寻找数组中是否存在 `target - x`。

  使用哈希表，可以将寻找 `target - x`的时间复杂度降低到从 O(N) 降低到 O(1) —— 创建一个哈希表，对于每一个 `x`，我们首先查询哈希表中是否存在 `target - x`，然后将 `x`插入到哈希表中，即可保证不会让 `x`和自己匹配。

  ```c
  struct hashTable {
      int key;
      int val;
      UT_hash_handle hh;
  };
  
  struct hashTable* hashtable;
  
  struct hashTable* find(int ikey) {
      struct hashTable* tmp;
      HASH_FIND_INT(hashtable, &ikey, tmp);
      return tmp;
  }
  
  void insert(int ikey, int ival) {
      struct hashTable* it = find(ikey);
      if (it == NULL) {
          struct hashTable* tmp = malloc(sizeof(struct hashTable));
          tmp->key = ikey, tmp->val = ival;
          HASH_ADD_INT(hashtable, key, tmp);
      } else {
          it->val = ival;
      }
  }
  
  int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
      hashtable = NULL;
      for (int i = 0; i < numsSize; i++) {
          struct hashTable* it = find(target - nums[i]);
          if (it != NULL) {
              int* ret = malloc(sizeof(int) * 2);
              ret[0] = it->val, ret[1] = i;
              *returnSize = 2;
              return ret;
          }
          insert(nums[i], i);
      }
      *returnSize = 0;
      return NULL;
  }
  ```

  

