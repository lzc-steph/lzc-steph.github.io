---
date: 2025-01-25T11:00:59-04:00
description: ""
featured_image: "/images/algorithm/hash.jpg"
tags: ["algorithm"]
title: "hash map"

---

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

```c++
// 根据用户 ID 查找用户
struct my_struct *find_user(int user_id) {
    struct my_struct *s;

    // 查找元素
    HASH_FIND_INT(users, &user_id, s);
    return s;
}
```

<!--more-->

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
