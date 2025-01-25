---
date: 2025-01-22T11:00:59-04:00
description: ""
featured_image: "/images/algorithm/lucky.jpg"
tags: ["algorithm"]
title: "「持续更新」算法题笔记"
---



# 哈希表

1. ### 两数之和

   **题目描述**：给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

   **方法**：找数数 `x`，寻找数组中是否存在 `target - x`。

   使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N) 降低到 O(1) —— 创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存在 target - x，然后将 x 插入到哈希表中，即可保证不会让 x 和自己匹配。

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

   

