---
date: '2024-11-14T22:29:34+08:00'
draft: true
title: 'Python刷题技巧'
tags: 
- null
categories: null
comment : true
hidden: false
showToc: true # show contents
TocOpen: true 
---

本文主要总结介绍使用Python刷题时会用上的一些技巧。（为什么选择Python？当然是写起来简单了~

主要是：

1. 数据结构的使用方法
2. 库函数的使用方法
3. 代码组织方法

不会设计到太过工程和实现相关的内容（虽然去年实习结束后很想写这样一篇博客，但是还是因为懒惰没有写成

本节内容主要是在刷 leetcode hot100 时总结的。

1. 数字范围 `range(start, end)` range(i - 1, -1, -1)


交换赋值：
                        nums[j], nums[j + 1] = nums[j + 1], nums[j]


# 迭代器

## 字典（hashtable）

新建：
```python
        hashtable = dict()
```

遍历： 
enumerate(xxx) -> 得到index 和 value
ht.values() -> 得到value

记得初始化：
```python
            if key not in ht:
                ht[key] = []
```


collections.defaultdict 是 Python 中一个非常有用的工具，特别适合用于处理字典的默认值。使用 defaultdict(list) 可以方便地创建一个字典，其中的每个值都是一个列表，这样在向字典中添加新元素时，如果该键不存在，会自动创建一个空列表。

以下是如何使用 collections.defaultdict(list) 的示例：

python

Copy
from collections import defaultdict

# 创建一个 defaultdict，默认值为列表
mp = defaultdict(list)

# 向 defaultdict 添加元素
mp['a'].append(1)
mp['a'].append(2)
mp['b'].append(3)

# 打印 defaultdict
print(mp)


## 列表

list(xxx) 新建list
v.sort() 原地排序 默认ASCD
sorted(v) 排序后返回新对象
v.append(x)
v.pop()
[] = False for while

`leftMax = [height[0]] + [0] * (n - 1)`
反向遍历可以只操作下标：
`for i in range(n - 2, -1, -1):`

迭代器操作:
`ans = sum(min(leftMax[i], rightMax[i]) - height[i] for i in range(n))`

stack top:
`stack[-1]`

# Hot100

## 哈希

[两数之和](https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked): 第二遍扫描可转化为哈希表查找X
[字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked): 先排序，再哈希
[最长连续序列](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked): 每个数查看其邻居是否存在（哈希一下）

哈希： O(1)迅速解决存在性问题

## 双指针
[移动零](https://leetcode.cn/problems/move-zeroes/?envType=study-plan-v2&envId=top-100-liked): 仿照冒泡->重复计算; 一个指针负责其左边区间维护性质，一个指针负责辅助交换。
[盛水最多的容器](https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked): 双指针，限定遍历范围。代表可能的区间。
[三数之和](https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-100-liked): 操，这么难，我去年咋做出来的？？
总是还是先想着暴力，然后再思考优化！！！！
不要一开始就优化方法...
当我们需要枚举数组中的两个元素时，如果我们发现随着第一个元素的递增，第二个元素是递减的
（其实这一步哈希也可以）
所以两数之和也可以双指针了？？（但是排序N(LOGN)，不是买椟还珠哦吗）


[接雨水](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked): 这个明天一定得做出来...
做之前顺便再整理我的之前笔记...
AC!!!

1. 思路1：如何暴力求解 -> 从低向上扫描 -> 太慢 -> 优化： 从高到低扫描 -> 先计算水位，再计算面积
2. 思路2：计算水位 -> 观察：水位等于最低边界 -> 观察：存水的地方都是凹形状 -> 什么是凹？低的一边是单侧高峰！
3. 两边扫描先求出两侧高峰，然后一遍扫描求出所有凹区间，计算水位，计算面积

思路和官方题解动态规划思路相同，但代码需要优化...

方法二：单调栈
一层一层接雨水...
这谁想得到

方法三：双指针
方法一升级版
但感觉也很难...
统一化处理：也有接水为0的地方