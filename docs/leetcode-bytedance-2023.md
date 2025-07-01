# 2023 7月

## 新题

### [455. 分发饼干](https://leetcode.cn/problems/assign-cookies/)

注意，python的sort函数是**从小到大**排序

（以及heapq是最小堆）

以及注意 sorted 和 sort 的区别

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        ans = 0
        if s == []:
            return 0
        g.sort()
        s.sort()
        cur = len(s) - 1
        for i in range(len(g) - 1, -1, -1):
            if s[cur] >= g[i]:
                ans += 1
                cur -= 1
                if cur == -1:
                    return ans
        return ans
```

更简单的写法（显然也可从小到大贪心）

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        n = len(g)
        i = 0
        for x in s:
            if i < n and g[i] <= x:
                i += 1
        return i
链接：https://leetcode.cn/problems/assign-cookies/solutions/2974809/pai-xu-shuang-zhi-zhen-jian-ji-xie-fa-py-ttn8/
```

17min



### [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

记得每次循环后清空cur

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        cur = []
        if root == None:
            return []
        q = deque([(root, 0)])
        while q:
            node, layer = q.popleft()
            cur.append(node.val)
            if len(q) == 0 or layer != q[0][1]:
                if layer % 2 == 1:
                    cur.reverse()
                ans.append(cur[:])
                cur = []
            for sub in [node.left, node.right]:
                if sub != None:
                    q.append((sub, layer + 1))
        return ans
```

5min

### [344. 反转字符串](https://leetcode.cn/problems/reverse-string/)

我的写法

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        for i in range(len(s) // 2):
            s[i], s[len(s) - i - 1] = s[len(s) - i - 1], s[i]
```

但可以更简单

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        for i in range(len(s) // 2):
            s[i], s[-i - 1] = s[-i - 1], s[i]
```

< 1 min

## 旧题

[199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)  11min 犯了和第一次做同样的错误，下标0/-1搞错

[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) 7min 写得稍微比第一次好些

[33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)  30min...但思路比之前清楚了许多...

[215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/) emm...快速选择写对非常不容易啊

[146. LRU 缓存](https://leetcode.cn/problems/lru-cache/) todo

# 2023 年 6 月

## 新题

[225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

[88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

## 旧题

[25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

[15. 三数之和](https://leetcode.cn/problems/3sum/)

[31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

# 2023年 5 月

## 新题

[310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)

[852. 山脉数组的峰顶索引](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)

## 旧题

[92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

[121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)