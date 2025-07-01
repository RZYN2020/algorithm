625. 最长有效括号
626. 两数相加
627. 无重复字符的最长子串
628. 零钱兑换
629. 数据流的中位数

625. 正则表达式匹配排序数组
626. 合并两个有序数组



### [546. 移除盒子](https://leetcode.cn/problems/remove-boxes/)

这题在 Codeforces 上有 2400 分 [CF1107E](https://leetcode.cn/link/?target=https%3A%2F%2Fcodeforces.com%2Fproblemset%2Fproblem%2F1107%2FE)，换算成力扣难度分的话至少有 3000。

https://codeforces.com/problemset/problem/1107/E

本题可以使用三维动态规划来解决，定义状态 `dp[i][j][k]` 表示在区间 `[i, j]` 内，且在 `boxes[i]` 左边有 `k` 个与 `boxes[i]` 颜色相同的盒子时，移除该区间内所有盒子能获得的最大积分...

1. **直接移除 `boxes[i]` 及其左边的 `k` 个相同颜色的盒子**：此时先移除 `boxes[i]` 以及它左边的 `k` 个相同颜色的盒子，获得 `(k + 1) * (k + 1)` 积分，然后继续处理区间 `[i + 1, j]`，即 `dp[i + 1][j][0]`。
2. **先移除中间的盒子，再移除 `boxes[i]` 及其左边的相同颜色的盒子**：遍历区间 `[i + 1, j]`，找到与 `boxes[i]` 颜色相同的盒子 `boxes[m]`，先移除区间 `[i + 1, m - 1]` 内的盒子，获得 `dp[i + 1][m - 1][0]` 积分，然后将 `boxes[i]` 与 `boxes[m]` 合并，继续处理区间 `[m, j]`，此时 `boxes[m]` 左边有 `k + 1` 个与它颜色相同的盒子，即 `dp[m][j][k + 1]`。

```python
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        @cache
        def dfs(left: int, right: int, same: int) -> int:
            if left > right:
                return 0
            res = dfs(left, right - 1, 0) + (same + 1) ** 2
            for i in range(left, right):
                if boxes[i] == boxes[right]:
                    res = max(res, dfs(left, i, same + 1) + dfs(i + 1, right - 1, 0))
            return res
        return dfs(0, len(boxes) - 1, 0)
```

### 625 最小因式分解

https://github.com/doocs/leetcode/blob/main/solution/0600-0699/0625.Minimum%20Factorization/README.md

https://zhuanlan.zhihu.com/p/523712273

https://zhuanlan.zhihu.com/p/602415886

```python
class Solution:
    def smallestFactorization(self, a: int) -> int:
        # 如果 a 小于 10，直接返回 a
        if a < 10:
            return a
        # 用于存储分解得到的数字
        factors = []
        # 从 9 到 2 遍历，尝试分解 a
        for i in range(9, 1, -1):
            while a % i == 0:
                factors.append(i)
                a //= i
        # 如果最终 a 仍然大于 1，说明无法分解
        if a > 1:
            return 0
        # 将 factors 列表中的数字按从小到大的顺序排序
        factors.sort()
        # 将 factors 列表中的数字组合成一个整数
        result = int(''.join(map(str, factors)))
        # 检查结果是否超出 32 位有符号整数的范围
        if result > 2**31 - 1:
            return 0
        return result
```

### [354. 俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/)

思路是有了：先遍历构建树，然后求最大深度

构建树时，每个节点从根部开始一直判断到合适的leave

或者...直接dfs...看起来可行？但有些bug...

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        def insert_point(envs, e):
            if len(envs) == 0:
                return 0
            l = 0
            r = len(envs) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if envs[mid][0] > e[0] and envs[mid][1] > e[1]:
                    l = mid + 1
                else:
                    r = mid - 1
            if l == 0 and not (envs[0][0] < e[0] and envs[0][1] < e[1]):
                return -1
            if l == len(envs) and not (envs[-1][0] > e[0] and envs[-1][1] > e[1]):
                return -1
            return l
        def dfs(i, cur):
            if i == len(envelopes):
                return len(cur)
            res = dfs(i + 1, cur)
            e = envelopes[i]
            p = insert_point(cur, e)
            if p == -1:
                return res
            else:
                cur = cur[:]
                cur.insert(p, e)
                return max(res, dfs(i + 1, cur))
        return dfs(0, [])
```

看看answer吧

排序宽，只求高，不就是最长递增子序列吗...

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        
        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))

        f = [1] * n
        for i in range(n):
            for j in range(i):
                if envelopes[j][1] < envelopes[i][1]:
                    f[i] = max(f[i], f[j] + 1)
        
        return max(f)
```

### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

经典老题了，一个 dummy node 即可（环形链表）

```python
class Node:
    # 提高访问属性的速度，并节省内存
    __slots__ = 'prev', 'next', 'key', 'value'

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dummy = Node()  # 哨兵节点
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = dict()

    # 获取 key 对应的节点，同时把该节点移到链表头部
    def get_node(self, key: int) -> Optional[Node]:
        if key not in self.key_to_node:  # 没有这本书
            return None
        node = self.key_to_node[key]  # 有这本书
        self.remove(node)  # 把这本书抽出来
        self.push_front(node)  # 放在最上面
        return node

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1

    def put(self, key: int, value: int) -> None:
        node = self.get_node(key)
        if node:  # 有这本书
            node.value = value  # 更新 value
            return
        self.key_to_node[key] = node = Node(key, value)  # 新书
        self.push_front(node)  # 放在最上面
        if len(self.key_to_node) > self.capacity:  # 书太多了
            back_node = self.dummy.prev
            del self.key_to_node[back_node.key]
            self.remove(back_node)  # 去掉最后一本书

    # 删除一个节点（抽出一本书）
    def remove(self, x: Node) -> None:
        x.prev.next = x.next
        x.next.prev = x.prev

    # 在链表头添加一个节点（把一本书放在最上面）
    def push_front(self, x: Node) -> None:
        x.prev = self.dummy
        x.next = self.dummy.next
        x.prev.next = x
        x.next.prev = x
```

