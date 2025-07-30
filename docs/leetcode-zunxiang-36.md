## 设计

## 回溯

## 动态规划

### [276. 栅栏涂色](https://leetcode.cn/problems/paint-fence/)

有 `k` 种颜色的涂料和一个包含 `n` 个栅栏柱的栅栏，请你按下述规则为栅栏设计涂色方案：

- 每个栅栏柱可以用其中 **一种** 颜色进行上色。
- 相邻的栅栏柱 **最多连续两个** 颜色相同。

给你两个整数 `k` 和 `n` ，返回所有有效的涂色 **方案数** 。

**示例 1：**

![img](./assets/paintfenceex1.png)

```
输入：n = 3, k = 2
输出：6
解释：所有的可能涂色方案如上图所示。注意，全涂红或者全涂绿的方案属于无效方案，因为相邻的栅栏柱 最多连续两个 颜色相同。
```

**示例 2：**

```
输入：n = 1, k = 1
输出：1
```

**示例 3：**

```
输入：n = 7, k = 2
输出：42
```

**提示：**

- `1 <= n <= 50`
- `1 <= k <= 105`
- 题目数据保证：对于输入的 `n` 和 `k` ，其答案在范围 `[0, 231 - 1]` 内

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if k == 1 and n <= 2:
            return 1
        elif k == 1:
            return 0
        elif n == 1:
            return k
        
        @cache
        def dp(n, used):
            if n == 1 and used:
                return k - 1
            elif n == 1 and not used:
                return k
            if used:
                return (k - 1) * dp(n - 1, False)  
            else:
                return dp(n - 1, True) + (k - 1) * dp(n - 1, False)
        

        return k * dp(n - 1, False)
        
```

todo：更好的动态规划方法



### [256. 粉刷房子](https://leetcode.cn/problems/paint-house/)

假如有一排房子，共 `n` 个，每个房子可以被粉刷成红色、蓝色或者绿色这三种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 `n x 3` 的正整数矩阵 `costs` 来表示的。

例如，`costs[0][0]` 表示第 0 号房子粉刷成红色的成本花费；`costs[1][2]` 表示第 1 号房子粉刷成绿色的花费，以此类推。

请计算出粉刷完所有房子最少的花费成本。

**示例 1：**

```
输入: costs = [[17,2,17],[16,16,5],[14,3,19]]
输出: 10
解释: 将 0 号房子粉刷成蓝色，1 号房子粉刷成绿色，2 号房子粉刷成蓝色。
     最少花费: 2 + 5 + 3 = 10。
```

**示例 2：**

```
输入: costs = [[7,6,2]]
输出: 2
```

**提示:**

- `costs.length == n`
- `costs[i].length == 3`
- `1 <= n <= 100`
- `1 <= costs[i][j] <= 20`

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        @cache
        def dp(i, used):
            if i == -1:
                return 0
            min_cost = inf
            for j in range(3):
                if j != used:
                    min_cost = min(min_cost, costs[i][j] + dp(i - 1, j))
            return min_cost
        return dp(len(costs) - 1, -1)
     
```



### [265. 粉刷房子 II](https://leetcode.cn/problems/paint-house-ii/)

假如有一排房子共有 `n` 幢，每个房子可以被粉刷成 `k` 种颜色中的一种。房子粉刷成不同颜色的花费成本也是不同的。你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。

每个房子粉刷成不同颜色的花费以一个 `n x k` 的矩阵表示。

- 例如，`costs[0][0]` 表示第 `0` 幢房子粉刷成 `0` 号颜色的成本；`costs[1][2]` 表示第 `1` 幢房子粉刷成 `2` 号颜色的成本，以此类推。

返回 *粉刷完所有房子的最低成本* 。

**示例 1：**

```
输入: costs = [[1,5,3],[2,9,4]]
输出: 5
解释: 
将房子 0 刷成 0 号颜色，房子 1 刷成 2 号颜色。花费: 1 + 4 = 5; 
或者将 房子 0 刷成 2 号颜色，房子 1 刷成 0 号颜色。花费: 3 + 2 = 5. 
```

**示例 \**2:\****

```
输入: costs = [[1,3],[2,4]]
输出: 5
```

**提示：**

- `costs.length == n`
- `costs[i].length == k`
- `1 <= n <= 100`
- `2 <= k <= 20`
- `1 <= costs[i][j] <= 20`

**进阶：**您能否在 `O(nk)` 的时间复杂度下解决此问题？

```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        @cache
        def dp(i, used):
            if i == -1:
                return 0
            min_cost = inf
            for j in range(len(costs[0])):
                if j != used:
                    min_cost = min(min_cost, costs[i][j] + dp(i - 1, j))
            return min_cost
        return dp(len(costs) - 1, -1)
      
```

### [651. 四个键的键盘](https://leetcode.cn/problems/4-keys-keyboard/)

假设你有一个特殊的键盘包含下面的按键：

- `A`：在屏幕上打印一个 `'A'`。
- `Ctrl-A`：选中整个屏幕。
- `Ctrl-C`：复制选中区域到缓冲区。
- `Ctrl-V`：将缓冲区内容输出到上次输入的结束位置，并显示在屏幕上。

现在，*你可以 **最多** 按键 `n` 次（使用上述四种按键），返回屏幕上最多可以显示 `'A'` 的个数* 。

 

**示例 1:**

```
输入: n = 3
输出: 3
解释: 
我们最多可以在屏幕上显示三个'A'通过如下顺序按键：
A, A, A
```

**示例 2:**

```
输入: n = 7
输出: 9
解释: 
我们最多可以在屏幕上显示九个'A'通过如下顺序按键：
A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
```

 

**提示:**

- `1 <= n <= 50`

**关键是找出最后一步可能的前继**

```py
class Solution:
    def maxA(self, n: int) -> int:
        best = [0, 1]
        for k in range(2, n+1):
            best.append(max(best[x] * (k - x - 1) for x in range(k - 1)))
            best[-1] = max(best[-1], best[-2] + 1)
        return best[n]
```

