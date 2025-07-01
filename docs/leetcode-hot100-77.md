> 2025.2.11

部分代码来自官方题解

## 二叉树

二叉树题目一般通过遍历（dfs 和 bfs）即可求解。

其中略微有趣的题目有：

1. [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)
   1. 注意对称和相等的区别
2. [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)
   1. 注意每次递归时需要判断字节点值是否处于某个**范围**
   2. 以及**二叉搜索树的中序遍历序列就是有序的**
3. [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)
   1. 层序遍历需要bfs
4. [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)
   1. [其他的LCA算法](https://oi-wiki.org/graph/lca/#tarjan-%E7%AE%97%E6%B3%95)

## 图论

同二叉树算法，图算法一般均可以有DFS和BFS变化得到。

有以下重点算法：

1. 遍历相关
2. MST（最小生成树）
3. 最短路径
4. 最大流最小割

下面只涉及到遍历相关的算法。

遍历算法可以抽象为以下：

```
WhateverFirstSearch(s):
	put s into the bag
	while the bag is not empty
		take v from the bag
		if v is unmarked
			mark v
		for each edge vw
			put w into the bag
```

bag为stack -> DFS

bag为queue -> BFS

bag为优先队列 -> 某些适用于贪心算法的最优问题（如Dijkstra算法求最短路径）



下面两题遍历+标记即可求解：

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

[994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

下面一题实际为判断图中是否有环。可以通过拓扑排序判断。

[207. 课程表](https://leetcode.cn/problems/course-schedule/)

以及一个实现trie树的题： [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

## 回溯

回溯算法即是递归遍历时对全局状态也不断修改，以穷尽所有情况。

### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

可以动态规划预处理

### [79. 单词搜索](https://leetcode.cn/problems/word-search/)

> 给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。
>
> 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

典型的回溯做法可参考本题。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        found = False
        path = []
        
        def backtrack(c, i, j):
            nonlocal found
            if board[i][j] != word[c]:
                return
            if c == len(word) - 1:
                found = True
                return
            path.append((i, j))
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0  <= ni < len(board) and 0 <= nj < len(board[0]):
                    if not (ni, nj) in path:
                        backtrack(c + 1, ni, nj)
            path.pop()

        for i in range(len(board)):
            for j in range(len(board[0])):
                backtrack(0, i, j)
                if found:
                    return True
        return False
```



### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

本题有一种简便做法

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        return list(map(''.join, product(*(phoneMap[digit] for digit in digits))))

```

