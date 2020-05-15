class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def __init__(self):
        self.res = None
        self.s = None

    def inorderSuccessor(self, root, p):
        def dfs(data):
            if data is None:
                return
            dfs(data.left)
            if self.s == p:
                self.res = data
            self.s = data
            dfs(data.right)

        dfs(root)
        return self.res


def verifyPostorder(postorder):
    def cur(i, j):
        if i == j:
            return True
        p = i
        while postorder[p] < postorder[j]:
            p += 1
        m = p
        while postorder[p] > postorder[j]:
            p += 1

        return p == j and cur(i, m - 1) and cur(m, j - 1)

    return cur(0, len(postorder) - 1)


print(verifyPostorder([1, 2, 5, 10, 6, 9, 4, 3]))
