import random


def down_tower(m):
    # 高为m的塔
    Y_list = []
    for i in range(1, m):
        if m % i == 0:
            Y_list.append(i)

    return Y_list


def tower_breakers(n, list):
    for i in range(n):
        y_list = down_tower(list[i])
        if list1[i] == 1 or (not y_list):
            continue
        else:
            list1[i] = random.choices(y_list)[0]
            return True

    return False


exp_num = int(input("请输入测试用例数量>>"))
for i in range(exp_num):
    n, m = input("请输入测试用例:").split()
    n = int(n)
    m = int(m)
    list1 = [m] * n
    num = 0
    while True:
        res = tower_breakers(n, list1)
        if not res:
            print(num+1)
            break
        num = (num+1) % 2

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

