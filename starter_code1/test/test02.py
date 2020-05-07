import numpy as np


# a = np.array([[1, 1], [1, 0]])
# print(a)
# print(a.T)
# # u = np.dot(a, a.T)
# # lamda, hU = np.linalg.eig(u)
# #
# #
# # print(lamda)
# # print(hU)
#
# u, lamda, vT = np.linalg.svd(a)
# print('---------------/n')
# print(u)
# print(lamda)
# print(vT)
#
# b = np.array([1,1])
# print(b)
# print(b.reshape(2,1))


def get_fb(num):
    """
    求斐波那契数列第n个值
    时间复杂度O(1)
    """
    if num == 1:
        return 1
    if num == 2:
        return 1
    if num > 2:
        A = np.array([[1, 1],
                      [1, 0]])
        base_matrix = np.array([[1],
                                [1]])
        # 对A求特征值分解
        lamda, Q = np.linalg.eig(A)

        # 构建对角矩阵
        lamdas = np.diag([np.power(lamda[0], num - 2), np.power(lamda[1], num - 2)])

        # Q*lamda*Q(**-1)*base_matrix
        temp = np.dot(Q, lamdas)
        temp = np.dot(temp, np.linalg.pinv(Q))

        final_matrix = np.dot(temp, base_matrix)

        return int(final_matrix[0])


# 打印前10个数
for i in range(1, 11):
    print(get_fb(i), end=', ')
