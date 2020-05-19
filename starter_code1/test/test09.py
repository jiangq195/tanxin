def searchMatrix(matrix, target):
    i, j = len(matrix) - 1, 0
    result = False
    while i >= 0 and j < len(matrix[0]):
        if matrix[i][j] < target:
            j += 1
        elif matrix[i][j] > target:
            i -= 1
        else:
            result = True

    return result


m = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
target = 5


res = searchMatrix(m, target)
print(res)
