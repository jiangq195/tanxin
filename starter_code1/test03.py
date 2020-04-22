from functools import reduce

str = 'a b'
print(len(str))

a = 10
b = 20
print(a and b)

f = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5], 0)
print(f)

import numpy as np

arr = np.array([1, 2, 3], dtype=np.int32)

np.savetxt('any_array.txt', arr, delimiter='#')


np.linalg.eig