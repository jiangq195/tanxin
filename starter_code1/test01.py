# import pandas as pd
# import numpy as np
#
# graph = pd.DataFrame(np.zeros(shape=(3, 4)),
#                      index=list(range(3)),
#                      columns=list(range(1, 5)))
#
# print(graph)
# graph.loc[0, 1] = 1
# print(graph)
#
#
#
# print(np.log(1e-10))
import heapq


def get11(arr, k):
    a = []
    b = []
    for i in arr:
        heapq.heappush(a, -i)
        if len(a) > k:
            heapq.heappop(a)
    while len(a):
        b.append(-a[0])
        heapq.heappop(a)

    return b


arr = [3, 6, 5, 2, 9, 4, 8, 7]
print(get11(arr, 4))
