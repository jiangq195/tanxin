# dict1 = {'a': 1, 'b': 2, 'c': 2}
# new_s = []
# new_s = str([s for s, num in dict1.items()])


def compressString(S):
    dict1 = []
    head = 0
    tail = 0
    prev = S[0]
    len1 = len(S)
    for i in range(len1):
        if S[i] != prev:
            dict1.append(S[head:i])
            prev = s
            head = tail


# res = compressString("aabcccccaa")
# print(res)

# a = []
# a.push(1)
# print(a)

a = [0, 1, 2, 3]
print(a.index(2))
