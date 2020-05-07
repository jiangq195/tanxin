# dict1 = {'a': 1, 'b': 2, 'c': 2}
# new_s = []
# new_s = str([s for s, num in dict1.items()])


def compressString(S):
    if not S:
        return ""
    new_s = ''
    num = 0
    prev = S[0]

    for s in S:
        if s == prev:
            num += 1
        else:
            new_s += prev + str(num)
            prev = s
            num = 1
    new_s += prev + str(num)

    return new_s


# res = compressString("aabcccccaa")
# print(res)

def waysToStep(n):
    a = 1
    b = 1
    c = 2
    if n == 0:
        return a
    if n == 1:
        return b
    if n == 2:
        return c
    if n > 2:
        for i in range(2, n):
            a, b, c, = b, c, a + b + c
        return c


res = waysToStep(61)
print(res)
