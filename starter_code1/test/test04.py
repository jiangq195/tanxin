def parameter_verify(func):
    def type_verify(*keys, **kwargs):
        func(*keys, **kwargs)

    return type_verify


@parameter_verify
def addition(i, j):
    result = i + j

    print(result)
    return result


# res = addition(1, 2)
# print(res)

a = [[0, 1], [2, 3]]
s = [0, 1]
print(a[s])
