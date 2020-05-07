

def permutation(S):
    ans = []

    S = sorted(S)

    def backtrack(pre_str, r_str):
        if not len(r_str):
            ans.append(pre_str)
        else:
            pre = ''
            for i in range(len(r_str)):
                if r_str[i] != pre:
                    backtrack(pre_str + r_str[i], r_str[:i] + r_str[i + 1:])
                pre = r_str[i]

    backtrack('', S)
    return ans


res = permutation('abcda')

print(res)
