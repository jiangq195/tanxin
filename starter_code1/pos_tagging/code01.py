import numpy as np

tag2id, id2tag = {}, {}
word2id, id2word = {}, {}

for line in open('./traindata.txt'):
    items = line.split('/')
    word, tag = items[0], items[1].rstrip()  # 抽取每一行的单词和词性

    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word

    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

M = len(word2id)  # M:词典的大小
N = len(tag2id)  # N:词性的种类个数

pi = np.zeros(N)  # 每个词性出现在句子中第一个位置的概率
A = np.zeros((N, M))  # 给定tag i，出现单词j的概率
B = np.zeros((N, N))  # 之前的状态是i，之后转换成状态j的概率

prev_tag = ""
for line in open('./traindata.txt'):
    items = line.split('/')
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    A[tagId][wordId] += 1
    if prev_tag == "":  # 这意味着是句子的开始
        pi[tagId] += 1
    else:
        B[tag2id[prev_tag]][tagId] += 1

    if items[0] == ".":
        prev_tag = ""
    else:
        prev_tag = items[1].rstrip()

# normalize
pi = pi / sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])


# 到此为止计算完了模型的所有参数

def log(v):
    if v == 0:
        return np.log(v + 0.000001)

    return np.log(v)


def viterbi(x, pi, A, B):
    """
    x:用户输入的字符串或者句子：“I like playing soccer”
    pi:initial probability of tags
    A:给定tag，每个单词出现的概率
    B:tag之间的转移概率
    """
    x = [word2id[word] for word in x.split()]  # x:[45,22,33]
    T = len(x)

    dp = np.zeros((T, N))  # dp[i][j]:w1.....wi 假设wi的tag是第j个tag
    # ptr = np.array([[0 for x in range(N)] for y in range(T)])  # T*N
    ptr = np.zeros((T, N), dtype=int)

    for j in range(N):  # basecase for DP算法
        dp[0][j] = log(pi[j]) + log(A[j][x[0]])

    for i in range(1, T):  # 每个单词
        for j in range(N):  # 每个词性
            # TODO: 一下几行代码可以写成一行(vectorize的操作,会使得效率更高)
            dp[i][j] = -9999999
            for k in range(N):  # 从每一个k可以到达j
                score = dp[i - 1][k] + log(B[k][j]) + log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    # decoding:把最好的sequence打印出来
    best_seq = [0] * T
    # 找出最后一个单词的词性id
    best_seq[T - 1] = np.argmax(dp[T - 1])

    # 从后向前依次找出每个单词的词性id
    for i in range(T - 2, -1, -1):
        best_seq[i] = ptr[i + 1][best_seq[i + 1]]

    # 到目前为止，best_seq存放了对应于x的词性序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])


x = "Social Security number , passport number and details about the services provided for the payment"
viterbi(x, pi, A, B)
