import json
from collections import Counter
from queue import PriorityQueue

import matplotlib.pyplot as plt


def read_corpus(filepath):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """

    with open(filepath) as f:
        data = json.load(f)

    qlist = []
    alist = []

    for item in data['data']:
        for para in item['paragraphs']:
            for qa in para['qas']:
                qlist.append(qa['question'])

                try:
                    alist.append(qa['answers'][0]['text'])

                except IndexError:
                    qlist.pop()

    assert len(qlist) == len(alist)

    return qlist, alist


# TODO: 统计一下在qlist 总共出现了多少个单词？ 总共出现了多少个不同的单词？
#       这里需要做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）

qlist, alist = read_corpus('./data/train-v2.0.json')

# q_all_list = []
#
# for q in qlist:
#     q_list = q.strip(' .!?').split()
#     q_all_list.extend(q_list)

word_count = Counter()

for q in qlist:
    q_list = q.strip(' .!?').split()
    word_count.update(q_list)

print(len(word_count.keys()))
print(sum(word_count.values()))

# TODO: 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
#       把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
#       需要使用matplotlib里的plot函数。y轴是词频

value_sort = sorted(word_count.values(), reverse=True)
plt.subplot(221)
plt.plot(value_sort)
plt.subplot(222)
plt.plot(value_sort[:2000])
plt.subplot(223)
plt.plot(value_sort[:200])
plt.subplot(224)
plt.plot(value_sort[:20])

inverse_dict = dict(zip(word_count.values(), word_count.keys()))
list_20 = [[inverse_dict[num], num] for num in value_sort[:20]]

print(list_20)

# TODO: 对于qlist, alist做文本预处理操作。 可以考虑以下几种操作：
#       1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
#       2. 转换成lower_case： 这是一个基本的操作
#       3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
#       4. 去掉出现频率很低的词：比如出现次数少于10,20....
#       5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
#       6. stemming（利用porter stemming): 因为是英文，所以stemming也是可以做的工作
#       7. 其他（如果有的话）
#       请注意，不一定要按照上面的顺序来处理，具体处理的顺序思考一下，然后选择一个合理的顺序
#  hint: 停用词用什么数据结构来存储？ 不一样的数据结构会带来完全不一样的效率！


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math

# 加载nltk的自带停用词
sw = set(stopwords.words('english'))
# 个人感觉对于一个问题而言这些词不应该删去？？？？
# sw -= {'who', 'when', 'why', 'where', 'how', 'which'}
# 这里只是随便去了下符号
sw.update(["\'s", "‘’", "\'\'"])
ps = PorterStemmer()


def text_preprocessing(text):
    seg = list()
    # 直接使用nltk分词
    for word in word_tokenize(text):
        # 小写化、词干提取
        word = ps.stem(word.lower())
        # 数值归一
        word = '#number' if word.isdigit() else word
        if len(word) > 1 and word not in sw:
            seg.append(word)

    return seg


words_cnt = Counter()
qlist_seg = list()
for text in qlist:
    seg = text_preprocessing(text)
    qlist_seg.append(seg)
    words_cnt.update(seg)

value_sort = sorted(words_cnt.values(), reverse=True)

# 根据Zipf定律计算99%覆盖率下的过滤词频
min_tf = value_sort[int(math.exp(0.99 * math.log(len(words_cnt))))]

# 对于每个句子，去掉一些低频词语
for cur in range(len(qlist_seg)):
    qlist_seg[cur] = [word for word in qlist_seg[cur] if words_cnt[word] > min_tf]

# TODO: 把qlist中的每一个问题字符串转换成tf-idf向量, 转换之后的结果存储在X矩阵里。 X的大小是： N* D的矩阵。 这里N是问题的个数（样本个数），
#       D是字典库的大小。

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()  # 定义一个tf-idf的vectorizer
X = vectorizer.fit_transform([' '.join(seg) for seg in qlist_seg])  # 结果存放在X矩阵


# # TODO: 矩阵X有什么特点？ 计算一下它的稀疏度
def sparsity_ratio(X):
    return 1.0 - X.nnz / float(X.shape[0] * X.shape[1])


print(X.shape)
print("input sparsity ratio:", sparsity_ratio(X))  # 打印稀疏度(sparsity)


def top5results(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    q_vector = vectorizer.transform([' '.join(text_preprocessing(input_q))])
    # 计算余弦相似度,tfidf默认使用L2范数；矩阵乘法
    sim = (X * q_vector.T).toarray()
    # 使用优先队列找出top5，最先出来的是相似度小的
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], cur))
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的(存在qlist里的)问题的下表
    # hint: 利用priority queue来找top results

    return [alist[i] for i in top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# TODO: 编写几个测试用例，并输出结果
print(top5results("Which airport was shut down?"))  # 在问题库中存在，经过对比，返回的首结果正确
print(top5results("Which airport is closed?"))
print(top5results("What government blocked aid after Cyclone Nargis?"))
print(top5results("Which government stopped aid after Hurricane Nargis?"))

# TODO: 基于倒排表的优化。在这里，我们可以定义一个类似于hash_map, 比如 inverted_index = {}， 然后存放包含每一个关键词的文档出现在了什么位置，
#       也就是，通过关键词的搜索首先来判断包含这些关键词的文档（比如出现至少一个），然后对于candidates问题做相似度比较。
#

from collections import defaultdict

inverted_idx = defaultdict(set)  # 制定一个一个简单的倒排表
for cur in range(len(qlist_seg)):
    for word in qlist_seg[cur]:
        inverted_idx[word].add(cur)


#         if cur < 5:
#             print(inverted_idx)  # 看一下倒排表的效果

def top5results_invidx(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    seg = text_preprocessing(input_q)
    candidates = set()
    for word in seg:
        # 取所有包含任意一个词的文档的并集
        candidates = candidates.union(list(inverted_idx[word]))
    candidates = list(candidates)

    q_vector = vectorizer.transform([' '.join(seg)])
    # 计算余弦相似度，tfidf用的l2范数，所以分母为1；矩阵乘法
    sim = (X[candidates] * q_vector.T).toarray()

    # 使用优先队列找出top5
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], candidates[cur]))
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    top_alist = [alist[x[1]] for x in pq_rank]  # 返回相似度最高的问题对应的答案，作为TOP5答案

    return top_alist


# TODO: 编写几个测试用例，并输出结果
print(top5results_invidx("Which airport was shut down?"))  # 在问题库中存在，经过对比，返回的首结果正确
print(top5results_invidx("Which airport is closed?"))
print(top5results_invidx("What government blocked aid after Cyclone Nargis?"))
print(top5results_invidx("Which government stopped aid after Hurricane Nargis?"))

# TODO
# 读取每一个单词的嵌入。这个是 D*H的矩阵，这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，那句子向量怎么表达？
# 其中，最简单的方式 句子向量 = 词向量的平均（出现在问句里的）， 如果给定的词没有出现在词典库里，则忽略掉这个词。

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np

# 将Glove转化为word2vec
_ = glove2word2vec('./data/glove.6B.100d.txt', './data/glove2word2vec.6B.100d.txt')
model = KeyedVectors.load_word2vec_format('./data/glove2word2vec.6B.100d.txt')


def docvec_get(seg):
    """
    将分词数据转为句向量。
    seg: 分词后的数据

    return: 句向量
    """
    vector = np.zeros((1, 100))
    size = len(seg)
    for word in seg:
        try:
            vector += model.wv[word]
        except KeyError:
            size -= 1
    try:
        res = vector / size
    except Exception as e:
        return e
    return res


X = np.zeros((len(qlist_seg), 100))
for cur in range(X.shape[0]):
    X[cur] = docvec_get(qlist_seg[cur])

# 计算X每一行的L2范数
Xnorm2 = np.linalg.norm(X, axis=1, keepdims=True)
X = X / Xnorm2


def top5result_emb(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q，转换成句子向量
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    seg = text_preprocessing(input_q)
    candidate = set()
    for word in seg:
        candidate = candidate.union(list(inverted_idx[word]))
    candidate = list(candidate)
    # 将用户输入的seg(input_q)转换为句子向量
    input_vec = docvec_get(seg)
    q_norm = np.linalg.norm(input_vec, axis=1, keepdims=True)
    input_vec = input_vec / q_norm
    sim = np.dot(X[candidate], input_vec.T).tolist()
    pq = PriorityQueue()
    for i in range(len(sim)):
        pq.put([sim[i], candidate[i]])
        if len(pq.queue) > 5:
            pq.get()
    pq_sort = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    print([x[0] for x in pq_sort])
    top_alist = [alist[li[1]] for li in pq_sort]

    return top_alist


print('<--------glove词向量法--------->')
print(top5result_emb("Which airport was shut down?"))  # 在问题库中存在，经过对比，返回的首结果正确
print(top5result_emb("Which airport is closed?"))
print(top5result_emb("What government blocked aid after Cyclone Nargis?"))
print(top5result_emb("Which government stopped aid after Hurricane Nargis?"))

plt.show()
