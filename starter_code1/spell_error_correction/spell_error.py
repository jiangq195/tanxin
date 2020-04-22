from copy import deepcopy
import numpy as np
from nltk.corpus import reuters

# 词典库
vocab = set([line.rstrip() for line in open('../data/vocab.txt')])

# 需要生成所有候选集合
def generate_candidate(word):
    """
    :param word:给定的输入(错误的输入)
    :return: 返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert, 2.delete 3.replace
    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz'

    # candidates = set()
    # len_word = len(word)
    # splits = list(word)
    # # insert
    # for i in range(len_word + 1):
    #     for letter in letters:
    #         new_splits = deepcopy(splits)
    #         new_splits.insert(i, letter)
    #         candidates.add(''.join(new_splits))
    #
    # # delete
    # for i in range(len_word):
    #     new_splits = deepcopy(splits)
    #     new_splits.pop(i)
    #     candidates.add(''.join(new_splits))
    #
    # # replace
    # for i in range(len_word):
    #     for letter in letters:
    #         new_splits = deepcopy(splits)
    #         new_splits[i] = letter
    #         candidates.add(''.join(new_splits))
    splits = [[word[:i], word[i:]] for i in range(len(word) + 1)]
    # insert
    inserts = [L+letter+R for L, R in splits for letter in letters]
    # delete
    deletes = [L+R[1:] for L, R in splits]
    # replace
    replaces = [L+letter+R[1:] for L, R in splits for letter in letters]

    candidate = set(inserts+deletes+replaces)

    # 过滤掉词典库中不存在的单词
    return [word for word in candidate if word in vocab]


# res = generate_candidate('apple')
# print(res)
#
# 读取语料库
categories = reuters.categories()
corpus = reuters.sents(categories=categories)

# 构建语言模型：bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc + ['<p>']
    for i in range(len(doc)):
        # bigram
        term = doc[i]
        bigram = doc[i: i+2]

        if term in term_count:
            term_count[term] += 1
        else:
            term_count[term] = 1
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1



# sklearn有现成的包

# 用户打错的概率统计 - channel probability
channel_prob = {}
for line in open('../data/spell-errors.txt'):
    item = line.split(':')
    correct = item[0].strip()
    mistakes = [item.strip() for item in item[1].strip().split(',')]
    channel_prob[correct] = {}
    for mis in mistakes:
        channel_prob[correct][mis] = 1 / len(mistakes)


# print(channel_prob)

V = len(term_count.keys())

file = open('../data/testdata.txt', 'r')
for line in file:
    items = line.rstrip().split('\t')
    line = ['<s>'] + [spl.rstrip(',|.|?') for spl in items[2].split()] + ['<p>']
    for index in range(1, len(line)-1):
        word = line[index]
        if word not in vocab:
            # 需要替换word成正确的单词
            # Step1:生成所有的(valid)候选集合
            candidates = generate_candidate(word)
            # if candidate == [], 生成编辑距离为2的candidate
            if len(candidates) < 1:
                continue  # 不建议

            probs = []
            # 对于每一个candidate，计算它的score
            # score = p(c) * p(s | c)
            # log(score) = log(p(c)) + log(p(s|c))
            # 返回返回score最大的candidate
            for candi in candidates:
                prob = 0
                # a.计算channel probability
                if candi in channel_prob and word in channel_prob[candi]:
                    prob += np.log(channel_prob[candi][word])
                else:
                    prob += np.log(0.0001)

                # b.计算语言模型概率
                idx = line.index(word)
                if line[idx-1] not in term_count:
                    prob += np.log(1.0 / V)
                elif line[idx-1] + ' ' + candi in bigram_count:
                    prob += np.log((bigram_count[line[idx-1] + ' ' + candi] + 1.0) / (term_count[line[idx-1]] + V))
                else:
                    prob += np.log(1.0/(term_count[line[idx-1]] + V))

                # TODO：也要考虑当前[word, post_word]
                # prob += np.log(bigram概率)
                if word not in term_count:
                    prob += np.log(1.0 / V)
                elif candi + ' ' + line[idx + 1] in bigram_count:
                    prob += np.log((bigram_count[candi + ' ' + line[idx + 1]] + 1.0) / (term_count[candi] + V))
                else:
                    prob += np.log(1.0/(term_count[candi] + V))

                probs.append(prob)
            max_idx = probs.index(max(probs))
            print(word, candidates[max_idx])







