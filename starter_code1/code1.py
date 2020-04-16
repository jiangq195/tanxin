import xlrd
import jieba
import numpy as np
import pandas as pd

def read_xlsx(filename, sheet_index=0, col_index=0):
    workbook = xlrd.open_workbook(filename)
    table = workbook.sheets()[sheet_index]
    col_values = table.col_values(col_index)

    max_len_word = 0
    dic_words = {}
    for word in col_values:
        if word in word_prob:
            dic_words[word] = word_prob[word]
        else:
            dic_words[word] = 0.00001
        len_word = len(word)
        if len_word > max_len_word:
            max_len_word = len_word

    print(len(dic_words))
    print(max_len_word)
    return dic_words, max_len_word





word_prob = {"北京": 0.03, "的": 0.08, "天": 0.005, "气": 0.005, "天气": 0.06, "真": 0.04, "好": 0.05, "真好": 0.04, "啊": 0.01,
             "真好啊": 0.02,
             "今": 0.01, "今天": 0.07, "课程": 0.06, "内容": 0.06, "有": 0.05, "很": 0.03, "很有": 0.04, "意思": 0.06, "有意思": 0.005,
             "课": 0.01,
             "程": 0.005, "经常": 0.08, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.02, "分歧": 0.04, "分": 0.02, "歧": 0.005}

dic_words, max_len_word = read_xlsx('./data/综合类中文词库.xlsx')
print(dic_words['北'])

# print(sum(word_prob.values()))


# def word_segment_naive(input_str):
#     """
#     1. 对于输入字符串做分词，并返回所有可行的分词之后的结果。
#     2. 针对于每一个返回结果，计算句子的概率
#     3. 返回概率最高的最作为最后结果
#
#     input_str: 输入字符串   输入格式：“今天天气好”
#     best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]
#     """
#     generator = jieba.cut(input_str)
#     print('/'.join(generator))



def segment_recur(input_str, max_len_word=16):
    segments = []
    len_input_str = len(input_str)
    if len_input_str == 0:
        return segments
    max_split = min(len_input_str, max_len_word)
    for idx in range(max_split, 0, -1):
        word = input_str[0:idx]

        if word in dic_words:
            seg_ments = segment_recur(input_str[idx:])
            if len(input_str[idx:]) == 0:
                segments.append([word])
            else:
                for item in seg_ments:
                    temp = [word] + item
                    segments.append(temp)

    return segments

def get_best_segment(segments):
    best_score = np.inf
    best_segment = []
    for segment in segments:
        base_score = -sum(np.log([dic_words[word] + EPSILON for word in segment]))
        if base_score < best_score:
            best_score = base_score
            best_segment = segment

    return best_segment, best_score


# def segment_recur(input_str):
    # segments = []
    # len_input_str = len(input_str)
    #
    # if len_input_str == 0:
    #     return segments
    #
    # max_spilt = min(len_input_str, max_len_word) + 1
    # for idx in range(1, max_spilt):
    #
    #     word = input_str[0: idx]
    #
    #     if word in dic_words:
    #         segments_substr = segment_recur(input_str[idx:])
    #
    #         if (len(segments_substr) == 0) & (len(input_str[idx:]) == 0):
    #             segments.append([word])
    #         else:
    #             for seg in segments_substr:
    #                 seg = [word] + seg
    #                 segments.append(seg)
    #
    # return segments

def create_graph(input_str):
    list_input = list(input_str)
    len_list_input = len(list_input)

    word_list = []
    edge_list = []
    for start_idx in range(len_list_input):
        max_len_split = min(len(list_input[start_idx:]), max_len_word)
        for end_idx in range(1, max_len_split + 1):
            word = input_str[start_idx: start_idx+end_idx]
            if word in dic_words:
                word_list.append(word)
                edge_list.append([start_idx, start_idx+end_idx])

    graph = pd.DataFrame(data=np.zeros(shape=(len_list_input, len_list_input)),
                         index=list(range(len_list_input)),
                         columns=list(range(1, len_list_input+1)))

    for idx, edge in enumerate(edge_list):
        word = word_list[idx]
        graph.loc[edge[0], edge[1]] = dic_words[word]

    graph = - np.log(graph + EPSILON)

    return graph

def word_segment_viterbi(input_str):
    """
    1. 基于输入字符串，词典，以及给定的unigram概率来创建DAG(有向图）。
    2. 编写维特比算法来寻找最优的PATH
    3. 返回分词结果

    input_str: 输入字符串   输入格式：“今天天气好”
    best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]
    """

    # TODO: 第一步：根据词典，输入的句子，以及给定的unigram概率来创建带权重的有向图（Directed Graph） 参考：课程内容
    #      有向图的每一条边是一个单词的概率（只要存在于词典里的都可以作为一个合法的单词），这些概率在 word_prob，如果不在word_prob里的单词但在
    #      词典里存在的，统一用概率值0.00001。
    #      注意：思考用什么方式来存储这种有向图比较合适？ 不一定有只有一种方式来存储这种结构。

    graph = create_graph(input_str)

    # TODO： 第二步： 利用维特比算法来找出最好的PATH， 这个PATH是P(sentence)最大或者 -log P(sentence)最小的PATH。
    #              hint: 思考为什么不用相乘: p(w1)p(w2)...而是使用negative log sum:  -log(w1)-log(w2)-...
    len_input_str = len(input_str)

    possible_path_value = pd.Series(
        data=np.zeros(shape=(len_input_str+1,)),
        index=list(range(len_input_str+1))
    )

    possible_path_index = pd.Series(
        data=np.zeros(shape=(len_input_str+1,), dtype=np.int),
        index=list(range(len_input_str+1))
    )

    for idx_end in range(1, len_input_str+1):
        possible_path_temp = np.zeros(shape=(idx_end,))
        for idx_start in range(idx_end):
            possible_path_temp[idx_start] = possible_path_value[idx_start] + graph.loc[idx_start, idx_end]
        possible_path_value[idx_end] = np.min(possible_path_temp)
        possible_path_index[idx_end] = np.argmin(possible_path_temp)

    best_segment = []
    index = len_input_str
    while index > 0:
        best_segment.append(input_str[possible_path_index[index]:index])
        index = possible_path_index[index]

    best_segment.reverse()

    return best_segment




# def word_segment_viterbi(input_str):
#     """
#     1. 基于输入字符串，词典，以及给定的unigram概率来创建DAG(有向图）。
#     2. 编写维特比算法来寻找最优的PATH
#     3. 返回分词结果
#
#     input_str: 输入字符串   输入格式：“今天天气好”
#     best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]
#     """
#
#     # TODO: 第一步：根据词典，输入的句子，以及给定的unigram概率来创建带权重的有向图（Directed Graph） 参考：课程内容
#     #      有向图的每一条边是一个单词的概率（只要存在于词典里的都可以作为一个合法的单词），这些概率在 word_prob，如果不在word_prob里的单词但在
#     #      词典里存在的，统一用概率值0.00001。
#     #      注意：思考用什么方式来存储这种有向图比较合适？ 不一定有只有一种方式来存储这种结构。
#
#     graph = create_graph(input_str)
#
#     # TODO： 第二步： 利用维特比算法来找出最好的PATH， 这个PATH是P(sentence)最大或者 -log P(sentence)最小的PATH。
#     #              hint: 思考为什么不用相乘: p(w1)p(w2)...而是使用negative log sum:  -log(w1)-log(w2)-...
#     num_chars = len(input_str)
#     optimal_path_distance = pd.Series(
#         data=np.zeros(shape=(num_chars + 1)),
#         index=list(range(num_chars + 1))
#     )
#
#     optimal_path = pd.Series(
#         data=np.zeros(shape=(num_chars + 1), dtype=np.int),
#         index=list(range(num_chars + 1))
#     )
#
#     for idx_end in range(1, num_chars + 1):
#         possible_path_distance = np.zeros(shape=(idx_end,))
#         for idx_start in range(idx_end):
#             possible_path_distance[idx_start] = optimal_path_distance[idx_start] + graph.loc[idx_start, idx_end]
#         optimal_path_distance[idx_end] = np.min(possible_path_distance)
#         optimal_path[idx_end] = np.argmin(possible_path_distance)
#
#     # TODO:第三步：根据最好的PATH，返回最好的切分
#     best_segment = []
#     idx = num_chars
#
#     while idx > 0:
#         best_segment.append(input_str[optimal_path[idx]: idx])
#         idx = optimal_path[idx]
#
#     best_segment.reverse()
#     return best_segment




# # 测试
EPSILON = 1e-10
segments =segment_recur("北京的天气真好啊")
print(segments)
# print(get_best_segment(segments))
# print(create_graph("北京的天气真好啊"))
# print(word_segment_viterbi("北京的天气真好啊"))
# print(word_segment_naive("北京的天气真好啊"))
# print(word_segment_naive("今天的课程内容很有意思"))
# print(word_segment_naive("经常有意见分歧"))
