import xlrd
import jieba
import numpy as np

def read_xlsx(filename, sheet_index=0, col_index=0):
    workbook = xlrd.open_workbook(filename)
    table = workbook.sheets()[sheet_index]
    col_values = table.col_values(col_index)

    max_len_word = 0
    dic_words = {}
    for word in col_values:
        dic_words[word] = 0.00001
        len_word = len(word)
        if len_word > max_len_word:
            max_len_word = len_word

    print(len(dic_words))
    print(max_len_word)
    return dic_words, max_len_word


dic_words, max_len_word = read_xlsx('./data/综合类中文词库.xlsx')
print(dic_words['北'])


word_prob = {"北京": 0.03, "的": 0.08, "天": 0.005, "气": 0.005, "天气": 0.06, "真": 0.04, "好": 0.05, "真好": 0.04, "啊": 0.01,
             "真好啊": 0.02,
             "今": 0.01, "今天": 0.07, "课程": 0.06, "内容": 0.06, "有": 0.05, "很": 0.03, "很有": 0.04, "意思": 0.06, "有意思": 0.005,
             "课": 0.01,
             "程": 0.005, "经常": 0.08, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.02, "分歧": 0.04, "分": 0.02, "歧": 0.005}

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
            if (len(seg_ments) == 0) & (len(input_str[max_split:]) == 0):
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
        base_score = -sum(np.log([word_prob[word] if word in word_prob else dic_words[word] for word in segment]))
        if base_score < best_score:
            best_score = base_score
            best_segment = segment

    return best_segment, best_score


# def segment_recur(input_str):
#     segments = []
#     len_input_str = len(input_str)
#
#     if len_input_str == 0:
#         return segments
#
#     max_spilt = min(len_input_str, max_len_word) + 1
#     for idx in range(1, max_spilt):
#
#         word = input_str[0: idx]
#
#         if word in dic_words:
#             segments_substr = segment_recur(input_str[idx:])
#
#             if (len(segments_substr) == 0) & (len(input_str[idx:]) == 0):
#                 segments.append([word])
#             else:
#                 for seg in segments_substr:
#                     seg = [word] + seg
#                     segments.append(seg)
#
#     return segments

# # 测试
segments =segment_recur("北京的天气真好啊")
print(segments)
print(get_best_segment(segments))
# print(word_segment_naive("北京的天气真好啊"))
# print(word_segment_naive("今天的课程内容很有意思"))
# print(word_segment_naive("经常有意见分歧"))
