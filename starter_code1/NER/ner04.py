# <---------实体消歧------------>
import jieba
import pandas as pd
from copy import copy
import fool

# 找出test_data.csv中前25条样本中所有的任务名称，以及人物所在的文档的上下文内容
test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)

# 存储人物以及上下文信息(key为人物ID，value为人物名称、人物上下文内容)
person_name = {}

# 观察上下文的窗口大小
window = 10

# 遍历前25条样本
for i in range(25):
    sentence = copy(test_data.iloc[i, i])
    words, ners = fool.analysis(sentence)
    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == 'person':
            # TODO:提取上下文
            person_name[ner_name] = person_name.get(ner_name) + sentence[max(0, idx-window)]
