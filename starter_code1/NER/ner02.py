import pandas as pd
import fool
from copy import copy

train_data = pd.read_csv('../data/info_extract/train_data.csv', encoding='gb2312', header=0)
# print(train_data.head())

test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)
print(test_data.head())

test_data['ner'] = None
ner_id = 1001
ner_dict_new = {}  # 存储所有实体
ner_dict_reverse_new = {}  # 储存所有实体

for i in range(len(test_data)):
    sentence = copy(test_data.iloc[i, 1])
    # TODO: 调用fool积极性实体识别，得到words和ners结果
    words, ners = fool.analysis(sentence)
    print(words)
    print(ners)

    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type=='company' or ner_type=='person':
            
