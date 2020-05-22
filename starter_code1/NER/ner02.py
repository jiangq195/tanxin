import re

import pandas as pd
import fool
from copy import copy
from starter_code1.NER.ner01 import *

test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)
# print(test_data.head())

test_data['ner'] = None
ner_id = 1001
ner_dict_new = {}  # 存储所有实体
ner_dict_reverse_new = {}  # 储存所有实体

for i in range(len(test_data)):
    sentence = copy(test_data.iloc[i, 1])
    # TODO: 调用fool积极性实体识别，得到words和ners结果
    words, ners = fool.analysis(sentence)
    # print(words)
    # print(ners)

    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == 'company' or ner_type == 'person':
            # ner_dict_new
            lst = main_extract(ner_name, stop_word, d_4_delete, d_city_province)
            company_main_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
            if company_main_name not in ner_dict_new:
                ner_dict_new[company_main_name] = ner_id
                ner_dict_reverse_new[ner_id] = company_main_name
                ner_id += 1

            sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end:]

    test_data.iloc[i, -1] = sentence


X_test = test_data[['ner']]

# 处理train数据,利用开源工具进行实体识别和并使用实体统一函数储存实体
train_data = pd.read_csv('../data/info_extract/train_data.csv', encoding='gb2312', header=0)
train_data['ner'] = None

for i in range(len(train_data)):
    # 判断正负样本
    if train_data.iloc[i, :]['member1'] == '0' and train_data.iloc[i, :]['member2'] == '0':
        sentence = copy(train_data.iloc[i, 1])
        # TODO:调用fool进行实体识别，得到wods和ners结果
        words, ners = fool.analysis(sentence)
        ners[0].sort(key=lambda x: x[0], reverse=True)
        for start, end, ner_type, ner_name in ners[0]:
            # TODO:调用实体统一函数，储存统一后的实体
            # 并自增ner_id
            if ner_type == 'company' or ner_type == 'person':
                company_main_name = ''.join(
                    main_extract(ner_name, stop_word, d_4_delete, d_city_province))  # 提取公司主体名称
                if company_main_name not in ner_dict_new:
                    ner_dict_new[company_main_name] = ner_id
                    ner_dict_reverse_new[ner_id] = company_main_name
                    ner_id += 1

                # 在句子中用编号替换实体名
                sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end:]

        train_data.iloc[i, -1] = sentence
    else:
        # 将训练集中正样本已经标注的实体也使用编码替换
        sentence = copy(train_data.iloc[i, :])['sentence']
        for company_main_name in [train_data.iloc[i, :]['member1'], train_data.iloc[i, :]['member2']]:
            # TODO:调用实体统一函数，储存统一后的实体
            # 并自增ner_id
            company_main_name = ''.join(
                main_extract(company_main_name, stop_word, d_4_delete, d_city_province))  # 提取公司主体名称
            if company_main_name not in ner_dict_new:
                ner_dict_new[company_main_name] = ner_id
                ner_dict_reverse_new[ner_id] = company_main_name
                ner_id += 1

            # 在句子中用编号替换实体名
            sentence = re.sub(company_main_name, ' ner_%s_ ' % (str(ner_dict_new[company_main_name])), sentence)

        train_data.iloc[i, -1] = sentence

y = train_data.loc[:, ['tag']]
train_num = len(train_data)
X_train = train_data[['ner']]

# 将train和test放在一起提取特征
X = pd.concat([X_train, X_test])
X.to_csv('./x.csv', index=False)
print(X)
