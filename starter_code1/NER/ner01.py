import jieba
import jieba.posseg as pseg
import re
import datetime


# 从输入的公司名提取主体
def main_extract(input_str, stop_word, d_4_delete, d_city_province):
    # 开始分词并处理
    seg = jieba.cut(input_str)
    seg_list = remove_word(seg, stop_word, d_4_delete)
    seg_list = city_prov_ahead(seg_list, d_city_province)
    return seg_list


# 实现公司名称中地名提前
def city_prov_ahead(seg, d_city_province):
    city_prov_list = []
    seg_list = []
    for word in seg:
        if word in d_city_province:
            city_prov_list.append(word)
        else:
            seg_list.append(word)

    return city_prov_list + seg_list


# 替换特殊字符
def remove_word(seg, stop_word, d_4_delete):
    # TODO
    seg_list = []
    for word in seg:
        if (word not in stop_word) and (word not in d_4_delete):
            seg_list.append(word)

    return seg_list


# 初始化， 加载词典
def my_initial():
    fr1 = open(r"../data/dict/co_City_Dim.txt", encoding='utf-8')
    fr2 = open(r"../data/dict/co_Province_Dim.txt", encoding='utf-8')
    fr3 = open(r"../data/dict/company_business_scope.txt", encoding='utf-8')
    fr4 = open(r"../data/dict/company_suffix.txt", encoding='utf-8')
    # 城市名
    lines1 = fr1.readlines()
    d_4_delete = []
    d_city_province = [re.sub(r'(\r|\n)*', '', line) for line in lines1]
    # 省份名
    lines2 = fr2.readlines()
    l2_temp = [re.sub(r'(\r|\n)*', '', line) for line in lines2]
    d_city_province.extend(l2_temp)
    # 公司后缀
    lines3 = fr3.readlines()
    l3_temp = [re.sub(r'(\r|\n)*', '', line) for line in lines3]
    # d_4_delete.extend(l3_temp)
    lines4 = fr4.readlines()
    l4_temp = [re.sub(r'(\r|\n)*', '', line) for line in lines4]
    d_4_delete.extend(l4_temp)
    # get_stop_word
    fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
    stop_word = fr.readlines()
    stop_word_after = [re.sub(r'(\r|\n)*', '', word) for word in stop_word]
    stop_word_after[-1] = stop_word[-1].rstrip('.')
    stop_word = stop_word_after
    return d_4_delete, stop_word, d_city_province


d_4_delete, stop_word, d_city_province = my_initial()
# company_name = "河北银行股份有限公司"
# lst = main_extract(company_name, stop_word, d_4_delete, d_city_province)
# company_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
# print(company_name)
