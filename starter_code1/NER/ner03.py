from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
from pyltp import Segmentor, Parser, Postagger
import networkx as nx
import pylab

# 实体符号加入分词词典
with open('../data/user_dict.txt', 'w', encoding='utf-8') as fw:
    for v in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        fw.write(v + '号企业 ni\n')

# 初始化实例
segmentor = Segmentor()
# 加载模型,加载自定义词典
segmentor.load_with_lexicon('F:\ltp_data\cws.model', '../data/user_dict.txt')
# segmentor.load_with_lexicon('E:\ltp_data\cws.model', '../data/user_dict.txt')

# 加载停用词
fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
stop_word = fr.readlines()
stop_word = [re.sub(r'(\r|\n)*', '', stop_word[i]) for i in range(len(stop_word))]

# 分词
f = lambda x: ' '.join(
    [word for word in segmentor.segment(x) if word not in stop_word and not re.findall(r'(ner|\d\d\d\d)', word)])

X = pd.read_csv('./x.csv')
corpus = X['ner'].map(f).tolist()

# print(corpus)

tfidf = TfidfVectorizer()
tfidf.fit(corpus)
tfidf_train = tfidf.transform(corpus)

tfidf_feature = pd.DataFrame(tfidf_train.toarray())

postagger = Postagger()  # 初始化实例
postagger.load_with_lexicon('F:\ltp_data\pos.model', '../data/user_dict.txt')  # 加载模型
# postagger.load_with_lexicon('E:\ltp_data\pos.model', '../data/user_dict.txt')  # 加载模型


def parse(s):
    """
    对语句进行句法分析，并返回句法结果
    parse_result：依存句法解析结果
    source：企业实体的词序号
    target：另一个企业实体的词序号
    keyword_pos：关键词词序号列表
    source_dep：企业实体依存句法类型
    target_dep：另一个企业实体依存句法类型
    """
    tmp_ner_dict = {}
    num_lst = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    # 将公司代码替换为特殊称谓，保证分词词性正确
    for i, ner in enumerate(list(set(re.findall(r'(ner\_\d\d\d\d\_)', s)))):
        try:
            tmp_ner_dict[num_lst[i] + '号企业'] = ner
        except IndexError:
            return None, None, None, None, None, None
        s = s.replace(ner, num_lst[i] + '号企业')
    words = segmentor.segment(s)
    tags = postagger.postag(words)
    parser = Parser()  # 初始化实例
    parser.load('F:\ltp_data\parser.model')  # 加载模型
    # parser.load('E:\ltp_data\parser.model')  # 加载模型
    arcs = parser.parse(words, tags)  # 句法分析
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))

    # 句法分析结果输出
    # parse_result = pd.DataFrame([[a, b, c, d] for a, b, c, d in zip(list(words), list(tags), arcs_lst[0], arcs_lst[1])],
    #                             index=range(1, len(words) + 1))
    parse_result = pd.DataFrame(list(map(list, zip(list(words), list(tags), arcs_lst[0], arcs_lst[1]))),
                                index=range(1, len(words) + 1))
    parser.release()

    # 能找到两个企业以上才返回结果，目前简化，只考虑两家企业关系
    try:
        source = list(words).index('一号企业') + 1
        target = list(words).index('二号企业') + 1
        source_dep = arcs_lst[1][source - 1]
        target_dep = arcs_lst[1][target - 1]

    except:
        return None, None, None, None, None, None

    # 找投资关系关键词
    key_words = ["收购", "竞拍", "转让", "扩张", "并购", "注资", "整合", "并入", "竞购", "竞买", "支付", "收购价", "收购价格", "承购", "购得", "购进",
                 "购入", "买进", "买入", "赎买", "购销", "议购", "函购", "函售", "抛售", "售卖", "销售", "转售"]

    keyword_pos = [list(words).index(w) + 1 if w in list(words) else -1 for w in key_words]

    return parse_result, source, target, keyword_pos, source_dep, target_dep


def shortest_path(arcs_ret, source, target):
    """
    求出两个词最短依存句法路径，不存在路径返回-1
    """
    G = nx.DiGraph()
    # 为证网络添加节点
    for i in list(arcs_ret.index):
        G.add_node(i)
    # 在网络中添加带权重的边
    for i in list(arcs_ret.index):
        G.add_edges_from([(arcs_ret.iloc[i - 1, 2], i)])
        G.add_edges_from([(i, arcs_ret.iloc[i - 1, 2])])

    try:
        distance = nx.shortest_path_length(G, source, target)
        return distance

    except:
        return -1


def get_parse_feature(s):
    """综合上述函数汇总句法分析特征"""
    parse_result, source, target, keyword_pos, source_dep, target_dep = parse(s)
    if parse_result is None:
        return [-1] * 59
    features = []
    features.append(shortest_path(parse_result, source, target))
    keyword_feature = []
    for p in keyword_pos:
        if p == -1:
            keyword_feature.append(-1)
            keyword_feature.append(-1)
        else:
            keyword_feature.append(shortest_path(parse_result, source, p))
            keyword_feature.append(shortest_path(parse_result, target, p))
    features.extend(keyword_feature)
    features.extend([source_dep, target_dep])
    return features


# 对所有样本统一派生特征
f = lambda x: get_parse_feature(x)
parse_feature = X['ner'].map(f)

whole_feature = []
for i in range(len(parse_feature)):
    whole_feature.append(list(tfidf_feature.iloc[i, :]) + parse_feature[i])

whole_feature = pd.DataFrame(whole_feature)




# sentence = '想飞上天和太阳肩并肩'
# words = segmentor.segment(sentence)
# print('/'.join(words))
