from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from starter_code1.NER.ner02 import X
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

# 加载停用词
fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
stop_word = fr.readlines()
stop_word = [re.sub(r'(\r|\n)*', '', stop_word[i]) for i in range(len(stop_word))]

# 分词
f = lambda x: ' '.join(
    [word for word in segmentor.segment(x) if word not in stop_word and not re.findall(r'(ner|\d\d\d\d)', word)])

corpus = X['ner'].map(f).tolist()

print(corpus)

tfidf = TfidfVectorizer()
tfidf.fit(corpus)
tfidf_train = tfidf.transform(corpus)

tfidf_feature = pd.DataFrame(tfidf_train.toarray())

postagger = Postagger()  # 初始化实例
postagger.load_with_lexicon('F:\ltp_data\pos.model', '../data/user_dict.txt')  # 加载模型


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
            tmp_ner_dict[num_lst[i]+'号企业'] = ner
        except IndexError:
            return None, None, None, None, None, None
        s = s.repace(ner, num_lst[i]+'号企业')
    words = segmentor.segment(s)
    tags = postagger.postag(words)
    parser = Parser()  # 初始化实例
    parser.load('F:\ltp_data\parser.model')  # 加载模型
    arcs = parser.parse(words, tags)  # 句法分析
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))





















# sentence = '想飞上天和太阳肩并肩'
# words = segmentor.segment(sentence)
# print('/'.join(words))
