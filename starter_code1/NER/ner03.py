from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# from pyltp

# 实体符号加入分词词典
with open('../data/user_dict.txt', 'w') as fw:
    for v in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        fw.write(v + '号企业 ni\n')

# 初始化实例
segmentor = Segm
