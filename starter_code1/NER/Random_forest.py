import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('./ner_dataset.csv', encoding='latin1')
data = data.fillna(method="ffill")  # ffill前值填充，pfill后值填充
print(data.tail(10))

words = list(set(data["Word"].values))  # 获取词典库
n_words = len(words)  # 词典库大小


class MajorityVotingTagger(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """
        :param X: list of words
        :param y: list 0f tags
        :return:
        """
        word2cnt = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in word2cnt:
                if t in word2cnt[x]:
                    word2cnt[x][t] += 1
                else:
                    word2cnt[x][t] = 1
            else:
                word2cnt[x] = {t: 1}

        self.majority = {}
        for k, d in word2cnt.items():
            self.majority[k] = max(d, key=d.get)

    def predict(self, X, y=None):
        """Predict the the tag from memory, If word is unknown, predict 'o'"""
        return [self.majority.get(x, 'O') for x in X]


def get_feature(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(), word.isalpha()])


words = [get_feature(w) for w in data["Word"].values.tolist()]
tags = data["Tag"].values.tolist()

pred = cross_val_predict(RandomForestClassifier(n_estimators=5), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report)


def get_sentences(data):
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                       s["POS"].values.tolist(),
                                                       s["Tag"].values.tolist())]
    sentence_grouped = data.groupby("Sentence #").apply(agg_func)
    return [s for s in sentence_grouped]


sentences = get_sentences(data)

out = []
y = []
mv_tagger = MajorityVotingTagger()
tag_encoder = LabelEncoder()
pos_encoder = LabelEncoder()

words = data["Word"].values.tolist()
pos = data["POS"].values.tolist()
tags = data["Tag"].values.tolist()

mv_tagger.fit(words, tags)
tag_encoder.fit(tags)
pos_encoder.fit(pos)

for sentence in sentences:
    for i in range(len(sentence)):
        w, p, t = sentence[i][0], sentence[i][1], sentence[i][2]

        if i < len(sentence) - 1:
            # 如果不是最后一个单词，则可以用到下文的信息
            mem_tag_r = tag_encoder.transform(mv_tagger.predict([sentence[i + 1][0]]))[0]
            true_pos_r = pos_encoder.transform([sentence[i + 1][1]])[0]
        else:
            mem_tag_r = tag_encoder.transform(['O'])[0]
            true_pos_r = pos_encoder.transform(['.'])[0]

        if i > 0:
            # 如果不是第一个单词，则可以用到上文的信息
            mem_tag_l = tag_encoder.transform(mv_tagger.predict([sentence[i - 1][0]]))[0]
            true_pos_l = pos_encoder.transform([sentence[i - 1][1]])[0]
        else:
            mem_tag_l = tag_encoder.transform(['O'])[0]
            true_pos_l = pos_encoder.transform(['.'])[0]

        out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                             tag_encoder.transform(mv_tagger.predict([sentence[i][0]])),
                             pos_encoder.transform([p])[0], mem_tag_r, true_pos_r, mem_tag_l, true_pos_l]))

        y.append(t)


pred = cross_val_predict(RandomForestClassifier(n_estimators=5), X=out, y=y, cv=5)
report = classification_report(y_pred=pred, y_true=y)
print(report)



