import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report



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
        return [self.majority.get(x, 'o') for x in X]


words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()

pred = cross_val_predict(estimator=MajorityVotingTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report)
