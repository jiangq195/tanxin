# conding :utf-8
from sklearn.linear_model import LogisticRegression
import numpy as np

x_train = np.array([[1, 2, 3],
                    [1, 3, 4],
                    [2, 1, 2],
                    [4, 5, 6],
                    [3, 5, 3],
                    [1, 7, 2]])

y_train = np.array([3, 3, 3, 2, 2, 2])

x_test = np.array([[2, 2, 2],
                   [3, 2, 6],
                   [1, 7, 4]])

clf = LogisticRegression()
clf.fit(x_train, y_train)

# 返回预测标签
print(clf.predict(x_test))

# 返回预测属于某标签的概率

print(clf.predict_proba(x_test))
