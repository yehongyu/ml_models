from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split

train_data, train_target = load(filename)
train_x, test_x, train_y, test_y = train_test_split(train_data,
            train_target, test_size=0.2, random_state=27)

## start svm
clf = svm.SVC(C=5.0)
clf.fit(train_x, train_y)
## 预测，返回概率
predict_prob_y = clf.predict_proba(test_x)

test_auc = metrics.roc_auc_score(test_y, predict_prob_y)
print test_auc