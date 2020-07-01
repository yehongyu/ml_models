from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

## load data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print type(iris)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print type(X_train), X_train.shape
print type(X_test), X_test.shape
print type(y_train), y_train.shape
print type(y_test), y_test.shape

## train data
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

## predict
predict_res = knn.predict(X_test)

print(predict_res)
print(y_test)

size = predict_res.shape[0]
right = 0
for i in range(size):
    if predict_res[i] == y_test[i]:
        right += 1
print right, size, right/float(size)

