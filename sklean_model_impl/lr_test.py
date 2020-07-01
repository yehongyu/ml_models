# coding=utf-8
from sklearn import datasets
from sklearn.linear_model import LinearRegression

## import data
load_data = datasets.load_boston()
data_X = load_data.data
data_y = load_data.target
print(data_X.shape)

## train
model = LinearRegression()
model.fit(data_X, data_y)

## predict
model.predict(data_X[:4,:])

## print 系数
print model.coef_
## print 截距 bias
print model.intercept_

print(model.score(data_X, data_y))
