from sklearn import linear_model
import numpy as np

reg = linear_model.LinearRegression()

train_data = [[0, 0], [1, 1], [2, 2]]
y_data = [0, 1, 2]
print np.shape(train_data)
print np.shape(y_data)

print reg.fit(train_data, y_data)

print reg.coef_
print reg.intercept_

ridge = linear_model.Ridge(alpha=0.5)
train_data = [[0, 0], [0, 0], [1, 1]]
y_data = [0, .1, 1]
ridge.fit(train_data, y_data)
print ridge.coef_
print ridge.intercept_
print ridge.predict([[1, 1]])

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit([[0, 0], [1, 1]], [0, 1])
print lasso.coef_
print lasso.intercept_
print lasso.predict([[1, 1]])