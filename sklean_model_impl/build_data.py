from sklearn import datasets

X, y = datasets.make_regression(
    n_samples=100, n_features=1, n_targets=1, bias=2, noise=2
)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X, y)
plt.show()