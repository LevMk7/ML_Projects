import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 8.0)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

import sympy as sp
sp.init_printing(use_latex='mathjax')



"""
Generating and plotting the data if it is possible
"""

plt.rcParams['figure.figsize'] = (7.0, 5.0)
np.random.seed(0)
train_X = np.random.randn(300, 2)
train_Y = np.logical_xor(train_X[:, 0] > 0, train_X[:, 1] > 0) * 1

plt.scatter(train_X[:, 0], train_X[:, 1], s=30, c=train_Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.axis([-3, 3, -3, 3])
plt.show() # Make sure you've installed PyQt5 library

"""
Implementation
"""

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))

def plot_svm(model, X, Y, ax = None):
    """
    Plots the decision function for each datapoint on the grid
    """
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                   origin='lower', cmap=plt.cm.coolwarm)
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                    edgecolors='k')
        plt.axis([-3, 3, -3, 3])
        plt.show()
    else:
        ax.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.coolwarm)
        contours = ax.contour(xx, yy, Z, levels=[0], linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                    edgecolors='k')
        ax.axis([-3, 3, -3, 3])


"""
Testing Model with different params
"""

plt.rcParams['figure.figsize'] = (7.0, 5.0)
svm_model = SVC(kernel="poly", degree=3, coef0=5)
svm_model.fit(train_X, train_Y)

print(f"""degree = {3},
accuracy : {accuracy_score(train_Y,svm_model.predict(train_X))*100} %""")
plot_svm(svm_model, train_X, train_Y)

"""
Output: degree = 3,
        accuracy : 98.66666666666667 %
"""

svm_model = SVC(kernel="rbf", gamma=10)
svm_model.fit(train_X, train_Y)

print(f"""gamma = {3},
accuracy : {accuracy_score(train_Y,svm_model.predict(train_X))*100} %""")
plot_svm(svm_model, train_X, train_Y)

"""
Output: gamma = 10,
        accuracy : 100.0 %
"""