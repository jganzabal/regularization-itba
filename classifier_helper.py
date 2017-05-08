import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from IPython import display

def plot_it(X,y, s = 10, colors = ['r','b']):
    plt.scatter(*X[y==1].T, color=colors[0], marker='o',s = s)
    plt.scatter(*X[y==0].T, color=colors[1], marker='x', s = s)

import math
def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12):
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    return X_mat

def plot_classifier(X, y, predict, degree, N = 500):
    # create a mesh to plot in
    plt.figure(figsize=(8,8))
    x_min, x_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    y_min, y_max = X[:, 2].min() - 0.1, X[:, 2].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
                             np.linspace(y_min, y_max, N))
    polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
    Z = predict(polynomial_set)
    plot_it(X[:,1:3], y, s= 20)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_region(X, y, degree = 12):
    X_mat = get_polynimial_set(X, degree = degree)
    C1 = 100000000
    C2 = 1
    clf = LogisticRegression(C=C1, fit_intercept=False, max_iter=1000000000)
    clf.fit(X_mat,y)
    plot_classifier(X_mat, y, clf.predict, degree)
    print(clf.score(X_mat,y))