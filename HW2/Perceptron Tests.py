# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:11:26 2016

@author: andrew
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



class Perceptron (object):
    def __init__(self,eta = 0.01, n_iter = 100, shuffle=True, random_state=None):
            self.eta = eta
            self.n_iter = n_iter
            
            self.shuffle = shuffle
            if random_state:
                np.random.seed(random_state)
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            if self.shuffle:  # new
                X, y = self._shuffle(X, y)  # new
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def _shuffle(self, X, y):  # new
        """Shuffle training data"""  # new
        r = np.random.permutation(len(y))  # new
        return X[r], y[r]  # new

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

badPoints = np.array([[1,2,1],[0,5,1],[1,1,3],[1,3,4],[0,5,5],[1,1,1],[0,3,4],[1,3,2],[0,6,1],[1,2,1]])
badY = badPoints[:,0]
badX = np.delete(badPoints,0,1)

goodPoints = np.array([[1,1,2],[1,2,3],[1,3,4],[1,4,5],[1,1,3],[0,2,1],[0,3,1],[0,3,2],[0,5,1],[0,4,2]])
goodY = goodPoints[:,0]
goodX = np.delete(goodPoints,0,1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(badX,badY)
plot_decision_regions(badX, badY, classifier=ppn)
plt.xlabel('Bad Foos')
plt.ylabel('Bad Bars')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./perceptron_badData.png', dpi=300)
plt.show()

ppn2 = Perceptron(eta=0.1, n_iter=10000)
ppn2.fit(goodX,goodY)
plot_decision_regions(goodX, goodY, classifier=ppn2)
plt.xlabel('Good Foos')
plt.ylabel('Good Bars')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./perceptron_goodData.png', dpi=300)
plt.show()



