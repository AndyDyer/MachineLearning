# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:58:38 2016

@author: andrew

https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch02/ch02.ipynb
"""

import numpy as np
np.set_printoptions(threshold = np.inf)
import csv as csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from numpy.random import seed

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.35, 1, 0)

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


f = open('train.csv', 'rt')
data = []
try:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
finally:
    f.close()
    
#data cleaning
mynumpy = np.array(data)
mynumpy = np.delete(mynumpy,0,0) #removes headers
mynumpy = np.delete(mynumpy,0,1) #remove useless cols 
mynumpy = np.delete(mynumpy,10,1)
mynumpy = np.delete(mynumpy,8,1)
mynumpy = np.delete(mynumpy,7,1)
mynumpy = np.delete(mynumpy,7,1)
mynumpy = np.delete(mynumpy,2,1)
mynumpy = np.delete(mynumpy,1,1)
#mynumpy = np.delete(mynumpy,4,1)
mynumpy = np.delete(mynumpy,1,1)
mynumpy = np.delete(mynumpy,1,1)

mynumpy[mynumpy==''] = '1'
mynumpy[mynumpy=='male'] = '1'
mynumpy[mynumpy=='female'] = '0'
mynumpy = mynumpy.astype(float)
#mynumpy = np.around(mynumpy)
mynumpy.astype(float)

#data Standard
#X_std = np.copy(X)
#print (mynumpy)

# DATA SPLIT

SplitArrays = np.array_split(mynumpy,5)

TestX = SplitArrays[0]
trainNP = np.concatenate([SplitArrays[1],SplitArrays[2],SplitArrays[3],SplitArrays[4]]).astype(int)

TestY = TestX[:,0].astype(int)
TestX = np.delete(TestX,0,1).astype(float)

y = trainNP[:,0].astype(int)
X = np.delete(trainNP,0,1)
#print (X)





def SGD():
    
    
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X, y)

    plot_decision_regions(X, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    #plt.savefig('./adaline_4.png', dpi=300)
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.tight_layout()
    # plt.savefig('./adaline_5.png', dpi=300)
    plt.show()
    
    Answers = ada.predict(TestX)
    print ("Adaline Accuracy:")
    print (accuracy_score(TestY,Answers))






SGD()
