import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
np.set_printoptions(threshold = 5)

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

ar_train, ar_test = train_test_split(iris_X, test_size=0.30, random_state=42)
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=32)

iris_train_X = np.array(ar_train)
iris_dev_X = np.array(ar_dev)
iris_test_X = np.array(ar_test)

ar_train, ar_test = train_test_split(iris_Y, test_size=0.30, random_state=42)
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=32)

iris_train_Y = np.array(ar_train)
iris_dev_Y = np.array(ar_dev)
iris_test_Y = np.array(ar_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(iris_train_X, iris_train_Y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(iris_test_X) - iris_test_Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(iris_test_X, iris_test_Y))
