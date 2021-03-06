import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def cost(self,y,y1):
        cost=(-(1/len(y)))*(np.sum(y*np.log(y1)+(1-y)*(np.log(1-y1))))
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        costs=[]

        # gradient descent
        for i in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            loss=self.cost(y,y_predicted)
            cost = np.sum(loss) / X.shape[1]       
            costs.append(loss)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        print(len(costs))
        l = [i+1 for i in range(100000)]
        print(len(l))
        plt.plot(l, costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Iterations vs Cost (Without Regularization)')
        plt.show()

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
df=pd.read_csv("5.csv")
data=np.array(df)

X, y = df.drop("target",axis=1), df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LogisticRegression(learning_rate=0.000015, n_iters=100000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))

