import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Logisticregression:
    
    def _init_(self, learning_rate, n_iters, Lambda, regularization):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.Lambda = Lambda
        self.regularization = regularization
        
    def sigmoid(self, z):
        sig = 1 / (1 + np.exp(-z))
        return sig
    
    def cost_with_regularization(self, y, y1, theta):
        cost = ((-(1/len(y)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))) + ((self.Lambda/2) * np.sum(np.dot(theta, theta.T))))
        return cost
    
    def fit(self, x, y, theta):
        temp_theta = theta
        costList = []
        m = len(y)
        for _ in range(self.n_iters):
            y_pred = self.sigmoid(x.dot(temp_theta))
            loss = 0
            if self.regularization:
                loss = self.cost_with_regularization(y, y_pred, temp_theta)
                costList.append(loss)
                gradient = np.dot(x.T, (y_pred - y)) + self.Lambda*temp_theta
                temp_theta = temp_theta - ((self.lr/m) * gradient)
        return temp_theta, costList
    
    def predict(self, X, theta):
        y_test_pred = self.sigmoid(X.dot(theta))
        predictions = np.where(y_test_pred >= 0.5, 1, 0)
        return predictions
    
    def find_accuracy(self, y, y_pred):
        correct = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                correct += 1
        return (correct/len(y))*100

y = pd.read_csv("5.csv")["target"]
X = pd.read_csv("5.csv")
X.drop(["target"], axis=1, inplace=True)
X = X.astype(np.float32)
y = y.astype(np.float32)
X = np.array((X - np.mean(X))/np.std(X))
X = np.array(X)

onearr = np.ones((len(y), 1))
X = np.hstack((onearr, X))
theta = np.zeros((len(X[0]), 1))
y = y.values.reshape((len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)

# Logistic regression without scikit learn

Lr_with_reg = Logisticregression(learning_rate=0.01, n_iters=1000, Lambda=0.1, regularization=True)


final_theta_2, hist_loss_2 = Lr_with_reg.fit(X_train, y_train, theta)


y_pred_train_2 = Lr_with_reg.predict(X_train, final_theta_2)
y_pred_test_2 = Lr_with_reg.predict(X_test, final_theta_2)


train_accuracy_with_reg = Lr_with_reg.find_accuracy(y_pred_train_2, y_train)
test_accuracy_with_reg = Lr_with_reg.find_accuracy(y_pred_test_2, y_test)

print('Training accuracy with regularization : ', train_accuracy_with_reg)

print('Test accuracy with regularization : ', test_accuracy_with_reg)


l = [i+1 for i in range(1000)]
plt.plot(l, hist_loss_2, 'b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Iterations vs Cost (With Regularization)')
plt.show()
