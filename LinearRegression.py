import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def cost_function(X, y, theta, b):
    m = X.shape[0]
    error = (np.dot(X, theta) + b) - y
    cost = 1/(m) * np.dot(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, b, alpha, iters, batch_size):
    cost_array = np.zeros(iters)
    m = y.size
    batches = [(X[i:i + batch_size], y[i:i + batch_size])
                   for i in range(0, len(X), batch_size)]
    cost, error = None, None
    for i in range(iters):
        for x_, y_ in batches:
            cost, error = cost_function(x_, y_, theta, b)
            theta = theta - (alpha * (1/x_.shape[0]) * 2 * np.dot(x_.T, error))
            b = b - (alpha * (1/x_.shape[0]) * 2 * np.sum(error))
        if (i+1)%100 == 0:
            print("Iteration No: {}\t Cost : {}".format(i, cost))
        cost_array[i] = cost
    return theta, b, cost_array

def predict(X, theta, b):
    y_pred = np.dot(X, theta) + b
    return y_pred



def main():
    data = pd.read_csv('4.csv')
    data.dropna(axis=0, how='any')
    data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y%m%d").astype(int)

    X = np.array(data.drop(['id', 'price'], axis=1).copy())
    y = np.array(data['price'].copy())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    # 80% train, 20% test
    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test shape: {}\n".format(y_test.shape))

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
 
    X_train = np.c_[np.ones(X_train.shape[0]), X_train] 
    X_test = np.c_[np.ones(X_test.shape[0]), X_test] 

    alpha = 0.01
    b = 0
    iterations = 1000
    batch_size = 1000

    theta = np.zeros(X_train.shape[1])
    initial_cost, _ = cost_function(X_train, y_train, theta, b)

    print('With initial theta values of {0}, cost error on train set is {1}'.format(theta, initial_cost))

    theta, b, cost_num = gradient_descent(X_train, y_train, theta, b, alpha, iterations, batch_size)
    
    final_cost, _ = cost_function(X_train, y_train, theta, b)

    print('Weights:\n{0}\nCost error on train set is: {1}'.format(theta, final_cost))


    y_pred = predict(X_test, theta, b)
    print('r2 score :' ,r2_score(y_test, y_pred))
    print('Mean squared error on test set:', mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    main()