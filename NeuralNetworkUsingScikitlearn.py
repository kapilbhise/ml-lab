import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

heart_df = pd.read_csv('heart.csv')

X = heart_df.drop(columns=['target'])
y_label = heart_df['target'].values.reshape(X.shape[0], 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")

model = MLPClassifier(hidden_layer_sizes=8, learning_rate_init=0.001, max_iter=100)
model.fit(Xtrain, ytrain)

preds_train = model.predict(Xtrain)
preds_test = model.predict(Xtest)

print("Train accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_train, ytrain), 2) * 100))
print("Test accuracy of sklearn neural network: {}".format(round(accuracy_score(preds_test, ytest), 2) * 100))
