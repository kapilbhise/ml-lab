import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('5.csv')

df.head()

new_df=df.drop(["target"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_df, df["target"], test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)

x= model.score(X_test,y_test)
print("Accuracy with scikit learn :",x)
