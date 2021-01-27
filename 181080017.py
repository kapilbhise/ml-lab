import pandas as pd

df=pd.read_csv("1.csv")
dataset = df.values.tolist()

print("\nThe Given Data Set \n")
print(df)
# print(dataset)

rows=len(dataset)
columns=len(dataset[0])-1
# print(rows,columns)

print("\nThe initial value of hypothesis: ")
# Initialize the hypothesis
hypothesis = ['phi'] * columns
print(hypothesis)

# training with the first instance 
for i in range(0,columns):
        hypothesis[i] = dataset[0][i]

# print(hypothesis)

# training the dataset
for i in range(0,rows):
    if dataset[i][columns]=='Yes':
            for j in range(0,columns):
                if dataset[i][j]!=hypothesis[j]:
                    hypothesis[j]='?'
                else :
                    hypothesis[j]= dataset[i][j] 
    print("After {0} iteration,the hypothesis is :".format(i),hypothesis)


print("\nHypothesis after training :")
print(hypothesis)