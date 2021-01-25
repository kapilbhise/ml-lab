import pandas as pd 

# read csv file
df=pd.read_csv("https://github.com/kapilbhise/ml-lab/blob/main/g.csv",header=0)
dataset = df.values.tolist()
print(df)
dataset = df.values.tolist()
# print("\nThe dataset is :\n",dataset)

#initialize the specific hypothesis
s=dataset[0][0:-1]
print("The initial value of s is :\n",s)

#initialize the general hypothesis
g=[['?' for i in range(len(s))] for j in range(len(s))]
print("The initial value of g is :\n",g)

for row in dataset:
    if row[-1]=="Yes":
        for j in range(len(s)):
            if row[j]!=s[j]:
                s[j]='?'
                g[j][j]='?'
    elif row[-1]=="No":
        for j in range(len(s)):
            if row[j]!=s[j]:
                g[j][j]=s[j]
            else:
                g[j][j]="?"
    print("\nAfter",dataset.index(row)+1,"th insatnce")
        
    print("Specific boundary is :",s)
    print("General boundary is :",g)

print("\nFinal specific hypothesis:\n",s)
print("\nFinal general hypothesis:\n",g)
