import pandas as pd
from pandas import DataFrame 
df_tennis = pd.read_csv('https://github.com/kapilbhise/ml-lab')

def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def list_entropy(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list)  
    num_instances = len(a_list)*1.0  
    # print("\n Number of Instances of the Current Sub Class is {0}:".format(num_instances ))
    probs = [x / num_instances for x in cnt.values()]
    # print("\n Classes:",min(cnt),max(cnt))
    # print(" \n Probabilities of Class {0} is {1}:".format(min(cnt),min(probs)))
    # print(" \n Probabilities of Class {0} is {1}:".format(max(cnt),max(probs)))
    return entropy(probs)
# print("\n  input for entropy calculation is :\n", df_tennis['class'])

total_entropy = list_entropy(df_tennis['class'])
print("Total Entropy of given Data Set:",total_entropy)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    # print("Information Gain Calculation of ",split_attribute_name)
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [list_entropy, lambda x: len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = list_entropy(df[target_attribute_name])
    return old_entropy - new_entropy

for i in range(1,23):
    # print("Info Gain for",df_tennis.columns[i],"is :")
    information_gain(df_tennis,df_tennis.columns[i],"class")

def id3(df, target_attribute_name, attribute_names, default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class
    else:
        default_class = max(cnt.keys()) 
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        
        tree = {best_attr:{}}  
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]        
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

attribute_names = list(df_tennis.columns)
attribute_names.remove('class')

from pprint import pprint
tree = id3(df_tennis,'class',attribute_names)
print("The Resultant Decision Tree is :\n")
print(tree)
pprint(tree)

