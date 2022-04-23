import numpy
import matplotlib.pyplot as plt

import matplotlib.image as pltimg
import pandas
import sklearn.tree
import sklearn
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier

table = pandas.read_csv("pakistan.csv")

columns=table.columns[1:len(table.columns)]
features=[]
for i in range(len(columns)):
    features.append(columns[i])

X=table[features]
y=table['Landslide']

TestSize=0.33
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TestSize,random_state=7)


kfold=sklearn.model_selection.KFold(n_splits=10,shuffle=True,random_state=7)
type(kfold)
kfold2=sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
print(kfold2)
test_index=0
for train_index,test_index in kfold2.split(X,y):
    i=0
print(test_index)
"""for train_index, test_index in kfold2.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]"""

dtree=DecisionTreeClassifier(max_depth=5)

trening = dtree.fit(X_train, y_train)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
trening
plt.figure(figsize=(20,20))
sklearn.tree.plot_tree(trening)
results = dtree.score(X_test, y_test)
print("Accuracy for 33% test split: ", results)

from sklearn.model_selection import cross_val_score

results=cross_val_score(dtree,X,y,cv=kfold)
print("CV:")
from numpy import mean
print(results.mean())



"""graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
print("CHECK")
img = pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
"""




